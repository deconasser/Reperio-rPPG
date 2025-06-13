import torch
import math
import random
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur
from torch.utils import checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
from einops import rearrange
from functools import reduce
from operator import mul
from scipy import signal
from torch_geometric.nn import RGCNConv, TransformerConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_activation_module(activation):
    if isinstance(activation, str):
        if activation == "relu" or activation == 'reglu':
            return nn.ReLU()
        elif activation == "gelu" or activation == 'geglu':
            return nn.GELU()
        elif activation == "swish" or activation == 'swiglu':
            return nn.SiLU()
        else:
            raise ValueError(f"activation={activation} is not supported.")
    elif callable(activation):
        return activation()
    else:
        raise ValueError(f"activation={activation} is not supported.")


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, post_norm=None):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_t, self.max_h, self.max_w = max_seq_len

        self.emb_t = nn.Embedding(self.max_t, dim)
        self.emb_h = nn.Embedding(self.max_h, dim)
        self.emb_w = nn.Embedding(self.max_w, dim)

        self.post_norm = post_norm(dim) if callable(post_norm) else None

    def forward(self, x):
        t, h, w = x.shape[-3:]

        pos_t = torch.arange(t, device=x.device)
        pos_h = torch.arange(h, device=x.device)
        pos_w = torch.arange(w, device=x.device)

        pos_emb_t = self.emb_t(pos_t)
        pos_emb_h = self.emb_h(pos_h)
        pos_emb_w = self.emb_w(pos_w)

        pos_emb_t = rearrange(pos_emb_t, 't d -> d t 1 1') * self.scale
        pos_emb_h = rearrange(pos_emb_h, 'h d -> d 1 h 1') * self.scale
        pos_emb_w = rearrange(pos_emb_w, 'w d -> d 1 1 w') * self.scale

        x = x + pos_emb_t + pos_emb_h + pos_emb_w

        if self.post_norm is not None:
            x = rearrange(x, 'b c t h w -> b t h w c')
            x = self.post_norm(x)
            x = rearrange(x, 'b t h w c -> b c t h w')

        return x


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='swish', drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if 'glu' in act_layer:
            self.fc1 = GLU(in_features, hidden_features, get_activation_module(act_layer))
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                get_activation_module(act_layer)
            )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FrameEncoder(nn.Module):
    def __init__(self, in_channels=192, out_dim=128):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_dim, kernel_size=(1, 8, 8), bias=False)
        self.bn3d   = nn.BatchNorm3d(out_dim) 

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, 1, H, W)
        x = self.conv3d(x)
        x = self.bn3d(x)
        x = x.view(B, T, -1)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(1,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, batch_size=8, frame_len=8):
        B_, N, C = x.shape  
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        scale = 1.0 / math.sqrt(C // self.num_heads)
        scores = torch.einsum("b h n c, b h m c -> b h n m", q, k) * scale

        attn = F.softmax(scores, dim=-1)
        if self.training:
            attn = F.dropout(attn, p=self.attn_drop.p)

        self.last_attn = attn

        attn_output = torch.einsum("b h n m, b h m c -> b h n c", attn, v)
        attn_output = attn_output.transpose(1, 2).reshape(B_, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x



class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(1,4,4), shift_size=(0,0,0), mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer='swish', norm_layer=nn.LayerNorm,
                 use_checkpoint=False, shift=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.shift = shift
        self.norm1 = norm_layer(dim)

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, batch_size=B, frame_len=D)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, return_attention=False):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x)
        else:
            x = self.forward_part1(x)

        attn_map = None
        if return_attention:
            attn_map = self.attn.last_attn

        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        if return_attention:
            return x, attn_map

        return x


class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, n_heads=4):
        super(GNN, self).__init__()
        self.g_dim = g_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.pooling = nn.Linear(h2_dim * n_heads, self.g_dim)
        self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)
        self.conv2 = TransformerConv(h1_dim, h2_dim, heads=n_heads, concat=True)
        self.bn = nn.BatchNorm1d(h2_dim * n_heads)
        self.last_attn = None

    def forward(self, node_features, edge_index, edge_type, return_attention=False):
        x = self.conv1(node_features, edge_index, edge_type)

        if return_attention:
            x, attn_weights = self.conv2(x, edge_index, return_attention_weights=True)
            self.last_attn = attn_weights
        else:
            x = self.conv2(x, edge_index)
        x = self.bn(x)
        x = F.leaky_relu(x)
        return x

class RelationalTemporalGCN(nn.Module):
    def __init__(
        self,
        rt_num_layers=2,
        g_dim=128,
        h1_dim=128,
        h2_dim=128,
        n_heads=6,
        past_window=5,
        future_window=5,
        periodic_window=[15],
        device='cpu'
    ):
        super(RelationalTemporalGCN, self).__init__()
        self.device = device

        self.edge_type_to_idx = {
            'self-loop': 0,
            'past': 1,
            'future': 2,
            'periodic': 3
        }
        self.num_relations = len(self.edge_type_to_idx)

        # Only GNN layers (no LayerNorm)
        self.layers = nn.ModuleList([
            GNN(
                g_dim,
                h1_dim,
                h2_dim,
                num_relations=self.num_relations,
                n_heads=n_heads
            )
            for _ in range(rt_num_layers)
        ])

        self.frame_encoder = FrameEncoder(in_channels=192, out_dim=g_dim)
        self.past_window = past_window
        self.future_window = future_window
        self.periodic_window = periodic_window

    def forward(self, x):
        """
        x: [b, d, h, w, c]  -> returns [b, d, g_dim]
        """
        b, d, h, w, c = x.shape
        all_outputs = []

        for i in range(b):
            # Encode frame → node_features: [d, g_dim]
            node_features = self.frame_encoder(
                x[i].permute(0, 3, 1, 2).unsqueeze(0)
            ).squeeze(0)

            # Build graph edges
            edge_index, edge_type = self._build_edges(d)

            # Apply GNN layers sequentially
            for layer in self.layers:
                node_features = layer(node_features, edge_index, edge_type)

            all_outputs.append(node_features.unsqueeze(0))  # [1, d, g_dim]

        return torch.cat(all_outputs, dim=0)  # [b, d, g_dim]

    def _build_edges(self, d):
        """
        Build graph edges:
          - Self-loop: t -> t
          - Past: t -> t-1
          - Future: t -> t+1
          - Periodic forward/backward
        Returns:
          - edge_index: [2, E]
          - edge_type: [E]
        """
        edge_index_list = []
        edge_type_list = []

        for t in range(d):
            # Self-loop
            edge_index_list.append([t, t])
            edge_type_list.append(self.edge_type_to_idx['self-loop'])

            # Past connections
            for offset in range(1, self.past_window + 1):
                prev = t - offset
                if prev >= 0:
                    edge_index_list.append([t, prev])
                    edge_type_list.append(self.edge_type_to_idx['past'])

            # Future connections
            for offset in range(1, self.future_window + 1):
                nxt = t + offset
                if nxt < d:
                    edge_index_list.append([t, nxt])
                    edge_type_list.append(self.edge_type_to_idx['future'])

            # Periodic forward & backward
            for offset in self.periodic_window:
                nxt = t + offset
                if nxt < d:
                    edge_index_list.append([t, nxt])
                    edge_type_list.append(self.edge_type_to_idx['periodic'])
                prev = t - offset
                if prev >= 0:
                    edge_index_list.append([t, prev])
                    edge_type_list.append(self.edge_type_to_idx['periodic'])

        edge_index = torch.LongTensor(edge_index_list).T.to(self.device)
        edge_type = torch.LongTensor(edge_type_list).to(self.device)
        return edge_index, edge_type




class ReperioBasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 act_layer,
                 depth,
                 num_heads,
                 window_size=(1,5,5),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 g_dim=128,
                 h1_dim=100,
                 h2_dim=100,
                 rt_num_layers=4,
                 rtgraph_heads=7,
                 past_window=5,
                 future_window=5,
                 periodic_window=[15],
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shift_size = tuple(i // 2 for i in window_size)
        # Swin Transformer Block
        self.spatial_blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                shift = True,
            )
            for i in range(depth)])

        # Temporal Graph
        self.temporal_graph = RelationalTemporalGCN(rt_num_layers=rt_num_layers,g_dim=g_dim, h1_dim=h1_dim, h2_dim=h2_dim, n_heads=rtgraph_heads, past_window=past_window, future_window=future_window, periodic_window=periodic_window, device=device)

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        self.ln = norm_layer(dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        # Spatial Modeling by Swin Transformer
        for i, block in enumerate(self.spatial_blocks):
            x = block(x)

        # Temporal Modeling
        x = self.temporal_graph(x) # torch.Size([Batch_size, 180, h2*rt_number_heads])
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class Reperio(nn.Module):
    def __init__(
        self,
        patch_size=(1,16,16),
        input_resolution=128,
        depth=12,
        embed_dim=192,
        act_layer='swish',
        num_heads=8,
        window_size=(1,4,4),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        post_pos_norm=True,
        use_checkpoint=False,
        g_dim=128,
        h1_dim=100,
        h2_dim=100,
        rt_num_layers=1,
        rtgraph_heads=7,
        past_window=1,
        future_window=1,
        periodic_window=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        **kwargs,
    ):
        super().__init__()

        self.input_resolution = input_resolution
        self.num_layers = depth
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.rtgraph_heads = rtgraph_heads
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.g_dim = g_dim
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=9, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_embed = AbsolutePositionalEmbedding(
            embed_dim,
            max_seq_len=(1800, math.ceil(input_resolution/patch_size[1]), math.ceil(input_resolution/patch_size[2])),
            post_norm=norm_layer if post_pos_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.layers = ReperioBasicLayer(
            dim=embed_dim,
            act_layer=act_layer,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            downsample=None,
            g_dim=g_dim,
            h1_dim=h1_dim,
            h2_dim=h2_dim,
            rt_num_layers=rt_num_layers,
            rtgraph_heads=rtgraph_heads,
            past_window=past_window,
            future_window=future_window,
            periodic_window=periodic_window,
            use_checkpoint=use_checkpoint,
        )

        self.norm = norm_layer(embed_dim)
        self.out_pooling = nn.Linear(embed_dim, 1)
        self.out_fc = MLP(embed_dim, embed_dim, 1, act_layer)

        self.bn = nn.BatchNorm3d(9)
        self.register_buffer('rot', torch.tensor([[0, 1, -1], [-2, 1, 1]], dtype=torch.float))

        self.final_linear = nn.Linear(self.h2_dim * self.rtgraph_heads, 1)
        self.apply(_init_weights)

    def preprocess(self, x):

        x = rearrange(x, 'n d c h w -> n c d h w')

        N, C, D, H, W = x.shape

        # MPOS
        x_temp = x / x.mean(dim=2, keepdim=True)
        x_temp = rearrange(x_temp, 'n c d h w -> n h w c d')
        x_proj = torch.matmul(self.rot, x_temp)
        x_proj = rearrange(x_proj, 'n h w c d -> (n d) c h w')

        x_mpos = []
        for i in range(3):
            x_filt = gaussian_blur(x_proj, kernel_size=int(self.input_resolution / (2**(i+2))-1))
            x_filt = rearrange(x_filt, '(n d) c h w -> n c d h w', n=N)
            s0 = x_filt[:, :1]
            s1 = x_filt[:, 1:]
            mpos = s0 + (s0.std(dim=2, keepdim=True) / s1.std(dim=2, keepdim=True)) * s1
            mpos = (mpos - mpos.mean(dim=2, keepdim=True)) / mpos.std(dim=2, keepdim=True)
            x_mpos.append(mpos)

        # Raw
        x_norm = (x - 0.5) * 2

        # NDF
        x_diff = x.clone()
        x_diff[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        x_diff[:, :, :-1] = x_diff[:, :, :-1] / (x[:, :, 1:] + x[:, :, :-1])
        torch.nan_to_num_(x_diff, nan=0., posinf=0., neginf=0.)
        x_diff[:, :, :-1] = x_diff[:, :, :-1] / x_diff[:, :, :-1].std(dim=(1, 2, 3, 4), keepdim=True)
        x_diff[:, :, -1:].fill_(0.)

        x = torch.cat([x_norm, x_diff] + x_mpos, dim=1)

        torch.nan_to_num_(x, nan=0., posinf=0., neginf=0.)

        x = self.bn(x)

        return x

    def forward(self, x):
        """Forward function."""

        x = self.preprocess(x)

        x = self.patch_embed(x)

        x = self.pos_drop(self.pos_embed(x))

        x = self.layers(x) 

        x = self.final_linear(x)

        x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

        return x

    @torch.no_grad()
    def predict(self, x):
        return self(x)

def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)