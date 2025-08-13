import torch
from torch import nn


class NegPearson(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, predictions, ground_truths):
        predictions = predictions if predictions.ndim == 2 else predictions.squeeze(-1)
        ground_truths = ground_truths if ground_truths.ndim == 2 else ground_truths.squeeze(-1)
        pearson = self.cos(predictions - predictions.mean(dim=1, keepdim=True), ground_truths - ground_truths.mean(dim=1, keepdim=True))
        return torch.mean(-pearson + 1)
    

class MSELoss(nn.Module):
    """
    MSE loss (mean squared error) theo cùng định dạng đầu vào
    như các hàm loss trước đó.
    """
    def __init__(self, dim=1):
        """
        :param dim: Chiều mà ta coi là "chiều tín hiệu" (thường là T)
                    để tính trung bình hoặc làm các phép biến đổi.
        """
        super().__init__()
        self.dim = dim

    def forward(self, predictions, ground_truths):
        if predictions.ndim > 2:
            predictions = predictions.squeeze(-1)
        if ground_truths.ndim > 2:
            ground_truths = ground_truths.squeeze(-1)
        
        mse_per_sample = torch.mean((predictions - ground_truths) ** 2, dim=self.dim)
        return torch.mean(mse_per_sample)

