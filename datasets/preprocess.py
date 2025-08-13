from .mmpd import MMPDDataset
from .ucla import UCLADataset
from .rlap import RLAPDataset
from .pure import PUREDataset
from .ubfc import UBFCDataset

if __name__ == '__main__':
    #MMPDDataset('mmpd', 'Path/to/MMPD/dataset', 'Path/to/cache/directory', split='all', training=False, wave_type='normalized', img_height=128, img_width=128)
    #PUREDataset('pure', 'Path/to/PURE/dataset', 'Path/to/cache/directory', split='all', training=False, wave_type='normalized', img_height=128, img_width=128)
    #UBFCDataset('ubfc', 'Path/to/UBFC/dataset', 'Path/to/cache/directory', split='all', training=False, wave_type='normalized', img_height=128, img_width=128)