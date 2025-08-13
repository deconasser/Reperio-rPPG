from .corr import NegPearson
from .corr import MSELoss

def get_loss_cls(name: str):
    name = name.lower()
    if name == 'negpearson':
        return NegPearson
    if name == 'mse':
        return MSELoss
    raise ValueError(f'Unknown loss: {name}')