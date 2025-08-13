from .reperio import Reperio
def get_model_cls(name: str):
    name = name.lower()
    if name == 'reperio':
        return Reperio
    raise ValueError(f'Unknown model: {name}')