from franken.data.base import Configuration, Target

__all__ = [
    "Configuration",
    "Target",
    "FrankenAtomsDataset",
]

def __getattr__(name):
    if name == "FrankenAtomsDataset":
        from franken.data.dataset import FrankenAtomsDataset
        return FrankenAtomsDataset
    raise AttributeError(name)