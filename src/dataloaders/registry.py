"""Data loader registry for dynamic loader loading."""

DATA_LOADERS = {}

def register_dataloader(name):
    """
    Register a data loader class.
    
    Usage:
        @register_dataloader("TabularDataLoader")
        class TabularDataLoader:
            ...
    """
    def decorator(cls):
        DATA_LOADERS[name] = cls
        return cls
    return decorator