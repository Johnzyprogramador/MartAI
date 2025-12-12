"""Model registry for dynamic model loading."""

MODELS = {}

def register_model(name):
    """
    Register a model class.
    
    Usage:
        @register_model("VAE")
        class VAE(nn.Module):
            ...
    """
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator