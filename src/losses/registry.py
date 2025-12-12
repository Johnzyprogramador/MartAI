"""Loss function registry for dynamic loss loading."""

LOSSES = {}

def register_loss(name):
    """
    Register a loss function.
    
    Usage:
        @register_loss("VAELoss")
        def vae_loss(model_output, targets, **kwargs):
            ...
    """
    def decorator(fn):
        LOSSES[name] = fn
        return fn
    return decorator