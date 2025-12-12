# Losses Documentation

Loss functions compute scalar loss values for training.

## Interface Requirements

All loss functions must:

1. Be functions (not classes)
2. Be registered with `@register_loss(name)`
3. Accept `model_output`, `targets`, and `**kwargs`
4. Return a scalar tensor

---

## Interface
```python
from src.losses.registry import register_loss
import torch

@register_loss("MyLoss")
def my_loss(model_output: dict, targets: dict, **kwargs) -> torch.Tensor:
    """
    Compute loss.
    
    Args:
        model_output: Dictionary from model.forward()
        targets: Dictionary from batch['targets']
        **kwargs: Loss-specific parameters from config
    
    Returns:
        Scalar loss tensor
    """
    # Compute loss
    loss = ...
    return loss
```

---

## Input Format

### `model_output`
Dictionary returned by model's `forward()`:
```python
{
    'predictions': torch.Tensor,  # Required
    'mu': torch.Tensor,           # Optional, model-specific
    'logvar': torch.Tensor,       # Optional, model-specific
    # ...
}
```

### `targets`
Dictionary from batch:
```python
{
    'numerical': torch.Tensor | None,
    'categorical': torch.Tensor | None,
    'embeddings': torch.Tensor | None
}
```

### `**kwargs`
Parameters from config:
```yaml
loss:
  class: "MyLoss"
  params:
    param1: value1
    param2: value2
```

These are passed as `**kwargs` to the loss function.

---

## Example Implementation
```python
from src.losses.registry import register_loss
import torch
import torch.nn.functional as F

@register_loss("VAELoss")
def vae_loss(model_output, targets, beta=1.0, reconstruction_weight=1.0, **kwargs):
    """
    VAE loss = reconstruction loss + beta * KL divergence
    
    Args:
        model_output: Dict with 'predictions', 'mu', 'logvar'
        targets: Dict with target values
        beta: Weight for KL divergence term
        reconstruction_weight: Weight for reconstruction term
        **kwargs: Additional unused parameters
    
    Returns:
        Scalar loss
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(model_output['predictions'], targets['predictions'])
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(
        1 + model_output['logvar'] 
        - model_output['mu'].pow(2) 
        - model_output['logvar'].exp()
    )
    kl_loss = kl_loss / model_output['mu'].size(0)  # Average over batch
    
    # Total loss
    return reconstruction_weight * recon_loss + beta * kl_loss
```

---

## Available Losses

### `VAELoss`
VAE loss with reconstruction and KL divergence.

**Config:**
```yaml
loss:
  class: "VAELoss"
  params:
    beta: 1.0
    reconstruction_weight: 1.0
```

### `SAELoss`
Sparse autoencoder loss with sparsity penalty.

**Config:**
```yaml
loss:
  class: "SAELoss"
  params:
    sparsity_weight: 0.01
```

---

## Adding a New Loss

1. Create `src/losses/my_loss.py`
2. Implement the interface above
3. Register with `@register_loss("MyLoss")`
4. Import in `src/losses/__init__.py`
5. Use in config: `loss.class: "MyLoss"`

**See existing implementations** in `vae_loss.py` and `sae_loss.py` for reference.

---

## Key Points

### Parameters via kwargs

Loss parameters come from config:
```yaml
loss:
  params:
    beta: 1.0
    weight: 0.5
```

Accessed in function:
```python
def my_loss(model_output, targets, beta=1.0, weight=0.5, **kwargs):
    # beta and weight are available here
    pass
```

### Return Scalar

Always return a scalar tensor:
```python
# Good
return loss  # tensor(0.5432)

# Bad
return loss.item()  # Python float (breaks autograd)
```

### Different Params for Train/Val

Config can specify different params:
```yaml
train:
  loss:
    class: "VAELoss"
    params:
      beta: 1.0

val:
  loss:
    class: "VAELoss"
    params:
      beta: 0.5  # Different!
```

---

## Registry
```python
# src/losses/registry.py
LOSSES = {}

def register_loss(name):
    def decorator(fn):
        LOSSES[name] = fn
        return fn
    return decorator
```

Access in code:
```python
from src.losses import LOSSES
loss_fn = LOSSES["VAELoss"]
loss = loss_fn(model_output, targets, beta=1.0)
```