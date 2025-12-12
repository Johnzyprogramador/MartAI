# Models Documentation

Models define neural network architectures.

## Interface Requirements

All models must:

1. Inherit from `nn.Module`
2. Be registered with `@register_model(name)`
3. Implement `forward()` returning a dictionary
4. Implement `extract_embeddings()` for embedding extraction

---

## Interface
```python
import torch.nn as nn
from src.models.registry import register_model

@register_model("MyModel")
class MyModel(nn.Module):
    def __init__(self, **params):
        """
        Initialize model.
        
        Args:
            **params: Model-specific parameters from config
        """
        super().__init__()
        # Initialize layers
    
    def forward(self, batch: dict) -> dict:
        """
        Forward pass.
        
        Args:
            batch: Dictionary with keys 'numerical', 'categorical', 'embeddings'
                   (use whichever your model needs)
        
        Returns:
            Dictionary with at minimum:
            {
                'predictions': torch.Tensor,  # REQUIRED
                # ... any other model-specific outputs
            }
        """
        pass
    
    def extract_embeddings(self, batch: dict) -> torch.Tensor:
        """
        Extract embeddings from a specific layer.
        
        Args:
            batch: Same format as forward()
        
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        pass
```

---

## Input Format

Models receive batch dictionaries:
```python
batch = {
    'numerical': torch.FloatTensor | None,
    'categorical': torch.LongTensor | None,
    'embeddings': torch.FloatTensor | None,
    'targets': {...},
    'metadata': {...}
}
```

Models decide which keys to use:
- VAE might use `numerical` and `categorical`
- SAE might only use `embeddings`

---

## Output Format

Models must return a dictionary with at least `'predictions'`:
```python
{
    'predictions': torch.Tensor,  # REQUIRED: what gets compared to targets
    'mu': torch.Tensor,           # Optional: model-specific
    'logvar': torch.Tensor,       # Optional: model-specific
    # ... any other outputs
}
```

The `'predictions'` key is mandatory - this is what the loss function compares to targets.

---

## Example Implementation
```python
from src.models.registry import register_model
import torch
import torch.nn as nn

@register_model("VAE")
class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dims):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Build encoder, decoder
        self.encoder = nn.Sequential(...)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = nn.Sequential(...)
    
    def forward(self, batch):
        # Extract features from batch
        features = []
        if batch['numerical'] is not None:
            features.append(batch['numerical'])
        if batch['categorical'] is not None:
            # Embed categorical features
            cat_embedded = self.embedding_layer(batch['categorical'])
            features.append(cat_embedded)
        
        x = torch.cat(features, dim=-1)
        
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return {
            'predictions': x_recon,
            'mu': mu,
            'logvar': logvar
        }
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def extract_embeddings(self, batch):
        # Extract features
        features = []
        if batch['numerical'] is not None:
            features.append(batch['numerical'])
        if batch['categorical'] is not None:
            cat_embedded = self.embedding_layer(batch['categorical'])
            features.append(cat_embedded)
        
        x = torch.cat(features, dim=-1)
        
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        
        return mu  # Return latent mean as embeddings
```

---

## Available Models

### `VAE`
Variational Autoencoder for tabular data.

**Config:**
```yaml
model:
  class: "VAE"
  params:
    latent_dim: 32
    hidden_dims: [256, 128]
```

### `SAE`
Sparse Autoencoder for embeddings.

**Config:**
```yaml
model:
  class: "SAE"
  params:
    input_dim: 32
    hidden_dim: 16
    sparsity_weight: 0.01
```

---

## Adding a New Model

1. Create `src/models/my_model.py`
2. Implement the interface above
3. Register with `@register_model("MyModel")`
4. Import in `src/models/__init__.py`
5. Use in config: `model.class: "MyModel"`

**See existing implementations** in `vae.py` and `sae.py` for reference.

---

## Key Points

### Handling Different Feature Types
```python
def forward(self, batch):
    features = []
    
    if batch['numerical'] is not None:
        features.append(batch['numerical'])
    
    if batch['categorical'] is not None:
        # Embed categorical features first
        cat_embedded = self.embedding_layer(batch['categorical'])
        features.append(cat_embedded)
    
    if batch['embeddings'] is not None:
        features.append(batch['embeddings'])
    
    x = torch.cat(features, dim=-1)
    # ... rest of model
```

### Extract Embeddings

The `extract_embeddings()` method should return embeddings from the layer you want to extract:
- VAE: typically the latent mean (`mu`)
- SAE: typically the hidden layer activations
- Custom models: whatever makes sense

---

## Registry
```python
# src/models/registry.py
MODELS = {}

def register_model(name):
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator
```

Access in code:
```python
from src.models import MODELS
model = MODELS["VAE"](**params)
```