# Source Code Documentation

This directory contains all source code for the framework.

## Structure
```
src/
├── README.md              # This file
├── models/                # Model implementations
│   └── README.md
├── losses/                # Loss functions
│   └── README.md
├── data/                  # Data loaders
│   └── README.md
├── runners/               # Training/inference orchestration
│   └── README.md
└── utils/                 # Utilities
    └── README.md
```

## Component Overview

### Models (`src/models/`)
Neural network architectures (VAE, SAE, custom models).

**Key concepts:**
- Models receive batch dictionaries
- Models return output dictionaries with `'predictions'` key
- Models implement `extract_embeddings()` for embedding extraction
- Registry pattern: `@register_model("ModelName")`

**See:** `models/README.md` for interface details.

---

### Losses (`src/losses/`)
Loss functions for training.

**Key concepts:**
- Functions (not classes)
- Receive `model_output` dict, `targets` dict, and `**kwargs`
- Return scalar loss tensor
- Registry pattern: `@register_loss("LossName")`

**See:** `losses/README.md` for interface details.

---

### Data Loaders (`src/data/`)
Data loading and batching.

**Key concepts:**
- Load data from files (CSV, .npy, etc.)
- Return standardized batch dictionaries
- Handle numerical, categorical, and embedding features
- Registry pattern: `@register_dataloader("LoaderName")`

**See:** `data/README.md` for interface details.

---

### Runners (`src/runners/`)
Orchestrate training, validation, prediction, and embedding extraction.

**Key concepts:**
- `TrainRunner`: Training loop with validation
- `EmbeddingRunner`: Extract embeddings from trained models
- `PredictRunner`: Run inference on test data

**See:** `runners/README.md` for details.

---

### Utils (`src/utils/`)
Utility functions for config loading, checkpointing, logging, MLflow integration.

**See:** `utils/README.md` for available utilities.

---

## Adding New Components

### Adding a New Model

1. Create `src/models/my_model.py`
2. Define class inheriting from `nn.Module`
3. Register with `@register_model("MyModel")`
4. Implement `forward()` and `extract_embeddings()`
5. Import in `src/models/__init__.py`

**See:** `models/README.md` for full interface requirements.

---

### Adding a New Loss

1. Create `src/losses/my_loss.py`
2. Define function with signature: `def my_loss(model_output, targets, **kwargs)`
3. Register with `@register_loss("MyLoss")`
4. Import in `src/losses/__init__.py`

**See:** `losses/README.md` for full interface requirements.

---

### Adding a New Data Loader

1. Create `src/data/my_loader.py`
2. Define class with `__init__()`, `__iter__()`, `__len__()`
3. Register with `@register_dataloader("MyLoader")`
4. Return standardized batch dictionaries
5. Import in `src/data/__init__.py`

**See:** `data/README.md` for full interface requirements.

---

## Design Principles

### Registry Pattern
All pluggable components use registries:
```python
@register_model("VAE")
class VAE(nn.Module):
    ...

# In config
model:
  class: "VAE"
```

### Dictionary-Based Communication
Components communicate via dictionaries for flexibility:

**Batch format:**
```python
{
    'numerical': Tensor,
    'categorical': Tensor,
    'embeddings': Tensor,
    'targets': {...},
    'metadata': {...}
}
```

**Model output format:**
```python
{
    'predictions': Tensor,  # Required
    'mu': Tensor,           # Optional
    'logvar': Tensor,       # Optional
    ...
}
```

### Clear Interfaces
Each component has a well-defined interface:
- Models: Receive batches, return outputs
- Losses: Receive model outputs and targets, return scalar
- Data loaders: Yield standardized batches
- Runners: Orchestrate the workflow

---

## Code Organization

### Import Structure
```python
# In scripts or notebooks
from src.models import MODELS
from src.losses import LOSSES
from src.data import DATA_LOADERS
from src.runners import TrainRunner, EmbeddingRunner, PredictRunner
```

### Registry Access
```python
# Get registered components
model_class = MODELS["VAE"]
loss_fn = LOSSES["VAELoss"]
loader_class = DATA_LOADERS["TabularDataLoader"]

# Instantiate
model = model_class(**model_params)
loader = loader_class(**loader_params)
```

---

## Testing Components

### Test a Model
```python
import torch
from src.models import MODELS

# Create model
model = MODELS["VAE"](latent_dim=32, hidden_dims=[128, 64])

# Create dummy batch
batch = {
    'numerical': torch.randn(16, 10),
    'categorical': torch.randint(0, 5, (16, 3)),
    'embeddings': None
}

# Test forward pass
output = model(batch)
print(output.keys())  # Should include 'predictions'

# Test embedding extraction
embeddings = model.extract_embeddings(batch)
print(embeddings.shape)
```

### Test a Loss
```python
import torch
from src.losses import LOSSES

# Get loss function
loss_fn = LOSSES["VAELoss"]

# Create dummy data
model_output = {
    'predictions': torch.randn(16, 10),
    'mu': torch.randn(16, 32),
    'logvar': torch.randn(16, 32)
}
targets = {'predictions': torch.randn(16, 10)}

# Test loss computation
loss = loss_fn(model_output, targets, beta=1.0, reconstruction_weight=1.0)
print(loss.item())
```

### Test a Data Loader
```python
from src.data import DATA_LOADERS

# Create loader
loader = DATA_LOADERS["TabularDataLoader"](
    data_path="./data/train.csv",
    batch_size=32,
    columns={
        'numerical': ['feat1', 'feat2'],
        'categorical': ['cat1'],
        'embeddings': []
    },
    shuffle=False
)

# Test iteration
batch = next(iter(loader))
print(batch.keys())  # Should have 'numerical', 'categorical', 'targets'
```

---

## Development Workflow

1. **Add new component** (model, loss, or loader)
2. **Test in isolation** (see testing examples above)
3. **Register component** (with decorator)
4. **Import in `__init__.py`**
5. **Create config** (reference new component)
6. **Run experiment** (with scripts)

---

## Best Practices

### Models
- Always return dictionary with `'predictions'` key
- Implement `extract_embeddings()` even if not immediately needed
- Document expected input structure
- Handle missing feature types gracefully (numerical/categorical/embeddings)

### Losses
- Use `**kwargs` for flexible parameter passing
- Document required keys in `model_output`
- Return scalar tensor (not Python float)
- Add docstring with parameter descriptions

### Data Loaders
- Return consistent batch format
- Handle edge cases (last batch, empty columns)
- Use efficient data loading (num_workers, preprocessing)
- Document expected data format (CSV structure, .npy shape, etc.)

### Runners
- Log informative messages
- Handle errors gracefully
- Save checkpoints atomically (avoid corruption)
- Validate config before starting