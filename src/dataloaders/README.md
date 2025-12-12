# Data Loaders Documentation

Data loaders load data from files and return standardized batches.

## Interface Requirements

All data loaders must:

1. Be registered with `@register_dataloader(name)`
2. Implement `__init__()`, `__iter__()`, `__len__()`
3. Return batches in the standard format

---

## Standard Batch Format
```python
{
    'numerical': torch.FloatTensor | None,
    'categorical': torch.LongTensor | None,
    'embeddings': torch.FloatTensor | None,
    'targets': {
        'numerical': torch.FloatTensor | None,
        'categorical': torch.LongTensor | None,
        'embeddings': torch.FloatTensor | None
    },
    'metadata': dict | None  # Optional
}
```

**Key points:**
- At least one of `numerical`, `categorical`, `embeddings` must be non-None
- For self-supervised: `targets` references the **same tensors** as features (memory efficient)
- For supervised: `targets` contains different data

---

## Interface
```python
@register_dataloader("MyLoader")
class MyLoader:
    def __init__(self, data_path: str, batch_size: int, columns: dict,
                 cardinalities_path: str = None, shuffle: bool = False,
                 num_workers: int = 0, num_batches_per_epoch: int = None,
                 **kwargs):
        """
        Args:
            data_path: Path to data file
            batch_size: Batch size
            columns: Dict with 'numerical', 'categorical', 'embeddings' keys
            cardinalities_path: Path to categorical mappings JSON
            shuffle: Whether to shuffle
            num_workers: Number of workers
            num_batches_per_epoch: Limit batches (None = all)
        """
        pass
    
    def __iter__(self):
        """Yield batches in standard format"""
        pass
    
    def __len__(self) -> int:
        """Return number of batches"""
        pass
```

---

## Available Loaders

### `TabularDataLoader`
Loads CSV/Parquet files with numerical, categorical, and embedding columns.
```yaml
loader_class: "TabularDataLoader"
loader_params:
  batch_size: 128
  shuffle: true
```

### `EmbeddingDataLoader`
Loads `.npy` embedding files (for SAE training).
```yaml
loader_class: "EmbeddingDataLoader"
loader_params:
  batch_size: 128
  shuffle: true
```

---

## Adding a New Loader

1. Create `src/data/my_loader.py`
2. Implement the interface above
3. Register with `@register_dataloader("MyLoader")`
4. Import in `src/data/__init__.py`
5. Use in config: `loader_class: "MyLoader"`

**See existing implementations** in `tabular_loader.py` and `embedding_loader.py` for reference.

---

## Data Types

- Numerical: `torch.float32`
- Categorical: `torch.int64`
- Embeddings: `torch.float32`

---

## Registry
```python
# src/data/registry.py
DATA_LOADERS = {}

def register_dataloader(name):
    def decorator(cls):
        DATA_LOADERS[name] = cls
        return cls
    return decorator
```

Access in code:
```python
from src.data import DATA_LOADERS
loader = DATA_LOADERS["TabularDataLoader"](**params)
```