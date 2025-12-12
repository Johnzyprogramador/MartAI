# Embedding Extraction Configuration Reference

Configuration for `scripts/extract_embeddings.py`.

## Complete Structure
```yaml
experiment:
  name: str
  output_dir: str
  device: str

checkpoint:
  path: str
  load_strategy: str

model:
  class: str

data:
  columns:
    numerical: list[str]
    categorical: list[str]
    embeddings: list[str]
  cardinalities_path: str

# Include only the splits you want to extract
train:  # Optional
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: bool
    num_workers: int
  output: str

val:  # Optional
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: bool
    num_workers: int
  output: str

test:  # Optional
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: bool
    num_workers: int
  output: str
```

## Field Reference

### `checkpoint`
- `path`: Path to checkpoint file (e.g., `./experiments/vae/baseline/checkpoints/best_model.pt`)
- `load_strategy`: `"best"`, `"last"`, or `"epoch_N"` (where N is epoch number)

### `model`
- `class`: Must match the model used in training
- `params`: Not needed (loaded from checkpoint)

### `data`
- Same as training config
- Must match the data configuration used during training

### Split sections (`train`/`val`/`test`)
- Include only the splits you want to extract
- `shuffle`: Always `false` for extraction
- `output`: Path where embeddings will be saved (`.npy` file)

**Note:** `num_batches_per_epoch` is NOT supported for embedding extraction (always uses full dataset).

## Example 1: Extract All Splits
```yaml
experiment:
  name: "extract_embeddings"
  output_dir: "./experiments/vae/baseline"
  device: "cuda"

checkpoint:
  path: "./experiments/vae/baseline/checkpoints/best_model.pt"
  load_strategy: "best"

model:
  class: "VAE"

data:
  columns:
    numerical:
      - log_amount_normalized
      - distance_km
    categorical:
      - channel
    embeddings: []
  cardinalities_path: "./data/cardinalities.json"

train:
  data_path: "./data/train.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 512
    shuffle: false
    num_workers: 4
  output: "./experiments/vae/baseline/embeddings/train_embeddings.npy"

val:
  data_path: "./data/val.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 512
    shuffle: false
    num_workers: 4
  output: "./experiments/vae/baseline/embeddings/val_embeddings.npy"

test:
  data_path: "./data/test.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 512
    shuffle: false
    num_workers: 4
  output: "./experiments/vae/baseline/embeddings/test_embeddings.npy"
```

## Example 2: Extract Only Test Split
```yaml
experiment:
  name: "extract_test_only"
  output_dir: "./experiments/vae/baseline"
  device: "cuda"

checkpoint:
  path: "./experiments/vae/baseline/checkpoints/best_model.pt"
  load_strategy: "best"

model:
  class: "VAE"

data:
  columns:
    numerical:
      - log_amount_normalized
      - distance_km
    categorical:
      - channel
    embeddings: []
  cardinalities_path: "./data/cardinalities.json"

test:
  data_path: "./data/test.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 512
    shuffle: false
    num_workers: 4
  output: "./experiments/vae/baseline/embeddings/test_embeddings.npy"
```

## Example 3: Load Specific Epoch
```yaml
experiment:
  name: "extract_epoch_50"
  output_dir: "./experiments/vae/baseline"
  device: "cuda"

checkpoint:
  path: "./experiments/vae/baseline/checkpoints/"
  load_strategy: "epoch_50"  # Loads epoch_050.pt

model:
  class: "VAE"

data:
  columns:
    numerical:
      - log_amount_normalized
    categorical: []
    embeddings: []
  cardinalities_path: null

train:
  data_path: "./data/train.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 512
    shuffle: false
  output: "./experiments/vae/baseline/embeddings/train_embeddings_epoch50.npy"
```

## Output Format

Embeddings are saved as NumPy `.npy` files with shape `(n_samples, embedding_dim)`.

Load with:
```python
import numpy as np
embeddings = np.load("path/to/embeddings.npy")
```

## Tips

- Use larger `batch_size` for extraction (faster, no backward pass)
- Always use `shuffle: false` for consistent ordering
- Only include splits you need (saves time)
- Embeddings can be large - ensure sufficient disk space