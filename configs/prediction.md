# Prediction Configuration Reference

Configuration for `scripts/predict.py`.

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

test:
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: bool
    num_workers: int
  output: str
  
  metrics:  # Optional
    compute: bool
    loss:
      class: str
      params: dict
```

## Field Reference

### `checkpoint`
- `path`: Path to checkpoint file
- `load_strategy`: `"best"`, `"last"`, or `"epoch_N"`

### `model`
- `class`: Must match the model used in training

### `test`
- `data_path`: Path to test data
- `output`: Path where predictions will be saved (`.npy` file)
- `shuffle`: Should be `false` for consistent ordering

### `test.metrics` (optional)
- `compute`: If `true`, compute and log metrics
- `loss`: Loss function to use for metrics

## Example 1: Predictions Without Metrics
```yaml
experiment:
  name: "predict"
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
  output: "./experiments/vae/baseline/predictions/test_predictions.npy"
```

## Example 2: Predictions With Metrics
```yaml
experiment:
  name: "predict_with_metrics"
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
  output: "./experiments/vae/baseline/predictions/test_predictions.npy"
  
  metrics:
    compute: true
    loss:
      class: "VAELoss"
      params:
        beta: 1.0
        reconstruction_weight: 1.0
```

## Output Format

Predictions are saved as NumPy `.npy` files containing the full model output dictionary.

Load with:
```python
import numpy as np
predictions = np.load("path/to/predictions.npy", allow_pickle=True)
```

## Tips

- Use larger `batch_size` for faster inference
- Always use `shuffle: false` for consistent ordering
- Metrics are logged but not saved to file