# Training Configuration Reference

Configuration for `scripts/train.py`.

## Complete Structure
```yaml
experiment:
  name: str
  output_dir: str
  seed: int
  device: str

model:
  class: str
  params: dict

data:
  columns:
    numerical: list[str]
    categorical: list[str]
    embeddings: list[str]
  cardinalities_path: str

train:
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: bool
    num_workers: int
    num_batches_per_epoch: int | null
  epochs: int
  optimizer_config:
    optimizer: str
    learning_rate: float
    # ... additional optimizer params
  loss:
    class: str
    params: dict

val:  # Optional
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: bool
    num_workers: int
    num_batches_per_epoch: int | null
  frequency: int
  loss:
    class: str
    params: dict
  early_stopping:  # Optional
    patience: int
    metric: str
    direction: str

checkpoint:
  metric: str
  mode: str
  save_strategy: str

mlflow:  # Optional
  tracking_uri: str
  experiment_name: str
  credentials:
    username: str
    password: str
```

## Field Reference

### `experiment`
- `name`: Folder name for this experiment
- `output_dir`: Base directory (final path = `{output_dir}/{name}/`)
- `seed`: Random seed for reproducibility
- `device`: `"cuda"`, `"cpu"`, or `"cuda:0"`

### `model`
- `class`: Registered model name (e.g., `"VAE"`, `"SAE"`)
- `params`: Model-specific parameters passed to `__init__()`

### `data.columns`
- `numerical`: Numerical feature column names
- `categorical`: Categorical feature column names (requires `cardinalities_path`)
- `embeddings`: Embedding column names
- At least one must be non-empty

### `train.loader_params`
- `batch_size`: Number of samples per batch
- `shuffle`: Always `true` for training
- `num_workers`: Data loading workers (0 = main process)
- `num_batches_per_epoch`: Limit batches for debugging (`null` = use all)

### `train.optimizer_config`
- `optimizer`: PyTorch optimizer name (`"Adam"`, `"SGD"`, `"AdamW"`, etc.)
- `learning_rate`: Learning rate
- Additional params: Any valid optimizer parameter (e.g., `weight_decay`, `betas`, `momentum`)

### `train.loss`
- `class`: Registered loss function name
- `params`: Loss-specific parameters (passed as `**kwargs`)

### `val` (optional)
If omitted, no validation is performed.

- `frequency`: Validate every N epochs
- `shuffle`: Should be `false` for validation
- `loss`: Can have different params than training loss

### `val.early_stopping` (optional)
- `patience`: Stop after N validations without improvement
- `metric`: Exact metric name to monitor (e.g., `"val_loss"`)
- `direction`: `"min"` (lower is better) or `"max"` (higher is better)

### `checkpoint`
- `metric`: Metric for selecting best model
- `mode`: `"min"` or `"max"`
- `save_strategy`: `"best"`, `"last"`, or `"all"`

### `mlflow` (optional)
If omitted, no MLflow logging.

## Example 1: VAE with Validation
```yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/vae"
  seed: 42
  device: "cuda"

model:
  class: "VAE"
  params:
    latent_dim: 32
    hidden_dims: [256, 128]

data:
  columns:
    numerical:
      - log_amount_normalized
      - distance_km
    categorical:
      - channel
      - merchant_type
    embeddings: []
  cardinalities_path: "./data/cardinalities.json"

train:
  data_path: "./data/train.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 128
    shuffle: true
    num_workers: 4
    num_batches_per_epoch: null
  
  epochs: 100
  
  optimizer_config:
    optimizer: "Adam"
    learning_rate: 0.001
    weight_decay: 0.0001
  
  loss:
    class: "VAELoss"
    params:
      beta: 1.0
      reconstruction_weight: 1.0

val:
  data_path: "./data/val.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 256
    shuffle: false
    num_workers: 4
  
  frequency: 5
  
  loss:
    class: "VAELoss"
    params:
      beta: 1.0
      reconstruction_weight: 1.0
  
  early_stopping:
    patience: 20
    metric: "val_loss"
    direction: "min"

checkpoint:
  metric: "val_loss"
  mode: "min"
  save_strategy: "best"

mlflow:
  tracking_uri: "https://mlflow.example.com"
  experiment_name: "vae_experiments"
```

## Example 2: SAE on Embeddings
```yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/sae"
  seed: 42
  device: "cuda"

model:
  class: "SAE"
  params:
    input_dim: 32
    hidden_dim: 16
    sparsity_weight: 0.01

data:
  columns:
    numerical: []
    categorical: []
    embeddings:
      - vae_embeddings
  cardinalities_path: null

train:
  data_path: "./experiments/vae/baseline/embeddings/train_embeddings.npy"
  loader_class: "EmbeddingDataLoader"
  loader_params:
    batch_size: 128
    shuffle: true
  
  epochs: 50
  
  optimizer_config:
    optimizer: "Adam"
    learning_rate: 0.0001
  
  loss:
    class: "SAELoss"
    params:
      sparsity_weight: 0.01

val:
  data_path: "./experiments/vae/baseline/embeddings/val_embeddings.npy"
  loader_class: "EmbeddingDataLoader"
  loader_params:
    batch_size: 256
    shuffle: false
  
  frequency: 5
  
  loss:
    class: "SAELoss"
    params:
      sparsity_weight: 0.01

checkpoint:
  metric: "val_loss"
  mode: "min"
  save_strategy: "best"
```

## Example 3: Debug Configuration
```yaml
experiment:
  name: "debug"
  output_dir: "./experiments/vae"
  seed: 42
  device: "cuda"

model:
  class: "VAE"
  params:
    latent_dim: 32
    hidden_dims: [128]

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
    batch_size: 128
    shuffle: true
    num_batches_per_epoch: 10  # Only 10 batches!
  
  epochs: 3
  
  optimizer_config:
    optimizer: "Adam"
    learning_rate: 0.001
  
  loss:
    class: "VAELoss"
    params:
      beta: 1.0
      reconstruction_weight: 1.0

val:
  data_path: "./data/val.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 256
    shuffle: false
    num_batches_per_epoch: 5  # Only 5 batches!
  
  frequency: 1
  
  loss:
    class: "VAELoss"
    params:
      beta: 1.0
      reconstruction_weight: 1.0

checkpoint:
  metric: "val_loss"
  mode: "min"
  save_strategy: "last"
```

## Tips

- **Debugging**: Use `num_batches_per_epoch: 10` for quick testing
- **Optimizer params**: Any valid PyTorch optimizer parameter can be added to `optimizer_config`
- **Loss params**: Can differ between train and val
- **Early stopping metric**: Must match exact logged metric name
- **No validation**: Omit `val` section entirely, use `checkpoint.metric: "train_loss"`