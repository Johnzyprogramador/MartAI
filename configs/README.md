# Configuration Documentation

All experiments are defined using YAML configuration files. This document explains every configuration option available.

## Table of Contents

- [Configuration Structure](#configuration-structure)
- [Experiment Section](#experiment-section)
- [Model Section](#model-section)
- [Data Section](#data-section)
- [Train Section](#train-section)
- [Val Section](#val-section)
- [Checkpoint Section](#checkpoint-section)
- [MLflow Section](#mlflow-section)
- [Complete Examples](#complete-examples)
- [Special Configurations](#special-configurations)

---

## Configuration Structure

Every config file has the following top-level sections:
```yaml
experiment:      # Experiment metadata and settings
model:           # Model architecture and parameters
data:            # Data column definitions
train:           # Training configuration
val:             # Validation configuration (optional)
checkpoint:      # Checkpointing strategy
mlflow:          # MLflow logging (optional)
```

For embedding extraction and prediction, see [Special Configurations](#special-configurations).

---

## Experiment Section

Defines experiment metadata and runtime settings.
```yaml
experiment:
  name: str                    # REQUIRED: Experiment name (becomes folder name)
  output_dir: str              # REQUIRED: Base output directory
  seed: int                    # REQUIRED: Random seed for reproducibility
  device: str                  # REQUIRED: Device to use ("cuda", "cpu", "cuda:0", etc.)
```

### Fields

#### `name` (string, required)
- The experiment name
- Used as the folder name within `output_dir`
- Should be descriptive and unique
- Examples: `"baseline"`, `"high_beta"`, `"latent_dim_64"`

#### `output_dir` (string, required)
- Base directory for outputs
- Final path will be `{output_dir}/{name}/`
- Examples: `"./experiments/vae"`, `"./experiments/sae"`

#### `seed` (integer, required)
- Random seed for reproducibility
- Sets seeds for Python, NumPy, and PyTorch
- Use the same seed to reproduce results
- Example: `42`

#### `device` (string, required)
- Device to use for training/inference
- Options:
  - `"cuda"` - Use default GPU
  - `"cpu"` - Use CPU only
  - `"cuda:0"` - Use specific GPU (by index)
- Example: `"cuda"`

### Example
```yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/vae"
  seed: 42
  device: "cuda"
```

**Result:** Everything saved in `./experiments/vae/baseline/`

---

## Model Section

Defines the model architecture and parameters.
```yaml
model:
  class: str                   # REQUIRED: Registered model name
  params: dict                 # REQUIRED: Model-specific parameters
```

### Fields

#### `class` (string, required)
- Name of the registered model class
- Must match a model registered with `@register_model()`
- Available models: `"VAE"`, `"SAE"`, or your custom models
- Example: `"VAE"`

#### `params` (dictionary, required)
- Model-specific parameters
- Varies by model architecture
- Passed to model's `__init__()` method

### Example - VAE
```yaml
model:
  class: "VAE"
  params:
    input_dim: 784           # Input feature dimension (optional, can be inferred)
    latent_dim: 32           # Latent space dimension
    hidden_dims: [256, 128]  # Hidden layer dimensions
```

### Example - SAE
```yaml
model:
  class: "SAE"
  params:
    input_dim: 32            # Input dimension (VAE latent dim)
    hidden_dim: 16           # SAE latent dimension
    sparsity_weight: 0.01    # Sparsity penalty weight
```

### Notes

- `input_dim` can often be inferred from data, but explicitly setting it is clearer
- Each model may have different required/optional parameters
- See `src/models/README.md` for specific model parameter documentation

---

## Data Section

Defines column types and data-related configurations shared across train/val/test splits.
```yaml
data:
  columns:                     # REQUIRED: Column type definitions
    numerical: list[str]       # Numerical feature columns
    categorical: list[str]     # Categorical feature columns
    embeddings: list[str]      # Embedding columns
  cardinalities_path: str      # Path to cardinalities JSON (required if using categorical)
```

### Fields

#### `columns` (dictionary, required)
Specifies which columns are which type.

**`numerical` (list of strings)**
- Column names for numerical features
- Will be loaded as `torch.float32`
- Examples: `["amount", "distance", "age"]`
- Can be empty list: `[]`

**`categorical` (list of strings)**
- Column names for categorical features
- Will be loaded as `torch.int64`
- Requires `cardinalities_path` to be specified
- Examples: `["channel", "merchant_type", "country"]`
- Can be empty list: `[]`

**`embeddings` (list of strings)**
- Column names for pre-computed embeddings
- Will be loaded as `torch.float32`
- Used when training on embeddings (e.g., SAE on VAE embeddings)
- Examples: `["vae_embeddings"]`
- Can be empty list: `[]`

#### `cardinalities_path` (string, optional)
- Path to JSON file containing cardinality information for categorical features
- Required if `categorical` is not empty
- Can be `null` if no categorical features
- Format of JSON file:
```json
  {
    "channel": {"online": 0, "store": 1, "mobile": 2},
    "merchant_type": {"restaurant": 0, "retail": 1, "service": 2}
  }
```

### Example - Tabular Data
```yaml
data:
  columns:
    numerical:
      - log_amount_normalized
      - distance_km
      - transaction_hour
    categorical:
      - channel
      - merchant_type
      - country_code
    embeddings: []
  cardinalities_path: "./data/cardinalities.json"
```

### Example - Embeddings Only (for SAE)
```yaml
data:
  columns:
    numerical: []
    categorical: []
    embeddings:
      - vae_embeddings
  cardinalities_path: null
```

### Notes

- At least one column type must be non-empty
- For self-supervised learning, targets will match features automatically
- The data loader uses these definitions to properly load and type the data

---

## Train Section

Defines training configuration including data, optimization, and loss.
```yaml
train:
  data_path: str               # REQUIRED: Path to training data
  loader_class: str            # REQUIRED: Data loader class name
  loader_params: dict          # REQUIRED: Data loader parameters
  epochs: int                  # REQUIRED: Number of training epochs
  optimizer_config: dict       # REQUIRED: Optimizer configuration
  loss: dict                   # REQUIRED: Loss function configuration
```

### Fields

#### `data_path` (string, required)
- Path to training data file
- Format depends on loader class (CSV, parquet, .npy, etc.)
- Example: `"./data/train.csv"`

#### `loader_class` (string, required)
- Name of registered data loader class
- Must match a loader registered with `@register_dataloader()`
- Available loaders: `"TabularDataLoader"`, `"EmbeddingDataLoader"`, etc.
- Example: `"TabularDataLoader"`

#### `loader_params` (dictionary, required)
Parameters passed to the data loader.

**Common parameters:**
```yaml
loader_params:
  batch_size: int              # REQUIRED: Batch size
  shuffle: bool                # REQUIRED: Whether to shuffle data
  num_workers: int             # Optional: Number of data loading workers (default: 0)
  num_batches_per_epoch: int   # Optional: Limit batches per epoch (null = all)
```

**`batch_size` (integer, required)**
- Number of samples per batch
- Larger = faster but more memory
- Example: `128`

**`shuffle` (boolean, required)**
- Whether to shuffle data each epoch
- Use `true` for training, `false` for validation/test
- Example: `true`

**`num_workers` (integer, optional)**
- Number of parallel data loading workers
- `0` = load in main process
- `> 0` = use multiprocessing
- Default: `0`
- Example: `4`

**`num_batches_per_epoch` (integer or null, optional)**
- Limit number of batches per epoch
- Useful for quick debugging/testing
- `null` = use full dataset
- Example: `10` (only use 10 batches per epoch)

#### `epochs` (integer, required)
- Number of training epochs
- Example: `100`

#### `optimizer_config` (dictionary, required)
Configuration for the optimizer.
```yaml
optimizer_config:
  optimizer: str               # REQUIRED: Optimizer name
  learning_rate: float         # REQUIRED: Learning rate
  # Additional optimizer-specific parameters (optional)
```

**`optimizer` (string, required)**
- Name of PyTorch optimizer class
- Examples: `"Adam"`, `"SGD"`, `"AdamW"`, `"RMSprop"`

**`learning_rate` (float, required)**
- Learning rate
- Example: `0.001`

**Additional parameters (optional):**
- Any valid parameters for the chosen optimizer
- Passed directly to optimizer constructor

**Example - Adam:**
```yaml
optimizer_config:
  optimizer: "Adam"
  learning_rate: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 1e-08
  amsgrad: false
```

**Example - SGD:**
```yaml
optimizer_config:
  optimizer: "SGD"
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  nesterov: true
```

#### `loss` (dictionary, required)
Configuration for the loss function.
```yaml
loss:
  class: str                   # REQUIRED: Loss function name
  params: dict                 # REQUIRED: Loss-specific parameters
```

**`class` (string, required)**
- Name of registered loss function
- Must match a loss registered with `@register_loss()`
- Available losses: `"VAELoss"`, `"SAELoss"`, etc.
- Example: `"VAELoss"`

**`params` (dictionary, required)**
- Loss-specific parameters
- Passed to loss function via `**kwargs`
- Can be empty dict `{}` if loss has no parameters

**Example:**
```yaml
loss:
  class: "VAELoss"
  params:
    beta: 1.0
    reconstruction_weight: 1.0
```

### Complete Train Section Example
```yaml
train:
  data_path: "./data/train.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 128
    shuffle: true
    num_workers: 4
    num_batches_per_epoch: null  # Use full dataset
  
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
```

---

## Val Section

Defines validation configuration. **Optional** - if not provided, no validation is performed.
```yaml
val:
  data_path: str               # REQUIRED: Path to validation data
  loader_class: str            # REQUIRED: Data loader class name
  loader_params: dict          # REQUIRED: Data loader parameters
  frequency: int               # REQUIRED: Validate every N epochs
  loss: dict                   # REQUIRED: Loss function configuration
  early_stopping: dict         # Optional: Early stopping configuration
```

### Fields

#### `data_path` (string, required)
- Path to validation data file
- Example: `"./data/val.csv"`

#### `loader_class` (string, required)
- Same as train section
- Usually the same loader class as training
- Example: `"TabularDataLoader"`

#### `loader_params` (dictionary, required)
- Same structure as train section
- Typically use:
  - Larger batch size (faster validation)
  - `shuffle: false` (consistent validation order)
  - `num_batches_per_epoch: null` (use full validation set)

**Example:**
```yaml
loader_params:
  batch_size: 256
  shuffle: false
  num_workers: 4
  num_batches_per_epoch: null
```

#### `frequency` (integer, required)
- Run validation every N epochs
- Example: `5` (validate every 5 epochs)
- Use `1` to validate every epoch

#### `loss` (dictionary, required)
- Same structure as train section
- Can have different parameters than training loss
- This allows testing different loss configurations

**Example - Same as training:**
```yaml
loss:
  class: "VAELoss"
  params:
    beta: 1.0
    reconstruction_weight: 1.0
```

**Example - Different from training:**
```yaml
loss:
  class: "VAELoss"
  params:
    beta: 0.5  # Lower beta for validation
    reconstruction_weight: 1.0
```

#### `early_stopping` (dictionary, optional)
Configuration for early stopping. If not provided, no early stopping is used.
```yaml
early_stopping:
  patience: int                # REQUIRED: Number of validations without improvement
  metric: str                  # REQUIRED: Metric name to monitor
  direction: str               # REQUIRED: "min" or "max"
```

**`patience` (integer, required)**
- Number of validation runs without improvement before stopping
- Example: `20` (stop if no improvement for 20 validations)

**`metric` (string, required)**
- Name of metric to monitor
- Must match a metric logged during validation
- Examples: `"val_loss"`, `"val_reconstruction_loss"`, `"val_accuracy"`
- Important: Must be exact metric name

**`direction` (string, required)**
- Direction of improvement
- `"min"` - Lower is better (for losses)
- `"max"` - Higher is better (for accuracies)

**Example:**
```yaml
early_stopping:
  patience: 20
  metric: "val_loss"
  direction: "min"
```

### Complete Val Section Example
```yaml
val:
  data_path: "./data/val.csv"
  loader_class: "TabularDataLoader"
  loader_params:
    batch_size: 256
    shuffle: false
    num_workers: 4
    num_batches_per_epoch: null
  
  frequency: 5  # Validate every 5 epochs
  
  loss:
    class: "VAELoss"
    params:
      beta: 1.0
      reconstruction_weight: 1.0
  
  early_stopping:
    patience: 20
    metric: "val_loss"
    direction: "min"
```

### Notes

- If `val` section is omitted, training runs without validation
- Without validation, `checkpoint.metric` must be based on training metrics
- `num_batches_per_epoch` can be used in validation for quick debugging

---

## Checkpoint Section

Defines checkpointing strategy.
```yaml
checkpoint:
  metric: str                  # REQUIRED: Metric to use for "best" model
  mode: str                    # REQUIRED: "min" or "max"
  save_strategy: str           # REQUIRED: "best", "last", or "all"
```

### Fields

#### `metric` (string, required)
- Metric used to determine "best" model
- Must be a metric logged during training/validation
- Examples: `"val_loss"`, `"train_loss"`, `"val_accuracy"`
- If using validation: typically use validation metric
- If no validation: use training metric

#### `mode` (string, required)
- How to compare metric values
- `"min"` - Lower is better (for losses)
- `"max"` - Higher is better (for accuracies)

#### `save_strategy` (string, required)
- Which checkpoints to save
- Options:
  - `"best"` - Only save when metric improves
  - `"last"` - Only save most recent epoch
  - `"all"` - Save every epoch + best

**Comparison:**

| Strategy | Best | Last | All Epochs | Disk Usage |
|----------|------|------|------------|------------|
| `"best"` | ✓ | ✗ | ✗ | Low |
| `"last"` | ✗ | ✓ | ✗ | Low |
| `"all"` | ✓ | ✓ | ✓ | High |

### Examples

**Example 1 - Save only best (recommended for most cases):**
```yaml
checkpoint:
  metric: "val_loss"
  mode: "min"
  save_strategy: "best"
```

**Example 2 - Save all epochs (for important experiments):**
```yaml
checkpoint:
  metric: "val_loss"
  mode: "min"
  save_strategy: "all"
```

**Example 3 - No validation, use training loss:**
```yaml
checkpoint:
  metric: "train_loss"
  mode: "min"
  save_strategy: "best"
```

### Notes

- `save_strategy: "all"` can use significant disk space (one file per epoch)
- If no validation, must use training metric
- Checkpoint files include full model state, optimizer state, and config

---

## MLflow Section

Defines MLflow experiment tracking. **Optional** - if not provided, no MLflow logging.
```yaml
mlflow:
  tracking_uri: str            # REQUIRED: MLflow server URI
  experiment_name: str         # REQUIRED: MLflow experiment name
  credentials: dict            # Optional: Authentication credentials
```

### Fields

#### `tracking_uri` (string, required)
- URI of MLflow tracking server
- Examples:
  - `"https://mlflow.example.com"` - Remote server
  - `"file:./mlruns"` - Local directory
  - `"sqlite:///mlflow.db"` - SQLite backend

#### `experiment_name` (string, required)
- Name of MLflow experiment
- Groups related runs together
- Example: `"vae_experiments"`

#### `credentials` (dictionary, optional)
- Authentication credentials for MLflow server
- Only needed if server requires authentication
```yaml
credentials:
  username: str
  password: str
```

### Example
```yaml
mlflow:
  tracking_uri: "https://mlflow.example.com"
  experiment_name: "vae_experiments"
  credentials:
    username: "user"
    password: "pass"
```

### What Gets Logged

**Automatic logging:**
- Training loss (every epoch)
- Validation loss (when validation runs)
- Learning rate
- Epoch number
- Config file (as artifact)
- Best checkpoint (as artifact)

**Custom logging:**
- You can log additional metrics in runner code
- Use `mlflow.log_metric()` and `mlflow.log_artifact()`

### Notes

- If `mlflow` section is omitted, no MLflow logging occurs
- Credentials can be omitted if server doesn't require auth
- Consider using environment variables for sensitive credentials

---

## Complete Examples

### Example 1: Training VAE with Validation
```yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/vae"
  seed: 42
  device: "cuda"

model:
  class: "VAE"
  params:
    input_dim: 784
    latent_dim: 32
    hidden_dims: [256, 128]

data:
  columns:
    numerical:
      - log_amount_normalized
      - distance_km
      - transaction_hour
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
    num_batches_per_epoch: null
  
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
  credentials:
    username: "user"
    password: "pass"
```

### Example 2: Training SAE on Embeddings
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
    num_batches_per_epoch: null
  
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
    num_batches_per_epoch: null
  
  frequency: 5
  
  loss:
    class: "SAELoss"
    params:
      sparsity_weight: 0.01

checkpoint:
  metric: "val_loss"
  mode: "min"
  save_strategy: "best"

mlflow:
  tracking_uri: "https://mlflow.example.com"
  experiment_name: "sae_experiments"
```

### Example 3: Quick Debug Configuration
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
    hidden_dims: [256, 128]

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
    batch_size: 128
    shuffle: true
    num_workers: 2
    num_batches_per_epoch: 10  # Only 10 batches for quick test!
  
  epochs: 3  # Just a few epochs
  
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
    num_workers: 2
    num_batches_per_epoch: 5  # Only 5 batches for validation
  
  frequency: 1  # Validate every epoch
  
  loss:
    class: "VAELoss"
    params:
      beta: 1.0
      reconstruction_weight: 1.0

checkpoint:
  metric: "val_loss"
  mode: "min"
  save_strategy: "last"  # Don't need best for debugging
```

---

## Special Configurations

### Embedding Extraction Configuration

Used with `scripts/extract_embeddings.py`:
```yaml
experiment:
  name: "extract_vae_embeddings"
  output_dir: "./experiments/vae/baseline"
  device: "cuda"

checkpoint:
  path: str                    # REQUIRED: Path to checkpoint file
  load_strategy: str           # REQUIRED: "best", "last", or "epoch_N"

model:
  class: str                   # REQUIRED: Must match training model
  # params not needed (loaded from checkpoint)

data:
  columns:                     # Same as training config
    numerical: [...]
    categorical: [...]
    embeddings: [...]
  cardinalities_path: str

# Optional: Include only splits you want to extract
train:
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: false             # Always false for extraction
    num_workers: int
  output: str                  # REQUIRED: Output path for embeddings

val:
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: false
    num_workers: int
  output: str

test:
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: false
    num_workers: int
  output: str
```

**Key differences from training config:**
- `checkpoint` section specifies which model to load
- `train`/`val`/`test` sections have `output` field for embeddings path
- Include only the splits you want to extract
- No `epochs`, `optimizer_config`, `loss`, or validation sections

**Example:**
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

### Prediction Configuration

Used with `scripts/predict.py`:
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
  columns: [...]

test:
  data_path: str
  loader_class: str
  loader_params:
    batch_size: int
    shuffle: false
    num_workers: int
  output: str                  # REQUIRED: Output path for predictions
  
  metrics:                     # Optional: compute metrics
    compute: bool
    loss:
      class: str
      params: dict
```

**Example:**
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
  
  metrics:
    compute: true
    loss:
      class: "VAELoss"
      params:
        beta: 1.0
        reconstruction_weight: 1.0
```

---

## Tips and Best Practices

### Organizing Configs

Group related configs in subdirectories:
```
configs/
├── examples/
│   ├── train_vae.yaml
│   ├── train_sae.yaml
│   └── extract_embeddings.yaml
├── production/
│   ├── vae_baseline.yaml
│   └── sae_baseline.yaml
└── experiments/
    ├── 2024_01_15/
    │   ├── experiment_01.yaml
    │   └── experiment_02.yaml
    └── ablations/
        ├── no_beta.yaml
        └── high_sparsity.yaml
```

### Naming Conventions

Use descriptive names that indicate what's being tested:
```yaml
# Good
name: "baseline"
name: "beta_2.0"
name: "latent_dim_64"
name: "no_weight_decay"

# Less descriptive
name: "test1"
name: "run"
name: "temp"
```

### Debugging Workflows

1. **Start with debug config:**
   - Small `num_batches_per_epoch` (10-20)
   - Few epochs (3-5)
   - `save_strategy: "last"`

2. **Scale up gradually:**
   - Increase to `num_batches_per_epoch: 100`
   - More epochs (20-30)

3. **Full training:**
   - `num_batches_per_epoch: null`
   - Full epochs (100+)
   - `save_strategy: "best"` or `"all"`

### Hyperparameter Sweeps

Create one config per hyperparameter combination:
```yaml
# configs/sweep/beta_0.5.yaml
experiment:
  name: "beta_0.5"
  output_dir: "./experiments/vae_sweep"
# ...
loss:
  params:
    beta: 0.5

# configs/sweep/beta_1.0.yaml
experiment:
  name: "beta_1.0"
  output_dir: "./experiments/vae_sweep"
# ...
loss:
  params:
    beta: 1.0

# configs/sweep/beta_2.0.yaml
experiment:
  name: "beta_2.0"
  output_dir: "./experiments/vae_sweep"
# ...
loss:
  params:
    beta: 2.0