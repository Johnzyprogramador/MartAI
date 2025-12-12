# Runners Documentation

Runners orchestrate training, validation, prediction, and embedding extraction.

## Available Runners

### `TrainRunner`
Handles training loop with optional validation.

**Usage:**
```python
from src.runners import TrainRunner

runner = TrainRunner(config)
runner.run()
```

**What it does:**
- Initializes model, optimizer, loss functions, data loaders
- Runs training epochs
- Runs validation every N epochs (if configured)
- Saves checkpoints according to strategy
- Implements early stopping (if configured)
- Logs metrics to MLflow (if configured)

---

### `EmbeddingRunner`
Extracts embeddings from trained models.

**Usage:**
```python
from src.runners import EmbeddingRunner

runner = EmbeddingRunner(config)
runner.run()
```

**What it does:**
- Loads checkpoint
- Calls `model.extract_embeddings()` on specified splits
- Saves embeddings to `.npy` files

---

### `PredictRunner`
Runs inference on test data.

**Usage:**
```python
from src.runners import PredictRunner

runner = PredictRunner(config)
runner.run()
```

**What it does:**
- Loads checkpoint
- Runs `model.forward()` on test data
- Optionally computes metrics
- Saves predictions to `.npy` file

---

## TrainRunner Details

### Initialization
```python
class TrainRunner:
    def __init__(self, config: dict):
        """
        Initialize from config dictionary.
        
        Sets up:
        - Device
        - Random seeds
        - Model
        - Data loaders (train + val if configured)
        - Optimizer
        - Loss functions (train + val)
        - Checkpointing
        - Early stopping (if configured)
        - MLflow (if configured)
        """
```

### Main Loop
```python
def run(self):
    """
    Main training loop.
    
    For each epoch:
        1. Run training epoch
        2. Log training metrics
        3. Run validation (if configured and if frequency matches)
        4. Log validation metrics
        5. Check early stopping
        6. Save checkpoints
    """
```

### Key Methods
```python
def _train_epoch(self) -> dict:
    """
    Run one training epoch.
    
    Returns:
        dict: Training metrics (e.g., {'train_loss': 0.5})
    """

def _validate_epoch(self) -> dict:
    """
    Run validation epoch.
    
    Returns:
        dict: Validation metrics (e.g., {'val_loss': 0.3})
    """

def _save_checkpoint(self, name: str):
    """
    Save model checkpoint.
    
    Args:
        name: Checkpoint name ('best', 'last', 'epoch_N')
    """

def _check_early_stopping(self, metrics: dict) -> bool:
    """
    Check if should stop early.
    
    Args:
        metrics: Validation metrics
    
    Returns:
        True if should stop, False otherwise
    """
```

---

## EmbeddingRunner Details

### Initialization
```python
class EmbeddingRunner:
    def __init__(self, config: dict):
        """
        Initialize from config dictionary.
        
        Sets up:
        - Device
        - Model (loads from checkpoint)
        - Data loaders (for each split to extract)
        """
```

### Extraction
```python
def run(self):
    """
    Extract embeddings for all configured splits.
    
    For each split (train/val/test):
        1. Load data
        2. Run model.extract_embeddings() on all batches
        3. Concatenate results
        4. Save to .npy file
    """
```

---

## PredictRunner Details

### Initialization
```python
class PredictRunner:
    def __init__(self, config: dict):
        """
        Initialize from config dictionary.
        
        Sets up:
        - Device
        - Model (loads from checkpoint)
        - Test data loader
        - Loss function (if metrics configured)
        """
```

### Prediction
```python
def run(self):
    """
    Run predictions on test data.
    
    1. Run model.forward() on all test batches
    2. Concatenate predictions
    3. Compute metrics (if configured)
    4. Save predictions to .npy file
    """
```

---

## Checkpoint Loading

All runners that load checkpoints use:
```python
def _load_checkpoint(self, checkpoint_path: str, load_strategy: str):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        load_strategy: 'best', 'last', or 'epoch_N'
    
    Returns:
        Loaded checkpoint dictionary
    """
```

**Load strategies:**
- `"best"`: Load `best_model.pt`
- `"last"`: Load `last_model.pt`
- `"epoch_50"`: Load `epoch_050.pt`

---

## Logging

### Training Metrics

Automatically logged:
- `train_loss` (every epoch)
- `val_loss` (when validation runs)
- Learning rate
- Epoch number

### Custom Metrics

Add custom logging in runner:
```python
# In _train_epoch() or _validate_epoch()
metrics = {
    'train_loss': loss.item(),
    'custom_metric': value  # Your custom metric
}
return metrics
```

These are automatically logged to MLflow (if configured).

---

## Error Handling

Runners handle common errors:
- Config validation
- Missing checkpoints
- Device availability
- Data loading errors

Errors are logged and raised with informative messages.

---

## Extending Runners

To add custom functionality:

1. Subclass existing runner
2. Override specific methods
3. Use in scripts

**Example:**
```python
class CustomTrainRunner(TrainRunner):
    def _train_epoch(self):
        # Custom training logic
        metrics = super()._train_epoch()
        # Add custom metrics
        metrics['custom'] = compute_custom()
        return metrics
```

---

## Runner Workflow

### Training Workflow
```
Config → TrainRunner
    ↓
Initialize (model, data, optimizer, loss)
    ↓
For each epoch:
    ↓
    Train epoch → compute loss → backward → update
    ↓
    Validate (if configured) → compute metrics
    ↓
    Check early stopping
    ↓
    Save checkpoint (according to strategy)
    ↓
Save final checkpoint
```

### Embedding Extraction Workflow
```
Config → EmbeddingRunner
    ↓
Load checkpoint
    ↓
For each split (train/val/test):
    ↓
    Load data
    ↓
    For each batch:
        extract_embeddings() → collect
    ↓
    Concatenate → save to .npy
```

### Prediction Workflow
```
Config → PredictRunner
    ↓
Load checkpoint
    ↓
Load test data
    ↓
For each batch:
    forward() → collect predictions
    ↓
    (optional) compute metrics
    ↓
Concatenate → save to .npy
```

---

## Best Practices

### Runner Selection
- Use `TrainRunner` for training
- Use `EmbeddingRunner` for extraction
- Use `PredictRunner` for inference
- Don't mix responsibilities

### Checkpointing
- Always save config with checkpoints
- Use atomic writes (avoid corruption)
- Clean up old checkpoints if using `save_strategy: "all"`

### Logging
- Log informative messages during training
- Use consistent metric names
- Log errors with context

### Memory Management
- Clear gradients properly
- Use `torch.no_grad()` for inference
- Delete unnecessary variables