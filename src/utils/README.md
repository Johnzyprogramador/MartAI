# Utils Documentation

Utility functions for configuration, checkpointing, logging, and MLflow integration.

## Available Modules

### `config.py`
Configuration loading and validation.

**Functions:**
```python
def load_config(config_path: str) -> dict:
    """
    Load YAML config file.
    
    Args:
        config_path: Path to config YAML
    
    Returns:
        Config dictionary
    """

def save_config(config: dict, save_path: str):
    """
    Save config to YAML file.
    
    Args:
        config: Config dictionary
        save_path: Where to save
    """

def validate_config(config: dict):
    """
    Validate config structure.
    
    Args:
        config: Config dictionary
    
    Raises:
        ValueError: If config is invalid
    """
```

---

### `checkpoint.py`
Checkpoint saving and loading.

**Functions:**
```python
def save_checkpoint(checkpoint: dict, save_path: str):
    """
    Save model checkpoint.
    
    Args:
        checkpoint: Dict containing:
            - 'epoch': int
            - 'model_state_dict': dict
            - 'optimizer_state_dict': dict
            - 'metric_value': float
            - 'config': dict
        save_path: Where to save (e.g., 'checkpoints/best_model.pt')
    """

def load_checkpoint(checkpoint_path: str, device: str) -> dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to ('cuda', 'cpu')
    
    Returns:
        Checkpoint dictionary
    """

def get_checkpoint_path(base_path: str, load_strategy: str) -> str:
    """
    Get checkpoint path based on load strategy.
    
    Args:
        base_path: Base checkpoint directory
        load_strategy: 'best', 'last', or 'epoch_N'
    
    Returns:
        Full path to checkpoint file
    """
```

---

### `logging_utils.py`
Logging setup and utilities.

**Functions:**
```python
def setup_logger(log_path: str, name: str = 'train') -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        log_path: Path to log file
        name: Logger name
    
    Returns:
        Configured logger
    """

def log_metrics(logger: logging.Logger, metrics: dict, epoch: int):
    """
    Log metrics to logger.
    
    Args:
        logger: Logger instance
        metrics: Dict of metric names and values
        epoch: Current epoch
    """
```

---

### `mlflow_utils.py`
MLflow integration utilities.

**Functions:**
```python
def setup_mlflow(config: dict):
    """
    Setup MLflow tracking.
    
    Args:
        config: Config dict with 'mlflow' section
    """

def log_config(config: dict):
    """
    Log config as MLflow artifact.
    
    Args:
        config: Config dictionary
    """

def log_model(model_path: str, artifact_path: str = "model"):
    """
    Log model as MLflow artifact.
    
    Args:
        model_path: Path to model checkpoint
        artifact_path: MLflow artifact path
    """
```

---

## Usage Examples

### Loading Config
```python
from src.utils.config import load_config

config = load_config("configs/train_vae.yaml")
```

### Saving Checkpoint
```python
from src.utils.checkpoint import save_checkpoint

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metric_value': val_loss,
    'config': config
}

save_checkpoint(checkpoint, "experiments/vae/baseline/checkpoints/best_model.pt")
```

### Loading Checkpoint
```python
from src.utils.checkpoint import load_checkpoint

checkpoint = load_checkpoint(
    "experiments/vae/baseline/checkpoints/best_model.pt",
    device="cuda"
)

model.load_state_dict(checkpoint['model_state_dict'])
```

### Setup Logging
```python
from src.utils.logging_utils import setup_logger, log_metrics

logger = setup_logger("experiments/vae/baseline/logs/training.log")
log_metrics(logger, {'train_loss': 0.5, 'val_loss': 0.3}, epoch=10)
```

### MLflow Integration
```python
from src.utils.mlflow_utils import setup_mlflow, log_config
import mlflow

# Setup
setup_mlflow(config)

# Start run
with mlflow.start_run():
    log_config(config)
    mlflow.log_metric("train_loss", 0.5, step=epoch)
```

---

## Helper Functions

### Random Seed Setting
```python
def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Device Setup
```python
def get_device(device_str: str) -> torch.device:
    """
    Get torch device from string.
    
    Args:
        device_str: 'cuda', 'cpu', or 'cuda:0'
    
    Returns:
        torch.device
    """
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    return torch.device(device_str)
```

---

## File Structure Utilities

### Creating Output Directories
```python
def create_experiment_dirs(output_dir: str, name: str) -> dict:
    """
    Create experiment directory structure.
    
    Args:
        output_dir: Base output directory
        name: Experiment name
    
    Returns:
        Dict with paths:
        {
            'root': path to experiment root,
            'checkpoints': path to checkpoints dir,
            'logs': path to logs dir,
            'embeddings': path to embeddings dir,
            'predictions': path to predictions dir
        }
    """
    import os
    
    root = os.path.join(output_dir, name)
    paths = {
        'root': root,
        'checkpoints': os.path.join(root, 'checkpoints'),
        'logs': os.path.join(root, 'logs'),
        'embeddings': os.path.join(root, 'embeddings'),
        'predictions': os.path.join(root, 'predictions')
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths
```

---

## Best Practices

### Config Management
- Always save config with outputs
- Validate config before running
- Use absolute or project-relative paths

### Checkpointing
- Save atomically (write to temp, then rename)
- Include all necessary state
- Save config for reproducibility

### Logging
- Log to both file and console
- Use appropriate log levels (INFO, DEBUG, ERROR)
- Include timestamps

### MLflow
- Use consistent experiment names
- Log hyperparameters and metrics
- Tag runs appropriately