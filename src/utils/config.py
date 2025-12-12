"""
Configuration loading, saving, and validation utilities.
"""

import os
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    return config


def save_config(config: dict, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path where to save config
    """
    save_path = Path(save_path)
    
    # Create parent directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: dict, config_type: str = 'train'):
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        config_type: Type of config ('train', 'extract', 'predict')
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Common required fields
    if 'experiment' not in config:
        raise ValueError("Config missing 'experiment' section")
    
    experiment = config['experiment']
    required_experiment_fields = ['name', 'output_dir', 'seed', 'device']
    for field in required_experiment_fields:
        if field not in experiment:
            raise ValueError(f"Experiment section missing required field: '{field}'")
    
    if 'model' not in config:
        raise ValueError("Config missing 'model' section")
    
    model = config['model']
    if 'class' not in model:
        raise ValueError("Model section missing 'class' field")
    if 'params' not in model:
        raise ValueError("Model section missing 'params' field")
    
    if 'data' not in config:
        raise ValueError("Config missing 'data' section")
    
    data = config['data']
    if 'columns' not in data:
        raise ValueError("Data section missing 'columns' field")
    
    columns = data['columns']
    required_column_types = ['numerical', 'categorical', 'embeddings']
    for col_type in required_column_types:
        if col_type not in columns:
            raise ValueError(f"Data columns missing '{col_type}' field")
    
    # Type-specific validation
    if config_type == 'train':
        _validate_train_config(config)
    elif config_type == 'extract':
        _validate_extract_config(config)
    elif config_type == 'predict':
        _validate_predict_config(config)


def _validate_train_config(config: dict):
    """Validate training-specific config fields."""
    if 'train' not in config:
        raise ValueError("Training config missing 'train' section")
    
    train = config['train']
    required_train_fields = ['data_path', 'loader_class', 'loader_params', 
                            'epochs', 'optimizer_config', 'loss']
    for field in required_train_fields:
        if field not in train:
            raise ValueError(f"Train section missing required field: '{field}'")
    
    # Validate optimizer_config
    optimizer_config = train['optimizer_config']
    if 'optimizer' not in optimizer_config:
        raise ValueError("optimizer_config missing 'optimizer' field")
    if 'learning_rate' not in optimizer_config:
        raise ValueError("optimizer_config missing 'learning_rate' field")
    
    # Validate loss
    loss = train['loss']
    if 'class' not in loss:
        raise ValueError("Loss section missing 'class' field")
    if 'params' not in loss:
        raise ValueError("Loss section missing 'params' field")
    
    # Validate checkpoint
    if 'checkpoint' not in config:
        raise ValueError("Training config missing 'checkpoint' section")
    
    checkpoint = config['checkpoint']
    required_checkpoint_fields = ['metric', 'mode', 'save_strategy']
    for field in required_checkpoint_fields:
        if field not in checkpoint:
            raise ValueError(f"Checkpoint section missing required field: '{field}'")
    
    # Validate checkpoint values
    if checkpoint['mode'] not in ['min', 'max']:
        raise ValueError(f"Checkpoint mode must be 'min' or 'max', got: {checkpoint['mode']}")
    
    if checkpoint['save_strategy'] not in ['best', 'last', 'all']:
        raise ValueError(
            f"Checkpoint save_strategy must be 'best', 'last', or 'all', "
            f"got: {checkpoint['save_strategy']}"
        )


def _validate_extract_config(config: dict):
    """Validate embedding extraction config fields."""
    if 'checkpoint' not in config:
        raise ValueError("Extraction config missing 'checkpoint' section")
    
    checkpoint = config['checkpoint']
    if 'path' not in checkpoint:
        raise ValueError("Checkpoint section missing 'path' field")
    if 'load_strategy' not in checkpoint:
        raise ValueError("Checkpoint section missing 'load_strategy' field")
    
    # Check at least one split is specified
    has_split = False
    for split in ['train', 'val', 'test']:
        if split in config:
            has_split = True
            split_config = config[split]
            required_fields = ['data_path', 'loader_class', 'loader_params', 'output']
            for field in required_fields:
                if field not in split_config:
                    raise ValueError(f"{split} section missing required field: '{field}'")
    
    if not has_split:
        raise ValueError(
            "Extraction config must have at least one split section (train/val/test)"
        )


def _validate_predict_config(config: dict):
    """Validate prediction config fields."""
    if 'checkpoint' not in config:
        raise ValueError("Prediction config missing 'checkpoint' section")
    
    checkpoint = config['checkpoint']
    if 'path' not in checkpoint:
        raise ValueError("Checkpoint section missing 'path' field")
    if 'load_strategy' not in checkpoint:
        raise ValueError("Checkpoint section missing 'load_strategy' field")
    
    if 'test' not in config:
        raise ValueError("Prediction config missing 'test' section")
    
    test = config['test']
    required_test_fields = ['data_path', 'loader_class', 'loader_params', 'output']
    for field in required_test_fields:
        if field not in test:
            raise ValueError(f"Test section missing required field: '{field}'")


def create_experiment_dirs(output_dir: str, name: str) -> dict:
    """
    Create experiment directory structure.
    
    Args:
        output_dir: Base output directory
        name: Experiment name
    
    Returns:
        Dict with paths to created directories:
        {
            'root': experiment root directory,
            'checkpoints': checkpoints directory,
            'logs': logs directory,
            'embeddings': embeddings directory,
            'predictions': predictions directory
        }
    """
    root = Path(output_dir) / name
    
    paths = {
        'root': root,
        'checkpoints': root / 'checkpoints',
        'logs': root / 'logs',
        'embeddings': root / 'embeddings',
        'predictions': root / 'predictions'
    }
    
    # Create all directories
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in paths.items()}