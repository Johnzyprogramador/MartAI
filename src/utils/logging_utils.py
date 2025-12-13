"""
Logging utilities for training and inference.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    log_path: Optional[str] = None,
    name: str = 'train',
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        log_path: Path to log file (if None, only console logging)
        name: Logger name
        level: Logging level (default: INFO)
        console: Whether to also log to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_metrics(logger: logging.Logger, metrics: dict, epoch: int, prefix: str = ''):
    """
    Log metrics to logger.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        epoch: Current epoch number
        prefix: Prefix to add to log message (e.g., 'Train', 'Val')
    """
    metric_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    if prefix:
        log_message = f"[Epoch {epoch}] {prefix} - {metric_str}"
    else:
        log_message = f"[Epoch {epoch}] {metric_str}"
    
    logger.info(log_message)


def log_config(logger: logging.Logger, config: dict):
    """
    Log configuration to logger.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("=" * 80)
    
    _log_dict_recursive(logger, config, indent=0)
    
    logger.info("=" * 80)


def _log_dict_recursive(logger: logging.Logger, d: dict, indent: int = 0):
    """
    Recursively log dictionary with indentation.
    
    Args:
        logger: Logger instance
        d: Dictionary to log
        indent: Current indentation level
    """
    indent_str = "  " * indent
    
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info(f"{indent_str}{key}:")
            _log_dict_recursive(logger, value, indent + 1)
        else:
            logger.info(f"{indent_str}{key}: {value}")


def log_model_info(logger: logging.Logger, model):
    """
    Log model architecture and parameter count.
    
    Args:
        logger: Logger instance
        model: PyTorch model
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 80)
    logger.info("Model Information:")
    logger.info("=" * 80)
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info("=" * 80)


def log_training_start(logger: logging.Logger, total_epochs: int, train_batches: int, val_batches: int = None):
    """
    Log training start information.
    
    Args:
        logger: Logger instance
        total_epochs: Total number of epochs
        train_batches: Number of training batches per epoch
        val_batches: Number of validation batches per epoch (optional)
    """
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    logger.info(f"Total epochs: {total_epochs}")
    logger.info(f"Training batches per epoch: {train_batches}")
    if val_batches is not None:
        logger.info(f"Validation batches per epoch: {val_batches}")
    logger.info("=" * 80)


def log_training_end(logger: logging.Logger, best_epoch: int = None, best_metric: float = None):
    """
    Log training end information.
    
    Args:
        logger: Logger instance
        best_epoch: Epoch with best metric (optional)
        best_metric: Best metric value (optional)
    """
    logger.info("=" * 80)
    logger.info("Training Completed")
    logger.info("=" * 80)
    if best_epoch is not None and best_metric is not None:
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best metric value: {best_metric:.4f}")
    logger.info("=" * 80)


def log_checkpoint_saved(logger: logging.Logger, checkpoint_path: str, epoch: int, metric_value: float = None):
    """
    Log checkpoint save information.
    
    Args:
        logger: Logger instance
        checkpoint_path: Path where checkpoint was saved
        epoch: Epoch number
        metric_value: Metric value (optional)
    """
    if metric_value is not None:
        logger.info(f"Checkpoint saved: {checkpoint_path} (epoch {epoch}, metric: {metric_value:.4f})")
    else:
        logger.info(f"Checkpoint saved: {checkpoint_path} (epoch {epoch})")


def log_early_stopping(logger: logging.Logger, epoch: int, patience: int):
    """
    Log early stopping trigger.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        patience: Patience value that was exceeded
    """
    logger.info("=" * 80)
    logger.info(f"Early stopping triggered at epoch {epoch}")
    logger.info(f"No improvement for {patience} validation runs")
    logger.info("=" * 80)