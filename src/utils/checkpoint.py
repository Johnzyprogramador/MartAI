"""
Checkpoint saving and loading utilities.
"""

import os
import torch
from pathlib import Path
from typing import Optional


def save_checkpoint(
    checkpoint: dict,
    save_path: str,
    is_best: bool = False,
    best_path: Optional[str] = None
):
    """
    Save model checkpoint to disk.
    
    Args:
        checkpoint: Dictionary containing:
            - 'epoch': Current epoch number
            - 'model_state_dict': Model state dictionary
            - 'optimizer_state_dict': Optimizer state dictionary
            - 'metric_value': Value of tracked metric
            - 'config': Full configuration dictionary
            - (optional) 'random_state': Random number generator states
        save_path: Path where to save checkpoint
        is_best: Whether this is the best checkpoint so far
        best_path: Path where to save best checkpoint (if is_best=True)
    """
    save_path = Path(save_path)
    
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint atomically (write to temp file, then rename)
    temp_path = save_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(save_path)
    
    # If this is the best checkpoint, save a copy
    if is_best and best_path is not None:
        best_path = Path(best_path)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        temp_best = best_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_best)
        temp_best.rename(best_path)


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> dict:
    """
    Load model checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        Checkpoint dictionary
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    
    return checkpoint


def get_checkpoint_path(base_path: str, load_strategy: str) -> str:
    """
    Get checkpoint path based on load strategy.
    
    Args:
        base_path: Base checkpoint directory or full path to checkpoint file
        load_strategy: Strategy for loading checkpoint:
            - 'best': Load best_model.pt
            - 'last': Load last_model.pt
            - 'epoch_N': Load epoch_N.pt (e.g., 'epoch_50')
    
    Returns:
        Full path to checkpoint file
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If load_strategy is invalid
    """
    base_path = Path(base_path)
    
    # If base_path is a file, return it directly
    if base_path.is_file():
        return str(base_path)
    
    # If base_path is a directory, construct path based on strategy
    if base_path.is_dir():
        if load_strategy == 'best':
            checkpoint_path = base_path / 'best_model.pt'
        elif load_strategy == 'last':
            checkpoint_path = base_path / 'last_model.pt'
        elif load_strategy.startswith('epoch_'):
            # Extract epoch number and format with leading zeros
            try:
                epoch_num = int(load_strategy.split('_')[1])
                checkpoint_path = base_path / f'epoch_{epoch_num:03d}.pt'
            except (IndexError, ValueError):
                raise ValueError(
                    f"Invalid load_strategy format: '{load_strategy}'. "
                    f"Expected 'epoch_N' where N is an integer."
                )
        else:
            raise ValueError(
                f"Invalid load_strategy: '{load_strategy}'. "
                f"Must be 'best', 'last', or 'epoch_N'."
            )
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Available checkpoints: {list(base_path.glob('*.pt'))}"
            )
        
        return str(checkpoint_path)
    
    raise FileNotFoundError(f"Checkpoint path does not exist: {base_path}")


def create_checkpoint(
    epoch: int,
    model,
    optimizer,
    metric_value: float,
    config: dict
) -> dict:
    """
    Create checkpoint dictionary with all necessary information.
    
    Args:
        epoch: Current epoch number
        model: PyTorch model
        optimizer: PyTorch optimizer
        metric_value: Value of tracked metric
        config: Configuration dictionary
    
    Returns:
        Checkpoint dictionary ready to be saved
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric_value': metric_value,
        'config': config,
        'random_state': {
            'torch': torch.get_rng_state(),
        }
    }
    
    # Add CUDA random state if available
    if torch.cuda.is_available():
        checkpoint['random_state']['cuda'] = torch.cuda.get_rng_state()
    
    return checkpoint


def load_model_from_checkpoint(checkpoint: dict, model, optimizer=None, strict: bool = True):
    """
    Load model and optimizer states from checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer to load state into (optional)
        strict: Whether to strictly enforce that keys match
    
    Returns:
        epoch: Epoch number from checkpoint
    """
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore random states if present
    if 'random_state' in checkpoint:
        if 'torch' in checkpoint['random_state']:
            torch.set_rng_state(checkpoint['random_state']['torch'])
        if 'cuda' in checkpoint['random_state'] and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['random_state']['cuda'])
    
    return checkpoint.get('epoch', 0)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for all .pt files
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    
    if not checkpoints:
        return None
    
    # Return the most recently modified checkpoint
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_best: bool = True,
    keep_last: bool = True,
    keep_n_latest: int = 0
):
    """
    Clean up old checkpoint files, keeping only specified ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Whether to keep best_model.pt
        keep_last: Whether to keep last_model.pt
        keep_n_latest: Number of latest epoch checkpoints to keep (0 = delete all)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    # Files to keep
    keep_files = set()
    if keep_best:
        keep_files.add('best_model.pt')
    if keep_last:
        keep_files.add('last_model.pt')
    
    # Get all epoch checkpoints
    epoch_checkpoints = sorted(
        checkpoint_dir.glob('epoch_*.pt'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    # Keep N latest epoch checkpoints
    for checkpoint in epoch_checkpoints[:keep_n_latest]:
        keep_files.add(checkpoint.name)
    
    # Delete other checkpoints
    for checkpoint in checkpoint_dir.glob('*.pt'):
        if checkpoint.name not in keep_files:
            checkpoint.unlink()