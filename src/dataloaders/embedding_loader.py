"""
Embedding data loader for pre-computed embeddings stored in .npy files.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .registry import register_dataloader


@register_dataloader("EmbeddingDataLoader")
class EmbeddingDataLoader:
    """
    Loads pre-computed embeddings from .npy files.
    
    Used for training models on embeddings (e.g., SAE on VAE embeddings).
    """
    
    def __init__(self, data_path, batch_size, shuffle=False,
                 num_workers=0, num_batches_per_epoch=None, **kwargs):
        """
        Initialize embedding data loader.
        
        Args:
            data_path: Path to .npy file containing embeddings
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            num_batches_per_epoch: Limit batches per epoch (None = use all)
            **kwargs: Additional unused parameters (for compatibility)
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.num_batches_per_epoch = num_batches_per_epoch
        
        # Load embeddings
        try:
            embeddings = np.load(data_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load embeddings from {data_path}: {str(e)}")
        
        # Validate shape
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings array (n_samples, embedding_dim), "
                f"got shape {embeddings.shape}"
            )
        
        self.embeddings = torch.FloatTensor(embeddings)
        
        # Create PyTorch dataset and dataloader
        dataset = EmbeddingDataset(self.embeddings)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def __iter__(self):
        """Yield batches in standard format."""
        batch_count = 0
        for batch in self.dataloader:
            yield batch
            batch_count += 1
            
            # Stop early if num_batches_per_epoch is set
            if self.num_batches_per_epoch is not None:
                if batch_count >= self.num_batches_per_epoch:
                    break
    
    def __len__(self):
        """Return number of batches per epoch."""
        total_batches = len(self.dataloader)
        if self.num_batches_per_epoch is not None:
            return min(self.num_batches_per_epoch, total_batches)
        return total_batches


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for embeddings.
    """
    
    def __init__(self, embeddings):
        """
        Initialize dataset.
        
        Args:
            embeddings: torch.FloatTensor of shape (n_samples, embedding_dim)
        """
        self.embeddings = embeddings
    
    def __len__(self):
        """Return number of samples."""
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict: Batch dictionary in standard format
        """
        emb = self.embeddings[idx]
        
        # Return in standard batch format
        # For embeddings: numerical and categorical are None
        return {
            'numerical': None,
            'categorical': None,
            'embeddings': emb,
            'targets': {
                'numerical': None,
                'categorical': None,
                'embeddings': emb  # Self-supervised: same reference
            },
            'metadata': {'index': idx}
        }