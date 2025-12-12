"""Data loaders for the framework."""

from .registry import DATA_LOADERS, register_dataloader
from .tabular_loader import TabularDataLoader
from .embedding_loader import EmbeddingDataLoader

__all__ = [
    'DATA_LOADERS', 
    'register_dataloader', 
    'TabularDataLoader',
    'EmbeddingDataLoader'
]