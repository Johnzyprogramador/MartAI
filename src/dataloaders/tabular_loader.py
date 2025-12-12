"""
Tabular data loader for CSV, Parquet, Feather, and other tabular formats.
"""

import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .registry import register_dataloader


@register_dataloader("TabularDataLoader")
class TabularDataLoader:
    """
    Loads tabular data from various formats (CSV, Parquet, Feather, etc.).
    Format is auto-detected from file extension.
    
    Supports:
        - CSV (.csv)
        - Parquet (.parquet, .pq)
        - Feather (.feather)
        - Pickle (.pkl, .pickle)
    """
    
    SUPPORTED_FORMATS = {
        '.csv': pd.read_csv,
        '.parquet': pd.read_parquet,
        '.pq': pd.read_parquet,
        '.feather': pd.read_feather,
        '.pkl': pd.read_pickle,
        '.pickle': pd.read_pickle,
    }
    
    def __init__(self, data_path, batch_size, columns, 
                 cardinalities_path=None, shuffle=False,
                 num_workers=0, num_batches_per_epoch=None,
                 format=None, **kwargs):
        """
        Initialize tabular data loader.
        
        Args:
            data_path: Path to data file
            batch_size: Number of samples per batch
            columns: Dict with keys 'numerical', 'categorical', 'embeddings'
                     Each is a list of column names
            cardinalities_path: Path to JSON with categorical mappings (optional)
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            num_batches_per_epoch: Limit batches per epoch (None = use all)
            format: Optional format override (e.g., 'csv', 'parquet')
                   If None, auto-detected from extension
            **kwargs: Additional arguments passed to pandas read function
                     (e.g., sep=';' for CSV, engine='pyarrow' for Parquet)
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.num_batches_per_epoch = num_batches_per_epoch
        self.columns = columns
        
        # Load dataframe
        self.df = self._load_dataframe(data_path, format, **kwargs)
        
        # Validate columns exist
        self._validate_columns()
        
        # Load cardinalities if categorical columns present
        self.cardinalities = None
        if cardinalities_path and columns['categorical']:
            with open(cardinalities_path, 'r') as f:
                self.cardinalities = json.load(f)
            self._validate_cardinalities()
        
        # Create PyTorch dataset and dataloader
        dataset = TabularDataset(self.df, columns, self.cardinalities)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def _load_dataframe(self, data_path, format=None, **kwargs):
        """
        Load dataframe with format auto-detection.
        
        Args:
            data_path: Path to data file
            format: Optional format override
            **kwargs: Additional arguments for pandas read function
        
        Returns:
            pandas.DataFrame
        """
        if format is not None:
            # Explicit format provided
            ext = f".{format}"
        else:
            # Auto-detect from extension
            ext = os.path.splitext(data_path)[1].lower()
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {ext}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        # Load with appropriate function
        read_fn = self.SUPPORTED_FORMATS[ext]
        
        try:
            df = read_fn(data_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load {data_path}: {str(e)}")
        
        return df
    
    def _validate_columns(self):
        """Validate that all specified columns exist in dataframe."""
        all_specified_columns = (
            self.columns['numerical'] + 
            self.columns['categorical'] + 
            self.columns['embeddings']
        )
        
        if not all_specified_columns:
            raise ValueError("No columns specified. At least one column type must be non-empty.")
        
        missing_columns = set(all_specified_columns) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Columns not found in data: {missing_columns}")
    
    def _validate_cardinalities(self):
        """Validate that cardinalities are provided for all categorical columns."""
        if not self.cardinalities:
            return
        
        missing_cardinalities = set(self.columns['categorical']) - set(self.cardinalities.keys())
        if missing_cardinalities:
            raise ValueError(
                f"Cardinalities missing for categorical columns: {missing_cardinalities}"
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


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    
    Handles numerical, categorical, and embedding features.
    """
    
    def __init__(self, df, columns, cardinalities):
        """
        Initialize dataset.
        
        Args:
            df: pandas DataFrame
            columns: Dict with 'numerical', 'categorical', 'embeddings' keys
            cardinalities: Dict mapping categorical columns to value mappings
        """
        self.df = df
        self.columns = columns
        self.cardinalities = cardinalities
        
        # Prepare data tensors
        self.numerical = self._prepare_numerical()
        self.categorical = self._prepare_categorical()
        self.embeddings = self._prepare_embeddings()
    
    def _prepare_numerical(self):
        """Prepare numerical features as tensor."""
        if not self.columns['numerical']:
            return None
        
        numerical_data = self.df[self.columns['numerical']].values
        return torch.FloatTensor(numerical_data)
    
    def _prepare_categorical(self):
        """Prepare categorical features as tensor."""
        if not self.columns['categorical']:
            return None
        
        # Map categorical values to integers using cardinalities
        cat_data = self.df[self.columns['categorical']].copy()
        
        for col in self.columns['categorical']:
            if col in self.cardinalities:
                cat_data[col] = cat_data[col].map(self.cardinalities[col])
                
                # Check for unmapped values (NaN after mapping)
                if cat_data[col].isna().any():
                    unmapped = self.df[col][cat_data[col].isna()].unique()
                    raise ValueError(
                        f"Unmapped values in column '{col}': {unmapped}. "
                        f"Update cardinalities mapping."
                    )
        
        return torch.LongTensor(cat_data.values)
    
    def _prepare_embeddings(self):
        """Prepare embedding features as tensor."""
        if not self.columns['embeddings']:
            return None
        
        # Embeddings might be stored as strings/lists - need to parse
        embedding_data = self.df[self.columns['embeddings']].values
        
        # If embeddings are stored as strings, parse them
        # This handles formats like "[0.1, 0.2, 0.3]"
        if isinstance(embedding_data[0, 0], str):
            import ast
            embedding_data = [[ast.literal_eval(cell) for cell in row] 
                            for row in embedding_data]
        
        return torch.FloatTensor(embedding_data)
    
    def __len__(self):
        """Return number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict: Batch dictionary in standard format
        """
        batch = {}
        
        # Add features
        if self.numerical is not None:
            batch['numerical'] = self.numerical[idx]
        else:
            batch['numerical'] = None
        
        if self.categorical is not None:
            batch['categorical'] = self.categorical[idx]
        else:
            batch['categorical'] = None
        
        if self.embeddings is not None:
            batch['embeddings'] = self.embeddings[idx]
        else:
            batch['embeddings'] = None
        
        # For self-supervised: targets reference the same tensors (memory efficient)
        batch['targets'] = {
            'numerical': batch['numerical'],
            'categorical': batch['categorical'],
            'embeddings': batch['embeddings']
        }
        
        # Optional metadata
        batch['metadata'] = {'index': idx}
        
        return batch