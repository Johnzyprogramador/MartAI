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
                 format=None, target=None, **kwargs):
        """
        Initialize tabular data loader.
        
        Args:
            data_path: Path to data file
            batch_size: Number of samples per batch
            columns: Dict with keys 'numerical', 'categorical', 'embeddings'
                     These are exclusively INPUT features.
            cardinalities_path: Path to JSON with categorical mappings
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            num_batches_per_epoch: Limit batches per epoch (None = use all)
            format: Optional format override
            target: Dict defining target features, e.g. {'categorical': ['label']}
                   If None, defaults to self-supervised (inputs = targets).
            **kwargs: Additional arguments passed to pandas read function
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.num_batches_per_epoch = num_batches_per_epoch
        self.columns = columns
        self.target_config = target
        
        # Load dataframe
        self.df = self._load_dataframe(data_path, format, **kwargs)
        
        # Validate columns exist
        self._validate_columns()
        
        # Load cardinalities if categorical columns present
        self.cardinalities = None
        if cardinalities_path and (columns.get('categorical') or (target and target.get('categorical'))):
            with open(cardinalities_path, 'r') as f:
                self.cardinalities = json.load(f)
            self._validate_cardinalities()
        
        # Create PyTorch dataset and dataloader
        dataset = TabularDataset(self.df, columns, self.cardinalities, target_config=target)
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
        all_specified_columns = []
        
        # Input columns
        if self.columns.get('numerical'):
            all_specified_columns.extend(self.columns['numerical'])
        if self.columns.get('categorical'):
             all_specified_columns.extend(self.columns['categorical'])
        if self.columns.get('embeddings'):
             all_specified_columns.extend(self.columns['embeddings'])
             
        # Target columns if supervised
        if self.target_config:
            if self.target_config.get('numerical'):
                all_specified_columns.extend(self.target_config['numerical'])
            if self.target_config.get('categorical'):
                all_specified_columns.extend(self.target_config['categorical'])

        if not all_specified_columns:
            raise ValueError("No columns specified. At least one column type must be non-empty.")
        
        # Check for missing columns in dataframe
        missing_columns = set(all_specified_columns) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Columns not found in data: {missing_columns}")
    
    def _validate_cardinalities(self):
        """Validate that cardinalities are provided for all categorical columns."""
        if not self.cardinalities:
            return
        
        # Collect all categorical columns (input + target)
        required_cat_cols = []
        if self.columns.get('categorical'):
            required_cat_cols.extend(self.columns['categorical'])
            
        if self.target_config and self.target_config.get('categorical'):
             required_cat_cols.extend(self.target_config['categorical'])
        
        missing_cardinalities = set(required_cat_cols) - set(self.cardinalities.keys())
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
    
    def __init__(self, df, columns, cardinalities, target_config=None):
        """
        Initialize dataset.
        
        Args:
            df: pandas DataFrame
            columns: Dict with 'numerical', 'categorical', 'embeddings' keys (Inputs)
            cardinalities: Dict mapping categorical columns to value mappings
            target_config: Dict defining target features (Supervised) or None (Self-Supervised)
        """
        self.df = df
        self.columns = columns
        self.cardinalities = cardinalities
        self.target_config = target_config
        
        # Prepare INPUT data tensors 
        # (We use self.columns directly as they now strictly define inputs)
        self.numerical = self._prepare_numerical()
        self.categorical = self._prepare_categorical()
        self.embeddings = self._prepare_embeddings()
        
        # Prepare TARGET tensors if supervised
        print(f"DEBUG: TabularDataset init target_config: {target_config}")
        self.targets = self._prepare_targets() if target_config else None
        print(f"DEBUG: TabularDataset generated targets keys: {self.targets.keys() if self.targets else 'None'}")
    
    def _prepare_numerical(self):
        """Prepare numerical features as tensor."""
        if not self.columns.get('numerical'):
            return None
        
        numerical_data = self.df[self.columns['numerical']].values
        return torch.FloatTensor(numerical_data)
    
    def _prepare_categorical(self):
        """Prepare categorical features as tensor."""
        if not self.columns.get('categorical'):
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
        if not self.columns.get('embeddings'):
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

    def _prepare_targets(self):
        """Prepare target features as dict of tensors."""
        targets = {}
        
        # Numerical targets
        if self.target_config.get('numerical'):
             num_data = self.df[self.target_config['numerical']].values
             targets['numerical'] = torch.FloatTensor(num_data)
             
        # Categorical targets
        if self.target_config.get('categorical'):
             cat_data = self.df[self.target_config['categorical']].copy()
             for col in self.target_config['categorical']:
                  if col in self.cardinalities:
                        cat_data[col] = cat_data[col].map(self.cardinalities[col])
                        if cat_data[col].isna().any():
                             raise ValueError(f"Unmapped values in target '{col}'")
             targets['categorical'] = torch.LongTensor(cat_data.values)
             
        return targets
    
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

        if self.categorical is not None:
            batch['categorical'] = self.categorical[idx]

        if self.embeddings is not None:
            batch['embeddings'] = self.embeddings[idx]

        # Targets
        if self.targets:
            # Supervised: return specific target dict
            # We need to slice each tensor in the dictionary
            batch['targets'] = {
                k: v[idx] for k, v in self.targets.items()
            }
        else:
            # Self-supervised: targets reference the same tensors (memory efficient)
            batch['targets'] = {}
            if self.numerical is not None:
                batch['targets']['numerical'] = batch['numerical']
            if self.categorical is not None:
                batch['targets']['categorical'] = batch['categorical']
            if self.embeddings is not None:
                batch['targets']['embeddings'] = batch['embeddings']
        
        # Optional metadata
        batch['metadata'] = {'index': idx}
        
        return batch