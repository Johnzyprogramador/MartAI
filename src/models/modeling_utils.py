"""
Modeling utilities: feature encoders, layer dimension helpers, MLP blocks.
"""

import math
import numpy as np
import torch
import torch.nn as nn


def get_post_encoder_dim(
    dims: list[int] | int,
    log2_max_dim: int = 9,
    log2_last_dim: int = 5,
    log2_step: int = 1,
) -> list[int]:
    """
    Get the dimensions of encoder/decoder layers.
    
    Args:
        dims: Number of layers or list of layer sizes
        log2_max_dim: Maximum dimension (log2), default 9 (512)
        log2_last_dim: Log2 of the smallest layer, default 5 (32)
        log2_step: Step between layers (log2), default 1
    
    Returns:
        List of layer sizes
    
    Examples:
        >>> get_post_encoder_dim(3, log2_last_dim=5)
        [128, 64, 32]
    """
    if isinstance(dims, list):
        return dims
    
    if isinstance(dims, int):
        num_layers = dims
        log2_dims = np.minimum(
            np.arange(
                log2_last_dim,
                log2_last_dim + (num_layers * log2_step),
                log2_step,
            )[::-1],
            log2_max_dim,
        )
        return (2**log2_dims).tolist()
    
    raise ValueError(
        "dims must be either a list of layer sizes or an integer for number of layers"
    )


class CategoricalFeatureEncoder(nn.Module):
    """
    Embeds a single categorical feature.
    
    Embedding dimension is computed as: 2 * ceil(cardinality^0.25)
    This heuristic balances expressiveness with parameter efficiency.
    """
    
    def __init__(self, cardinality: int):
        super().__init__()
        output_dim = CategoricalFeatureEncoder._compute_embedding_dim(cardinality)
        self.embedding = nn.Embedding(cardinality, output_dim)
    
    @staticmethod
    def _compute_embedding_dim(cardinality: int) -> int:
        """Compute embedding dimension from cardinality."""
        if cardinality > 2:
            return 2 * int(math.ceil(cardinality ** 0.25))
        else:
            return 1
    
    @property
    def output_dim(self) -> int:
        """Output dimension of this encoder."""
        return self.embedding.embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Categorical indices [B]
        
        Returns:
            Embeddings [B, output_dim]
        """
        return self.embedding(x)


class CategoricalFeaturesEncoder(nn.Module):
    """
    Embeds multiple categorical features and concatenates them.
    """
    
    def __init__(self, input_dim_cat: int, cardinalities: list[int]):
        super().__init__()
        if input_dim_cat <= 0:
            raise ValueError("input_dim_cat must be greater than 0")
        
        if len(cardinalities) != input_dim_cat:
            raise ValueError(
                f"Number of cardinalities ({len(cardinalities)}) must match "
                f"input_dim_cat ({input_dim_cat})"
            )
        
        self.input_dim_cat = input_dim_cat
        self.categorical_encoder = nn.ModuleList([
            CategoricalFeatureEncoder(cardinality) 
            for cardinality in cardinalities
        ])
    
    @property
    def output_dim(self) -> int:
        """Total output dimension after concatenating all embeddings."""
        return sum(enc.output_dim for enc in self.categorical_encoder)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Categorical features [B, input_dim_cat]
        
        Returns:
            Concatenated embeddings [B, output_dim]
        """
        cat_embeddings = [
            self.categorical_encoder[i](x[:, i]) 
            for i in range(self.input_dim_cat)
        ]
        return torch.cat(cat_embeddings, dim=-1)


class FeaturesEncoder(nn.Module):
    """
    Encodes numerical, categorical, and embedding features into a single vector.
    
    - Numerical features: passed through directly
    - Categorical features: embedded then concatenated
    - Embeddings: passed through directly
    """
    
    def __init__(
        self,
        input_dim_num: int,
        input_dim_cat: int,
        cardinalities: list[int],
        input_dim_emb: int = 0
    ):
        super().__init__()
        self.input_dim_num = input_dim_num
        self.input_dim_cat = input_dim_cat
        self.input_dim_emb = input_dim_emb
        
        # Categorical encoder (if needed)
        if input_dim_cat > 0:
            self.categorical_encoder = CategoricalFeaturesEncoder(
                input_dim_cat, cardinalities
            )
            self.cat_emb_dims = [
                enc.output_dim 
                for enc in self.categorical_encoder.categorical_encoder
            ]
        else:
            self.categorical_encoder = None
            self.cat_emb_dims = []
    
    @property
    def output_dim(self) -> int:
        """Total output dimension after encoding all features."""
        total = self.input_dim_num + self.input_dim_emb
        if self.input_dim_cat > 0:
            total += self.categorical_encoder.output_dim
        return total
    
    def forward(self, batch: dict) -> torch.Tensor:
        """
        Encode all features from batch.
        
        Args:
            batch: Dict with keys 'numerical', 'categorical', 'embeddings'
        
        Returns:
            Concatenated features [B, output_dim]
        """
        features = []
        
        if batch.get('numerical') is not None:
            features.append(batch['numerical'])
        
        if batch.get('categorical') is not None:
            cat_emb = self.categorical_encoder(batch['categorical'])
            features.append(cat_emb)
        
        if batch.get('embeddings') is not None:
            features.append(batch['embeddings'])
        
        if not features:
            raise ValueError("Batch contains no valid features")
        
        return torch.cat(features, dim=-1)


class MLPBlock(nn.Module):
    """
    Multi-layer perceptron with configurable activation.
    """
    
    def __init__(self, layer_dims: list[int], activation_fn=nn.ReLU):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_fn())
        # Remove last activation
        self.net = nn.Sequential(*layers[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)