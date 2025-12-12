"""
Variational Autoencoder (VAE) for tabular data.
"""

import torch
import torch.nn as nn
from typing import NamedTuple
from .registry import register_model
from .modeling_utils import (
    FeaturesEncoder,
    MLPBlock,
    get_post_encoder_dim
)


class EncoderOutput(NamedTuple):
    """Output from VAE encoder."""
    last_hidden_state: torch.Tensor  # mu
    variances: torch.Tensor          # logvar


class Encoder(nn.Module):
    """
    VAE encoder: maps input to latent distribution parameters (mu, logvar).
    """
    
    def __init__(self, input_dim: int, hidden_dims: list[int], activation_fn=nn.ReLU):
        super().__init__()
        self.encoder_body = MLPBlock([input_dim] + hidden_dims, activation_fn)
        
        last_hidden_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.fc_mu = nn.Linear(last_hidden_dim, hidden_dims[-1])
        self.fc_logvar = nn.Linear(last_hidden_dim, hidden_dims[-1])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Xavier initialization for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """
        Args:
            x: Input features [B, input_dim]
        
        Returns:
            EncoderOutput with mu and logvar
        """
        x = self.encoder_body(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return EncoderOutput(last_hidden_state=mu, variances=logvar)


class Decoder(nn.Module):
    """
    VAE decoder: maps latent code back to reconstruction space.
    """
    
    def __init__(self, output_dim: int, hidden_dims: list[int], activation_fn=nn.ReLU):
        super().__init__()
        self.decoder_body = MLPBlock(hidden_dims + [output_dim], activation_fn)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Xavier initialization for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent code [B, latent_dim]
        
        Returns:
            Reconstructed features [B, output_dim]
        """
        return self.decoder_body(z)


@register_model("VAE")
class VAE(nn.Module):
    """
    Variational Autoencoder for tabular data.
    
    Supports:
    - Numerical features (reconstructed with MSE)
    - Categorical features (embedded, reconstructed with cross-entropy)
    - Pre-computed embeddings
    """
    
    def __init__(
        self,
        num_layers: int,
        log2_last_dim: int,
        cardinalities: list[int] = None,
        input_dim_num: int = 0,
        input_dim_cat: int = 0,
        input_dim_emb: int = 0,
        activation_fn=nn.ReLU
    ):
        """
        Initialize VAE.
        
        Args:
            num_layers: Number of encoder/decoder layers
            log2_last_dim: Log2 of the latent dimension
            cardinalities: List of cardinalities for categorical features
                          (must match order of categorical columns)
            input_dim_num: Number of numerical features
            input_dim_cat: Number of categorical features
            input_dim_emb: Dimension of pre-computed embeddings
            activation_fn: Activation function (default: ReLU)
        """
        super().__init__()
        
        # Store config
        self.num_layers = num_layers
        self.log2_last_dim = log2_last_dim
        self.input_dim_num = input_dim_num
        self.input_dim_cat = input_dim_cat
        self.input_dim_emb = input_dim_emb
        self.cardinalities = cardinalities or []
        
        # Features encoder
        self.features_encoder = FeaturesEncoder(
            input_dim_num, input_dim_cat, self.cardinalities, input_dim_emb
        )
        
        # Encoder/decoder layer dimensions
        encoder_layers = get_post_encoder_dim(num_layers, log2_last_dim=log2_last_dim)
        decoder_layers = encoder_layers[::-1]
        
        # Encoder and decoder
        self.encoder = Encoder(
            self.features_encoder.output_dim, 
            encoder_layers, 
            activation_fn
        )
        self.decoder = Decoder(
            self.features_encoder.output_dim, 
            decoder_layers, 
            activation_fn
        )
        
        # Categorical reconstruction heads
        if input_dim_cat > 0:
            self.cat_emb_dims = self.features_encoder.cat_emb_dims
            self.cat_heads = nn.ModuleList([
                nn.Linear(emb_dim, card)
                for emb_dim, card in zip(self.cat_emb_dims, cardinalities)
            ])
        else:
            self.cat_emb_dims = []
            self.cat_heads = nn.ModuleList([])
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        During training: sample from N(mu, std)
        During inference: return mu (deterministic)
        
        Args:
            mu: Mean of latent distribution [B, latent_dim]
            logvar: Log variance of latent distribution [B, latent_dim]
        
        Returns:
            Latent code z [B, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def forward(self, batch: dict) -> dict:
        """
        Forward pass through VAE.
        
        Args:
            batch: Dict with 'numerical', 'categorical', 'embeddings'
        
        Returns:
            Dict with:
            - 'predictions': Dict with 'numerical' (tensor or None) and 
                            'categorical' (list of logits or [])
            - 'mu': Latent mean [B, latent_dim]
            - 'logvar': Latent log variance [B, latent_dim]
        """
        # Encode features
        x_emb = self.features_encoder(batch)
        
        # Get latent distribution parameters
        output = self.encoder(x_emb)
        mu, logvar = output.last_hidden_state, output.variances
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        # Split reconstruction into components
        predictions = {}
        
        # Numerical reconstruction
        if self.input_dim_num > 0:
            predictions['numerical'] = x_reconstructed[:, :self.input_dim_num]
        else:
            predictions['numerical'] = None
        
        # Categorical reconstruction (apply heads to get logits)
        if self.cat_emb_dims:
            cat_logits = []
            start = self.input_dim_num
            for emb_dim, head in zip(self.cat_emb_dims, self.cat_heads):
                end = start + emb_dim
                emb_recon = x_reconstructed[:, start:end]
                logits = head(emb_recon)  # [B, cardinality]
                cat_logits.append(logits)
                start = end
            predictions['categorical'] = cat_logits
        else:
            predictions['categorical'] = []
        
        return {
            'predictions': predictions,
            'mu': mu,
            'logvar': logvar
        }
    
    def extract_embeddings(self, batch: dict) -> torch.Tensor:
        """
        Extract latent embeddings (mu) from input batch.
        
        Used for embedding extraction runner.
        
        Args:
            batch: Dict with 'numerical', 'categorical', 'embeddings'
        
        Returns:
            Latent embeddings (mu) [B, latent_dim]
        """
        x_emb = self.features_encoder(batch)
        output = self.encoder(x_emb)
        return output.last_hidden_state  # Return mu