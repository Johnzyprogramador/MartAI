"""
VAE loss function: reconstruction loss + KL divergence.
"""

import torch
import torch.nn.functional as F
from .registry import register_loss


@register_loss("VAELoss")
def vae_loss(model_output, targets, **kwargs):
    """
    Compute VAE loss = reconstruction loss + beta * KL divergence.
    
    Reconstruction loss:
    - Numerical features: MSE loss (mean per dimension)
    - Categorical features: Cross-entropy loss (mean across features)
    
    Args:
        model_output: Dict with:
            - 'predictions': Dict with 'numerical' and 'categorical'
            - 'mu': Latent mean [B, latent_dim]
            - 'logvar': Latent log variance [B, latent_dim]
        targets: Dict with:
            - 'numerical': Target numerical features [B, num_features] or None
            - 'categorical': Target categorical features [B, cat_features] or None
        **kwargs: Additional unused parameters
    
    Returns:
        Scalar loss tensor
    """
    predictions = model_output['predictions']
    mu = model_output['mu']
    logvar = model_output['logvar']
    
    total_losses = []
    
    # --- Numerical reconstruction loss ---
    num_preds = predictions.get('numerical')
    num_targets = targets.get('numerical')
    
    if num_preds is not None and num_targets is not None:
        # MSE loss: mean per dimension -> [B]
        num_loss = F.mse_loss(num_preds, num_targets, reduction='none').mean(dim=1)
        total_losses.append(num_loss)
    
    # --- Categorical reconstruction loss ---
    cat_preds = predictions.get('categorical')
    cat_targets = targets.get('categorical')
    
    if cat_preds and cat_targets is not None and len(cat_preds) > 0:
        cat_losses = []
        for i, feature_preds in enumerate(cat_preds):
            # Get target for this categorical feature
            feature_targets = cat_targets[:, i].long()
            # Cross-entropy: gives loss per sample
            loss = F.cross_entropy(feature_preds, feature_targets, reduction='none')
            cat_losses.append(loss)
        
        # Average across features -> [B]
        cat_loss = torch.stack(cat_losses, dim=1).mean(dim=1)
        total_losses.append(cat_loss)
    
    # Check that we have at least one valid loss component
    if not total_losses:
        raise ValueError("No valid features found for loss computation.")
    
    # Reconstruction loss: average across loss components -> [B]
    recon_loss = torch.stack(total_losses, dim=1).mean(dim=1)
    
    # --- KL divergence: mean per dimension -> [B] ---
    kl_divergence = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1
    )
    
    # Total loss: weighted sum, then average over batch
    total_loss = recon_loss + kl_divergence
    
    return total_loss