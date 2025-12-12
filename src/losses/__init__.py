"""Loss functions for the framework."""

from .registry import LOSSES, register_loss
from .vae_loss import vae_loss

__all__ = ['LOSSES', 'register_loss', 'vae_loss']