"""Models for the framework."""

from .registry import MODELS, register_model
from .vae import VAE

__all__ = ['MODELS', 'register_model', 'VAE']