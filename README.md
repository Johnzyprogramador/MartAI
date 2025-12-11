# Modular ML Training Framework

A flexible, configuration-driven ML training framework for training autoencoders and other models on tabular data.

## Overview

This framework provides a modular system for:
- Training models (VAE, SAE, custom architectures)
- Extracting embeddings from trained models
- Running inference on test data
- Experiment tracking with MLflow

Everything is configured through YAML files - no need to modify code for different experiments.

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Running Your First Experiment

1. **Train a VAE:**
```bash
python scripts/train.py configs/examples/train_vae.yaml
```

2. **Extract embeddings:**
```bash
python scripts/extract_embeddings.py configs/examples/extract_embeddings.yaml
```

3. **Run predictions:**
```bash
python scripts/predict.py configs/examples/predict.yaml
```

## Project Structure
```
├── configs/              # YAML configuration files
│   ├── README.md        # Config documentation
│   └── examples/        # Example configs
├── src/                 # Source code
│   ├── models/         # Model implementations
│   ├── losses/         # Loss functions
│   ├── data/           # Data loaders
│   ├── runners/        # Training/inference runners
│   └── utils/          # Utilities
├── scripts/            # Entry point scripts
└── experiments/        # Generated outputs
```

## Key Features

### Configuration-Driven
- Define entire experiments in YAML
- No code changes needed for different experiments
- Configs saved with outputs for reproducibility

### Modular Design
- Plug-and-play models, losses, and data loaders
- Registry pattern for easy component addition
- Clear interfaces between components

### Flexible Data Loading
- Support for numerical, categorical, and embedding features
- Multiple data loader backends (PyTorch, WebDataset)
- Efficient batching and memory management

### Experiment Management
- Automatic checkpointing (best, last, all epochs)
- MLflow integration for experiment tracking
- Organized output directory structure

### Debugging Support
- Limit batches per epoch for quick testing
- Separate debug configs
- Detailed logging

## Example Use Case: VAE → SAE Pipeline

1. **Train VAE on raw data:**
```yaml
# configs/train_vae.yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/vae"

model:
  class: "VAE"
  params:
    latent_dim: 32

data:
  columns:
    numerical: [feat1, feat2, feat3]
    categorical: [cat1, cat2]
```

2. **Extract embeddings:**
```yaml
# configs/extract_vae_embeddings.yaml
checkpoint:
  path: "./experiments/vae/baseline/checkpoints/best_model.pt"

train:
  output: "./experiments/vae/baseline/embeddings/train_embeddings.npy"
val:
  output: "./experiments/vae/baseline/embeddings/val_embeddings.npy"
```

3. **Train SAE on embeddings:**
```yaml
# configs/train_sae.yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/sae"

train:
  data_path: "./experiments/vae/baseline/embeddings/train_embeddings.npy"
  loader_class: "EmbeddingDataLoader"
```

## Adding New Components

### New Model
1. Create file in `src/models/my_model.py`
2. Register with `@register_model("MyModel")`
3. Implement required interface (see `src/models/README.md`)
4. Use in config: `model.class: "MyModel"`

### New Loss
1. Create file in `src/losses/my_loss.py`
2. Register with `@register_loss("MyLoss")`
3. Implement required signature (see `src/losses/README.md`)
4. Use in config: `loss.class: "MyLoss"`

### New Data Loader
1. Create file in `src/data/my_loader.py`
2. Register with `@register_dataloader("MyLoader")`
3. Implement required interface (see `src/data/README.md`)
4. Use in config: `loader_class: "MyLoader"`

## Documentation

- **Architecture:** See `ARCHITECTURE.md` for design decisions and system overview
- **Configuration:** See `configs/README.md` for complete config reference
- **Scripts:** See `scripts/README.md` for how to run experiments
- **Components:** See individual README files in `src/` subdirectories

## Requirements

- Python 3.8+
- PyTorch 1.10+
- See `requirements.txt` for complete dependencies

## License

[Your License]

## Contributing

[Contributing guidelines if applicable]