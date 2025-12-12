# Configuration Documentation

All experiments are defined using YAML configuration files. This directory contains config documentation and examples.

## Quick Links

- **[Training Configuration](training.md)** - How to configure training runs
- **[Embedding Extraction](embedding_extraction.md)** - How to extract embeddings from trained models
- **[Prediction Configuration](prediction.md)** - How to run predictions on test data

## Structure
```
configs/
├── README.md                    # This file
├── training.md                  # Training config reference
├── embedding_extraction.md      # Embedding extraction reference
├── prediction.md                # Prediction config reference
└── examples/                    # Example configs
    ├── train_vae.yaml
    ├── train_vae_debug.yaml
    ├── train_sae.yaml
    ├── extract_embeddings.yaml
    └── predict.yaml
```

## Basic Concepts

### Experiment Directory Structure

Configs specify where outputs are saved:
```yaml
experiment:
  name: "baseline"              # Experiment folder name
  output_dir: "./experiments/vae"   # Base directory
```

**Result:** `./experiments/vae/baseline/`

### Column Types

All configs must specify which columns are which type:
```yaml
data:
  columns:
    numerical: [...]    # Numerical features (float32)
    categorical: [...]  # Categorical features (int64)
    embeddings: [...]   # Pre-computed embeddings (float32)
```

### Registry Pattern

Models, losses, and data loaders are referenced by registered names:
```yaml
model:
  class: "VAE"  # Must match @register_model("VAE")

loss:
  class: "VAELoss"  # Must match @register_loss("VAELoss")

train:
  loader_class: "TabularDataLoader"  # Must match @register_dataloader(...)
```

## Quick Start

### 1. Training
```bash
python scripts/train.py configs/examples/train_vae.yaml
```
See [training.md](training.md) for full reference.

### 2. Extract Embeddings
```bash
python scripts/extract_embeddings.py configs/examples/extract_embeddings.yaml
```
See [embedding_extraction.md](embedding_extraction.md) for full reference.

### 3. Predict
```bash
python scripts/predict.py configs/examples/predict.yaml
```
See [prediction.md](prediction.md) for full reference.

## Example Workflow: VAE → SAE

1. **Train VAE:**
```yaml
# train_vae.yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/vae"
```

2. **Extract VAE embeddings:**
```yaml
# extract_vae_embeddings.yaml
checkpoint:
  path: "./experiments/vae/baseline/checkpoints/best_model.pt"
train:
  output: "./experiments/vae/baseline/embeddings/train_embeddings.npy"
```

3. **Train SAE on embeddings:**
```yaml
# train_sae.yaml
experiment:
  name: "baseline"
  output_dir: "./experiments/sae"
train:
  data_path: "./experiments/vae/baseline/embeddings/train_embeddings.npy"
  loader_class: "EmbeddingDataLoader"
```