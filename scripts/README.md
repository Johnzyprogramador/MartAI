# Scripts Documentation

Entry point scripts for running experiments. Each script corresponds to a specific task.

## Available Scripts

- **`train.py`** - Train models
- **`extract_embeddings.py`** - Extract embeddings from trained models
- **`predict.py`** - Run predictions on test data

## Usage

### Training
```bash
python scripts/train.py <path_to_config.yaml>
```

**Example:**
```bash
python scripts/train.py configs/examples/train_vae.yaml
```

**What it does:**
1. Loads configuration
2. Initializes model, data loaders, optimizer, loss
3. Runs training loop with validation (if configured)
4. Saves checkpoints according to strategy
5. Logs metrics to MLflow (if configured)

**Outputs:**
```
{output_dir}/{name}/
├── config.yaml           # Copy of config used
├── checkpoints/
│   ├── best_model.pt
│   └── last_model.pt
└── logs/
    └── training.log
```

**See:** `configs/training.md` for configuration reference.

---

### Extracting Embeddings
```bash
python scripts/extract_embeddings.py <path_to_config.yaml>
```

**Example:**
```bash
python scripts/extract_embeddings.py configs/examples/extract_embeddings.yaml
```

**What it does:**
1. Loads trained model from checkpoint
2. Runs `model.extract_embeddings()` on specified data splits
3. Saves embeddings as `.npy` files

**Outputs:**
```
{output_dir}/{name}/embeddings/
├── train_embeddings.npy
├── val_embeddings.npy
└── test_embeddings.npy
```

**Notes:**
- Only extracts splits specified in config (train/val/test are all optional)
- Uses full dataset (no batching limits)
- Embeddings saved as NumPy arrays: `(n_samples, embedding_dim)`

**See:** `configs/embedding_extraction.md` for configuration reference.

---

### Running Predictions
```bash
python scripts/predict.py <path_to_config.yaml>
```

**Example:**
```bash
python scripts/predict.py configs/examples/predict.yaml
```

**What it does:**
1. Loads trained model from checkpoint
2. Runs inference on test data
3. Optionally computes metrics
4. Saves predictions as `.npy` file

**Outputs:**
```
{output_dir}/{name}/predictions/
└── test_predictions.npy
```

**Notes:**
- Predictions contain full model output dictionary
- Metrics are logged but not saved to file
- Uses full test dataset

**See:** `configs/prediction.md` for configuration reference.

---

## Common Workflows

### Workflow 1: Train VAE → Extract Embeddings
```bash
# Step 1: Train VAE
python scripts/train.py configs/train_vae.yaml

# Step 2: Extract embeddings
python scripts/extract_embeddings.py configs/extract_vae_embeddings.yaml
```

### Workflow 2: VAE → SAE Pipeline
```bash
# Step 1: Train VAE
python scripts/train.py configs/train_vae.yaml

# Step 2: Extract VAE embeddings
python scripts/extract_embeddings.py configs/extract_vae_embeddings.yaml

# Step 3: Train SAE on embeddings
python scripts/train.py configs/train_sae.yaml

# Step 4: Extract SAE embeddings
python scripts/extract_embeddings.py configs/extract_sae_embeddings.yaml
```

### Workflow 3: Train → Evaluate
```bash
# Step 1: Train model
python scripts/train.py configs/train_vae.yaml

# Step 2: Run predictions on test set
python scripts/predict.py configs/predict_vae.yaml
```

---

## Debugging

### Quick Test Run

Use debug configs with limited batches:
```bash
python scripts/train.py configs/examples/train_vae_debug.yaml
```

Debug config should have:
- `num_batches_per_epoch: 10` (small number)
- `epochs: 3` (few epochs)
- `save_strategy: "last"` (save space)

### Verify Configuration

Check that config is valid before running:
```bash
python -c "import yaml; yaml.safe_load(open('configs/train_vae.yaml'))"
```

### Monitor Training

Training logs are written to:
```
{output_dir}/{name}/logs/training.log
```

Watch in real-time:
```bash
tail -f experiments/vae/baseline/logs/training.log
```

---

## Batch Processing

### Run Multiple Experiments
```bash
# Run all configs in a directory
for config in configs/sweep/*.yaml; do
  python scripts/train.py $config
done
```

### Parallel Execution
```bash
# Run multiple experiments in parallel (be careful with GPU memory!)
python scripts/train.py configs/experiment1.yaml &
python scripts/train.py configs/experiment2.yaml &
wait
```

---

## Error Handling

### Common Errors

**"Config file not found"**
- Check path is correct
- Use absolute or relative path from project root

**"Model class not found"**
- Check model is registered with `@register_model()`
- Check model is imported in `src/models/__init__.py`

**"CUDA out of memory"**
- Reduce `batch_size`
- Use `device: "cpu"`
- Close other programs using GPU

**"Checkpoint not found"**
- Check checkpoint path in config
- Verify training completed successfully
- Check `save_strategy` in training config

**"Data file not found"**
- Check `data_path` in config
- Use absolute or relative path from project root
- Verify file exists

---

## Tips

### Reproducibility

The config is automatically saved with outputs:
```bash
# Reproduce an experiment
python scripts/train.py experiments/vae/baseline/config.yaml
```

### Using Relative Paths

All paths in configs should be relative to project root:
```yaml
# Good
data_path: "./data/train.csv"
output_dir: "./experiments/vae"

# Avoid absolute paths (not portable)
data_path: "/home/user/project/data/train.csv"
```

### Organizing Experiments

Group related experiments:
```bash
# All VAE experiments
python scripts/train.py configs/vae/baseline.yaml
python scripts/train.py configs/vae/high_beta.yaml
python scripts/train.py configs/vae/large_latent.yaml

# All SAE experiments
python scripts/train.py configs/sae/baseline.yaml
python scripts/train.py configs/sae/high_sparsity.yaml
```

### MLflow Tracking

If MLflow is configured, view experiments at:
```
{mlflow.tracking_uri}
```

Filter by experiment name to see related runs.

---

## Script Arguments

All scripts accept a single argument: the path to the config file.
```bash
python scripts/train.py <config_path>
python scripts/extract_embeddings.py <config_path>
python scripts/predict.py <config_path>
```

No additional command-line arguments are supported - everything is configured in YAML.