# test_config_utils.py
from src.utils import load_config, save_config, validate_config, create_experiment_dirs
import yaml

# Create a test config
config = {
    'experiment': {
        'name': 'test_vae',
        'output_dir': './experiments/vae',
        'seed': 42,
        'device': 'cuda'
    },
    'model': {
        'class': 'VAE',
        'params': {
            'num_layers': 3,
            'log2_last_dim': 5,
            'cardinalities': [2, 2],
            'input_dim_num': 2,
            'input_dim_cat': 2,
            'input_dim_emb': 0
        }
    },
    'data': {
        'columns': {
            'numerical': ['amount', 'distance'],
            'categorical': ['channel', 'merchant'],
            'embeddings': []
        },
        'cardinalities_path': './data/cardinalities.json'
    },
    'train': {
        'data_path': './data/train.csv',
        'loader_class': 'TabularDataLoader',
        'loader_params': {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': 4
        },
        'epochs': 100,
        'optimizer_config': {
            'optimizer': 'Adam',
            'learning_rate': 0.001
        },
        'loss': {
            'class': 'VAELoss',
            'params': {
                'loss_factor': 1.0,
                'beta': 1.0
            }
        }
    },
    'checkpoint': {
        'metric': 'val_loss',
        'mode': 'min',
        'save_strategy': 'best'
    }
}

# Test save
save_config(config, 'test_config.yaml')
print("✓ Config saved")

# Test load
loaded_config = load_config('test_config.yaml')
print("✓ Config loaded")

# Test validate
try:
    validate_config(loaded_config, config_type='train')
    print("✓ Config validation passed")
except ValueError as e:
    print(f"✗ Config validation failed: {e}")

# Test create directories
paths = create_experiment_dirs('./experiments/vae', 'test_vae')
print("✓ Experiment directories created:")
for key, path in paths.items():
    print(f"  {key}: {path}")

# Test invalid config
invalid_config = {'experiment': {'name': 'test'}}  # Missing required fields
try:
    validate_config(invalid_config, config_type='train')
    print("✗ Should have failed validation")
except ValueError as e:
    print(f"✓ Invalid config correctly rejected: {e}")