# test_tabular_loader.py
from src.dataloaders import DATA_LOADERS
from src.models import MODELS
from src.losses import LOSSES
import pandas as pd
import json
import torch
import torch.optim as optim

# Create test data
df = pd.DataFrame({
    'amount': [10.5, 20.3, 15.7, 5.0, 30.1, 12.2], # Added more data for 4 batches of size 2 to work better
    'distance': [5.2, 10.1, 7.5, 2.1, 15.3, 6.8],
    'channel': ['online', 'store', 'online', 'store', 'store', 'online'],
    'merchant': ['restaurant', 'retail', 'restaurant', 'retail', 'restaurant', 'retail']
})
df.to_csv('test_data.csv', index=False)

# Create cardinalities
cardinalities = {
    'channel': {'online': 0, 'store': 1},
    'merchant': {'restaurant': 0, 'retail': 1}
}
with open('cardinalities.json', 'w') as f:
    json.dump(cardinalities, f)

# Test loader
loader = DATA_LOADERS["TabularDataLoader"](
    data_path='test_data.csv',
    batch_size=2,
    columns={
        'numerical': ['amount', 'distance'],
        'categorical': ['channel', 'merchant'],
        'embeddings': []
    },
    cardinalities_path='cardinalities.json',
    shuffle=True # Shuffle to get different batches
)

# Create VAE
model = MODELS["VAE"](
    num_layers=3,
    log2_last_dim=5,
    cardinalities=[2, 2],
    input_dim_num=2,
    input_dim_cat=2,
    input_dim_emb=0
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = LOSSES["VAELoss"]

print("Starting training loop...")
model.train()

# Train for a few steps
data_iter = iter(loader)
for i in range(4):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)
        
    optimizer.zero_grad()
    
    # Forward pass
    output = model(batch)
    
    # Compute loss
    loss = loss_fn(
        model_output=output,
        targets=batch['targets'],
    )
    
    # Handle loss if it's not a scalar (since vae_loss returns valid reduction='none' equivalent sometimes)
    if loss.ndim > 0:
        loss = loss.mean()
    
    print(f"Step {i+1} - Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    optimizer.step()

print("Training finished!")

print("\n=== Testing Supervised Mode ===")
# Test loader with target column
loader_sup = DATA_LOADERS["TabularDataLoader"](
    data_path='test_data.csv',
    batch_size=2,
    columns={
        'numerical': ['amount', 'distance'],
        'categorical': ['channel', 'merchant'],
        'embeddings': []
    },
    cardinalities_path='cardinalities.json',
    shuffle=True,
    target={'categorical': ['channel']} # New config format
)

# Get a batch
batch_sup = next(iter(loader_sup))

print("Supervised Batch Keys:", batch_sup.keys())
print("Supervised Targets:", batch_sup['targets'])

# Verification
# Input categorical should be unchanged (both present, as we are explicit now)
assert batch_sup['categorical'].shape[1] == 2, f"Expected 2 categorical features (inputs), got {batch_sup['categorical'].shape[1]}"
print("Verified: Input categorical has all input columns.")

# Target should be present and valid
assert 'categorical' in batch_sup['targets'], "Targets should contain categorical key"
assert batch_sup['targets']['categorical'].shape[1] == 1, "Target categorical should have 1 column"
print("Verified: Targets are presen and correct.")

print("Supervised Mode Test Passed!")
