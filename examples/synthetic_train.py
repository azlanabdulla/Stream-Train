import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Add the parent directory to sys.path so we can import streamtrain without installing it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamtrain import Trainer, StreamTrainConfig

class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int = 1000, input_dim: int = 10):
        self.num_samples = num_samples
        self.input_dim = input_dim
        # Generate some random data
        self.data = torch.randn(num_samples, input_dim)
        # Binary classification target
        self.targets = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="examples/config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Load configuration
    config = StreamTrainConfig.from_yaml(args.config)
    
    # Initialize mock model, optimizer, dataset
    model = SimpleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = SyntheticDataset(num_samples=1000)
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
        device=device
    )
    
    # Resume if requested
    if args.resume:
        trainer.resume()
        
    try:
        # Train
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining paused by user.")
        trainer.stop()

if __name__ == "__main__":
    main()
