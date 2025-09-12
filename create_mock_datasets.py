"""
create_mock_datasets.py

Create mock datasets for testing the PyTorch training pipeline.
This generates synthetic NPZ files with the expected structure.
"""

import numpy as np
import os
from pathlib import Path

def create_mock_dataset(name, num_classes, num_samples=1000, image_size=(224, 224, 3), metadata_dim=16):
    """Create a mock dataset with the expected structure."""
    
    # Generate random images (normalized to [0, 1])
    images = np.random.rand(num_samples, *image_size).astype(np.float32)
    
    # Generate random metadata (soil, weather, etc.)
    metadata = np.random.rand(num_samples, metadata_dim).astype(np.float32)
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, num_samples).astype(np.int64)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as NPZ file
    output_path = output_dir / f"{name}_dataset.npz"
    np.savez(output_path, images=images, metadata=metadata, labels=labels)
    
    print(f"Created {name} dataset: {output_path}")
    print(f"  - Images: {images.shape}")
    print(f"  - Metadata: {metadata.shape}")
    print(f"  - Labels: {labels.shape}")
    print(f"  - Classes: {num_classes}")
    
    return output_path

def main():
    """Create all mock datasets."""
    print("Creating mock datasets for Khet Guard training...")
    
    # Create datasets
    datasets = [
        ("plantvillage_color", 38),  # Disease detection
        ("pest", 20),                # Pest detection  
        ("cattle", 41)               # Cattle breed classification
    ]
    
    for name, num_classes in datasets:
        create_mock_dataset(name, num_classes, num_samples=1000)
    
    print("\nAll mock datasets created successfully!")
    print("You can now run: python train_pytorch_fusion.py")

if __name__ == "__main__":
    main()

