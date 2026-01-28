"""
Script to download and test CIFAR-10 dataset
Run this to download CIFAR-10 for Part C of the assignment
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.download_data import download_cifar10, CIFAR10_DIR
from data.preprocess import prepare_cifar10_data

print("\n" + "=" * 60)
print("CIFAR-10 DOWNLOAD AND PREPROCESSING TEST")
print("=" * 60)

# Step 1: Download CIFAR-10
print("\nStep 1: Downloading CIFAR-10...")
cifar_dir = download_cifar10()

# Step 2: Test preprocessing
print("\nStep 2: Testing data preprocessing...")
data = prepare_cifar10_data(
    cifar_dir,
    batch_size=128,
    val_split=0.1,
    subset_size=1000,  # Use small subset for quick test
    augment=True
)

print("\n" + "=" * 60)
print("DATA LOADER INFORMATION")
print("=" * 60)
print(f"Train batches: {len(data['train_loader'])}")
print(f"Val batches: {len(data['val_loader'])}")
print(f"Test batches: {len(data['test_loader'])}")
print(f"Number of classes: {data['num_classes']}")
print(f"Classes: {data['classes']}")
print(f"Input shape: {data['input_shape']}")

# Step 3: Load a sample batch
print("\n" + "=" * 60)
print("LOADING SAMPLE BATCH")
print("=" * 60)

for images, labels in data['train_loader']:
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Image min/max: {images.min():.3f} / {images.max():.3f}")
    print(f"Sample labels: {labels[:10].tolist()}")
    break

print("\n" + "=" * 60)
print("✓ CIFAR-10 SETUP COMPLETE!")
print("=" * 60)
print(f"\nDataset location: {CIFAR10_DIR}")
print("\nYou can now use CIFAR-10 for Part C:")
print("  - Shallow CNN")
print("  - Deep CNN")
print("  - Residual CNN")
print("\nNext steps:")
print("  1. Implement CNN architectures in src/models/")
print("  2. Create a main script for Part C training")
print("=" * 60)
