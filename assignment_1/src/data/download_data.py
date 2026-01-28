import kagglehub
import shutil
from pathlib import Path
import torchvision
import torchvision.transforms as transforms

# Wine dataset target directory
TARGET_DIR = Path("/home/coder/IKT469-1-26V-Deep-Neural-Networks/assignment_1/data/winedataset")

# CIFAR-10 target directory
CIFAR10_DIR = Path("/home/coder/IKT469-1-26V-Deep-Neural-Networks/assignment_1/data/cifar10")


def download_wine_dataset():
    """Download Wine Quality dataset from Kaggle"""
    print("Downloading Wine Quality dataset...")
    path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
    
    print(f"Dataset downloaded to: {path}")
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(path)
    for file in source_path.glob("*"):
        if file.is_file():
            target_file = TARGET_DIR / file.name
            shutil.copy2(file, target_file)
            print(f"Copied: {file.name}")
    
    print(f"\nDataset saved to: {TARGET_DIR}")
    print(f"Files in directory: {list(TARGET_DIR.glob('*'))}")
    return TARGET_DIR


def download_cifar10(download_dir=None):
    """
    Download CIFAR-10 dataset using torchvision
    
    Args:
        download_dir: Directory to download to (default: CIFAR10_DIR)
    
    Returns:
        Path to downloaded dataset
    """
    if download_dir is None:
        download_dir = CIFAR10_DIR
    
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DOWNLOADING CIFAR-10 DATASET")
    print("=" * 60)
    print(f"Download directory: {download_dir}")
    
    # Download training set
    print("\nDownloading training set...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(download_dir),
        train=True,
        download=True
    )
    
    # Download test set
    print("\nDownloading test set...")
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(download_dir),
        train=False,
        download=True
    )
    
    print("\n" + "=" * 60)
    print("CIFAR-10 DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Image shape: {train_dataset[0][0].size}")
    print("=" * 60)
    
    return download_dir


# Keep backward compatibility - run wine dataset download when script is run directly
if __name__ == "__main__":
    path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
    
    print(f"Dataset downloaded to: {path}")
    
    source_path = Path(path)
    for file in source_path.glob("*"):
        if file.is_file():
            target_file = TARGET_DIR / file.name
            shutil.copy2(file, target_file)
            print(f"Copied: {file.name}")
    
    print(f"\nDataset saved to: {TARGET_DIR}")
    print(f"Files in directory: {list(TARGET_DIR.glob('*'))}")
