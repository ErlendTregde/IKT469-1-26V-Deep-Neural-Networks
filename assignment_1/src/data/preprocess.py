import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

class WineDataset(Dataset):    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_wine_data(csv_path, test_size=0.2, val_size=0.1, random_state=42):
    
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    
    y_min = y.min()
    y = y - y_min
    
    print(f"Classes: {np.unique(y)} (remapped from {y_min}-{y_min + len(np.unique(y)) - 1})")
    print(f"Class distribution: {np.bincount(y)}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=random_state, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'input_dim': X_train.shape[1],
        'num_classes': len(np.unique(y)),
        'scaler': scaler
    }


def get_dataloaders(data_dict, batch_size=32):
    train_dataset = WineDataset(*data_dict['train'])
    val_dataset = WineDataset(*data_dict['val'])
    test_dataset = WineDataset(*data_dict['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def prepare_data(data_path, batch_size=32):
    data_dict = load_wine_data(data_path)
    train_loader, val_loader, test_loader = get_dataloaders(data_dict, batch_size)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_dim': data_dict['input_dim'],
        'num_classes': data_dict['num_classes']
    }


########################################
#########   CIFAR-10 Dataset   #########
########################################

def get_cifar10_transforms(train=True, augment=True):
    """
    Get data transforms for CIFAR-10
    
    Args:
        train: Whether this is for training data
        augment: Whether to apply data augmentation (only for training)
    
    Returns:
        torchvision.transforms.Compose object
    """
    # CIFAR-10 normalization values (mean and std for each channel)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
    
    if train and augment:
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Random crop with padding
            transforms.RandomHorizontalFlip(),      # Random horizontal flip
            transforms.ToTensor(),                  # Convert to tensor
            normalize                               # Normalize
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    return transform


def load_cifar10_data(data_dir, val_split=0.1, subset_size=None, augment=True):
    """
    Load CIFAR-10 dataset with train/val/test splits
    
    Args:
        data_dir: Directory containing CIFAR-10 data
        val_split: Proportion of training data to use for validation
        subset_size: If provided, use only this many samples (for quick experiments)
        augment: Whether to apply data augmentation to training data
    
    Returns:
        Dictionary with train/val/test datasets and metadata
    """
    print("=" * 60)
    print("LOADING CIFAR-10 DATASET")
    print("=" * 60)
    
    # Get transforms
    train_transform = get_cifar10_transforms(train=True, augment=augment)
    test_transform = get_cifar10_transforms(train=False, augment=False)
    
    # Load full training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,  # Should already be downloaded
        transform=train_transform
    )
    
    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )
    
    # Split training data into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation set
    # Note: val_dataset still uses train_transform from parent, but without augmentation
    # we'd need to create a separate dataset for proper val transform
    
    # Use subset if specified (for quick experiments)
    if subset_size is not None:
        print(f"\nUsing subset of {subset_size} samples for quick experimentation")
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(subset_size // 10, len(val_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(subset_size // 5, len(test_dataset))))
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {full_train_dataset.classes}")
    print(f"Number of classes: {len(full_train_dataset.classes)}")
    print(f"Image shape: (3, 32, 32)")
    print("=" * 60)
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'num_classes': len(full_train_dataset.classes),
        'classes': full_train_dataset.classes,
        'input_shape': (3, 32, 32)
    }


def get_cifar10_dataloaders(data_dict, batch_size=128, num_workers=2):
    """
    Create PyTorch DataLoaders for CIFAR-10
    
    Args:
        data_dict: Dictionary from load_cifar10_data()
        batch_size: Batch size (larger for CNNs, typically 128)
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        data_dict['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        data_dict['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        data_dict['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def prepare_cifar10_data(data_dir, batch_size=128, val_split=0.1, subset_size=None, augment=True):
    """
    Complete CIFAR-10 data preparation pipeline
    
    Args:
        data_dir: Directory containing CIFAR-10 data
        batch_size: Batch size for DataLoaders
        val_split: Proportion for validation split
        subset_size: Optional subset size for quick experiments
        augment: Whether to apply data augmentation
    
    Returns:
        Dictionary with loaders and metadata
    """
    data_dict = load_cifar10_data(data_dir, val_split, subset_size, augment)
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(data_dict, batch_size)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_classes': data_dict['num_classes'],
        'classes': data_dict['classes'],
        'input_shape': data_dict['input_shape']
    }