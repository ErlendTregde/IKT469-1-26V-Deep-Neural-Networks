from .preprocess import (
    prepare_data, 
    load_wine_data, 
    get_dataloaders,
    prepare_cifar10_data,
    load_cifar10_data,
    get_cifar10_dataloaders,
    get_cifar10_transforms
)

from .download_data import (
    download_wine_dataset,
    download_cifar10,
    TARGET_DIR,
    CIFAR10_DIR
)

# download_cifar10.py is now in this folder but is a standalone script

__all__ = [
    'prepare_data', 
    'load_wine_data', 
    'get_dataloaders',
    'prepare_cifar10_data',
    'load_cifar10_data',
    'get_cifar10_dataloaders',
    'get_cifar10_transforms',
    'download_wine_dataset',
    'download_cifar10',
    'TARGET_DIR',
    'CIFAR10_DIR'
]
