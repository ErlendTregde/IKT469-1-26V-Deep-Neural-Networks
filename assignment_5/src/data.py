import kagglehub as hub
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

path = hub.dataset_download("zalando-research/fashionmnist")

class FashionMNISTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row['label']
        image = row.drop('label').values.astype('float32').reshape(1, 28, 28) / 255.0
        return image, label


# Download latest version
def load_csv(path):
    data = pd.read_csv(path)
    return data


def get_data():
    train_data = load_csv(f"{path}/fashion-mnist_train.csv")
    test_data = load_csv(f"{path}/fashion-mnist_test.csv")

    train_loader = DataLoader(
        FashionMNISTDataset(train_data),
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        FashionMNISTDataset(test_data),
        batch_size=64,
        shuffle=False,
    )

    return train_loader, test_loader


def get_cifar10_data(data_dir='./data/cifar10', batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
