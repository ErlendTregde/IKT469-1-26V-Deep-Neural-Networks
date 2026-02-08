import kagglehub as hub
import pandas as pd
from torch.utils.data import DataLoader, Dataset

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
