import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

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