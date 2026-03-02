import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import corrupt


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, noise_type: str, noise_std: float = 0.5) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        noisy = corrupt(images, labels, noise_type, noise_std).to(device)

        reconstructed = model(noisy)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, noise_type: str, noise_std: float = 0.5) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            noisy = corrupt(images, labels, noise_type, noise_std)
            reconstructed = model(noisy)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()

    return total_loss / len(loader)
