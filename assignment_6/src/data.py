import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def get_dataloaders(data_dir: str, batch_size: int = 128):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def add_gaussian_noise(images: torch.Tensor, std: float = 0.5) -> torch.Tensor:
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0.0, 1.0)


def add_label_stamp(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = images.clone()
    for i, (img_tensor, label) in enumerate(zip(images, labels)):
        pil_img = transforms.ToPILImage()(img_tensor)
        draw = ImageDraw.Draw(pil_img)
        draw.text((1, 1), str(label.item()), fill=255)
        result[i] = transforms.ToTensor()(pil_img)
    return result


def corrupt(images: torch.Tensor, labels: torch.Tensor, noise_type: str, noise_std: float = 0.5) -> torch.Tensor:
    if noise_type == "gaussian":
        return add_gaussian_noise(images, std=noise_std)
    if noise_type == "label_stamp":
        return add_label_stamp(images, labels)
    raise ValueError(f"Unknown noise type: {noise_type}")
