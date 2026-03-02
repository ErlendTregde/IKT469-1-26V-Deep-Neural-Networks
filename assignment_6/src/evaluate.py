import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from .data import corrupt


def generate_samples(model: nn.Module, loader: DataLoader, device: torch.device, noise_type: str, output_dir: Path, noise_std: float = 0.5, num_samples: int = 10):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)

    noisy = corrupt(images, labels, noise_type, noise_std).to(device)

    with torch.no_grad():
        reconstructed = model(noisy)

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 1.5, 5))
    titles = ["Original", "Noisy", "Reconstructed"]

    for col in range(num_samples):
        for row, tensor in enumerate([images, noisy, reconstructed]):
            axes[row, col].imshow(tensor[col].cpu().squeeze(), cmap="gray")
            axes[row, col].axis("off")

    for row, title in enumerate(titles):
        axes[row, 0].set_ylabel(title, fontsize=12)

    plt.tight_layout()
    save_path = output_dir / f"samples_{noise_type}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved samples to {save_path}")
