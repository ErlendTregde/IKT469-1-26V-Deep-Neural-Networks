import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    class_names: list[str] | None = None,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = (all_preds == all_labels).mean()
    print(f"Accuracy: {accuracy:.4f}")

    target_names = class_names if class_names else [str(i) for i in range(all_labels.max() + 1)]
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(all_labels, all_preds, target_names, output_dir)

    return {"accuracy": float(accuracy)}


def _save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: list[str], output_dir: Path):
    cm = confusion_matrix(labels, preds)
    fig_size = max(8, len(class_names) // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=len(class_names) <= 20, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    save_path = output_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")
