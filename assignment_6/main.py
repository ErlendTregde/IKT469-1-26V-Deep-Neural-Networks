import torch
import argparse
from pathlib import Path

from src.data import get_dataloaders
from src.model import DenoisingAutoencoder
from src.train import train_epoch, evaluate_epoch
from src.evaluate import generate_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder — MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "label_stamp"])
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)

    model = DenoisingAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.noise_type, args.noise_std)
        val_loss = evaluate_epoch(model, test_loader, device, args.noise_type, args.noise_std)
        print(f"Epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    output_dir = Path(args.output_dir)
    generate_samples(model, test_loader, device, args.noise_type, output_dir, args.noise_std)

    weights_path = output_dir / f"model_{args.noise_type}.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model to {weights_path}")


if __name__ == "__main__":
    main()
