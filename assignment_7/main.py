import torch
import argparse
from pathlib import Path

from src.data import get_dataloaders
from src.model import MambaClassifier
from src.train import train_epoch, evaluate_epoch
from src.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="SSM-only Mamba baseline — Fashion Product Images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--max_classes", type=int, default=20)
    parser.add_argument("--label_col", type=str, default="articleType")
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, num_classes, class_to_idx = get_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        label_col=args.label_col,
        max_classes=args.max_classes,
    )
    print(f"Classes ({num_classes}): {sorted(class_to_idx)}")

    model = MambaClassifier(
        num_classes=num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        num_layers=args.num_layers,
        expand=args.expand,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, device)
        scheduler.step()
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

    output_dir = Path(args.output_dir)
    class_names = sorted(class_to_idx, key=class_to_idx.get)
    evaluate(model, val_loader, device, output_dir, class_names=class_names)

    weights_path = output_dir / "model_mamba.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model to {weights_path}")


if __name__ == "__main__":
    main()
