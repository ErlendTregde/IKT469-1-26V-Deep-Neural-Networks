import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from .data import get_data, get_cifar10_data
from .models.cnn import CNNModel
from .models.inception import InceptionNet
from .models.squeezenet import SqueezeNet
from .models.resnet import ResNet
from .models.custom import custom_model
from .models.mixtureOfExpert import MixtureOfExperts
import torch.optim as optim
import argparse
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
import os

def train_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Calculate precision, recall, and f1-score (macro average for multi-class)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0) #type: ignore
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0) #type: ignore
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)#type: ignore


    return epoch_loss, epoch_acc, precision, recall, f1

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)


def train_model(model, device, train_loader, test_loader, criterion, optimizer, num_epochs, save_path, writer=None, model_name="", lr=0.0, criterion_name=""):
    best_acc = 0.0
    best_metrics = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        test_loss, test_acc, precision, recall, f1 = evaluate(model, device, test_loader, criterion)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Log to TensorBoard
        if writer:
            writer.add_scalars('Loss', {'train': train_loss, 'test': test_loss}, epoch)
            writer.add_scalars('Accuracy', {'train': train_acc, 'test': test_acc}, epoch)
            writer.add_scalar('Precision', precision, epoch)
            writer.add_scalar('Recall', recall, epoch)
            writer.add_scalar('F1-Score', f1, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            best_metrics = {
                'epoch': epoch + 1,
                'model': model_name,
                'error_function': criterion_name,
                'learning_rate': lr,
                'accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            save_model(model, save_path)
            print(f"Model saved with accuracy: {best_acc:.4f}")

    # Log best model metrics to CSV
    if best_metrics:
        log_best_model_metrics(best_metrics)
        print(f"\n{'='*60}")
        print(f"Best Model Results (Epoch {best_metrics['epoch']}):")
        print(f"  Model: {best_metrics['model']}")
        print(f"  Error Function: {best_metrics['error_function']}")
        print(f"  Learning Rate: {best_metrics['learning_rate']}")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"  F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"{'='*60}\n")

    return best_metrics


def log_best_model_metrics(metrics):
    """Log best model metrics to a CSV file."""
    csv_file = "model_results.csv"
    file_exists = os.path.isfile(csv_file)

    fieldnames = ['Model', 'Error Function', 'Learning Rate', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Best Epoch', 'Timestamp']

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Model': metrics['model'],
            'Error Function': metrics['error_function'],
            'Learning Rate': metrics['learning_rate'],
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'Best Epoch': metrics['epoch'],
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train models")

    args.add_argument("--model", type=str, choices=["cnn", "inception", "squeezenet", "custom", "resnet", "mixtureOfExpert"], default="cnn", help="Model to train")
    args.add_argument("--dataset", type=str, choices=["fashion-mnist", "cifar10"], default="fashion-mnist", help="Dataset to use")
    args.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = args.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "fashion-mnist":
        train_loader, test_loader = get_data()
        input_channels = 1
    else:
        train_loader, test_loader = get_cifar10_data()
        input_channels = 3

    model = CNNModel(input_channels=input_channels)
    if args.model == "cnn":
        model = CNNModel(input_channels=input_channels)
    elif args.model == "inception":
        model = InceptionNet(input_channels=input_channels)
    elif args.model == "squeezenet":
        model = SqueezeNet(input_channels=input_channels)
    elif args.model == "custom":
        model = custom_model(input_channels=input_channels)
    elif args.model == "resnet":
        model = ResNet(input_channels=input_channels)
    elif args.model == "mixtureOfExpert":
        model = MixtureOfExperts(freeze_experts=True)
        model.load_expert_weights(0, "weights/best_resnet_model.pth", device)
        model.load_expert_weights(1, "weights/best_inception_model.pth", device)
        model.load_expert_weights(2, "weights/best_squeezenet_model.pth", device)
        model.load_expert_weights(3, "weights/best_custom_model.pth", device)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create TensorBoard writer with timestamped run name
    run_name = f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    print(f"training {args.model} on {args.dataset} for {args.epochs} epochs, with a learning rate of {args.lr}")
    print(f"TensorBoard logs: runs/{run_name}")

    criterion_name = criterion.__class__.__name__

    train_model(
        model,
        device,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=args.epochs,
        save_path=f"weights/best_{args.model}_{args.dataset}_model.pth",
        writer=writer,
        model_name=args.model,
        lr=args.lr,
        criterion_name=criterion_name
    )

    if args.model == "mixtureOfExpert":
        model.analyze_expert_usage(test_loader, device) #type: ignore

    writer.close()
