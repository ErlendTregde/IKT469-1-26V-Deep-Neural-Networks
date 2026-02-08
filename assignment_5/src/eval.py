import torch
import torch.nn as nn
from data import get_data
from models.cnn import CNNModel
from models.inception import InceptionNet
from models.squeezenet import SqueezeNet
from models.resnet import ResNet
from models.custom import custom_model
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

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

    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0) #type: ignore
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0) #type: ignore
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0) #type: ignore
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return epoch_loss, epoch_acc, precision, recall, f1, conf_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, choices=["cnn", "inception", "squeezenet", "custom", "resnet"], 
                        required=True, help="Model to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint (e.g., best_cnn_model.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, test_loader = get_data()

    model = CNNModel()

    # Initialize model
    if args.model == "cnn":
        model = CNNModel()
    elif args.model == "inception":
        model = InceptionNet()
    elif args.model == "squeezenet":
        model = SqueezeNet()
    elif args.model == "custom":
        model = custom_model()
    elif args.model == "resnet":
        model = ResNet()

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Evaluate
    print(f"Evaluating {args.model} from {args.checkpoint}")
    test_loss, test_acc, precision, recall, f1, conf_matrix = evaluate(
        model, device, test_loader, criterion
    )

    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    print(f"{'='*60}\n")
