#type: ignore
import torch
import torch.nn as nn
from torch.utils.data import random_split

from a1 import get_dataset
from a3 import GNNClassifier


def _decode(graph, idx_to_word, idx_to_label):
    text = " ".join(idx_to_word[i.item()] for i in graph["x"])
    label = idx_to_label[graph["y"].item()]
    return text, label


def train():
    dataset = get_dataset()
    vocab_size = len(dataset.word_to_idx)
    num_classes = len(dataset.label_to_idx)
    idx_to_word = {v: k for k, v in dataset.word_to_idx.items()}

    print("=" * 52)
    print(f"  Dataset : {len(dataset):,} product graphs")
    print(f"  Vocab   : {vocab_size:,} unique words")
    print(f"  Classes : {num_classes}  {list(dataset.label_to_idx.keys())}")
    print("=" * 52)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    model = GNNClassifier(vocab_size, embed_dim=32, hidden_dim=64, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n{'Epoch':<8} {'Loss':<10} {'Train Acc':<12} {'Val Acc'}")
    print("-" * 44)

    for epoch in range(5):
        # --- train ---
        model.train()
        total_loss, correct = 0.0, 0
        for graph in train_set:
            x = graph["x"]
            edge_index = graph["edge_index"]
            y = graph["y"].unsqueeze(0)

            out = model(x, edge_index)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()

        # --- validate ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for graph in val_set:
                x = graph["x"]
                edge_index = graph["edge_index"]
                y = graph["y"].unsqueeze(0)
                out = model(x, edge_index)
                val_correct += (out.argmax(dim=1) == y).sum().item()

        print(
            f"{epoch + 1}/5     "
            f"{total_loss / len(train_set):<10.4f}"
            f"{correct / len(train_set):<12.3f}"
            f"{val_correct / len(val_set):.3f}"
        )

    # --- example predictions ---
    print("\n" + "=" * 52)
    print("  Example predictions (from validation set)")
    print("=" * 52)
    model.eval()
    with torch.no_grad():
        for graph in list(val_set)[:5]:
            x = graph["x"]
            edge_index = graph["edge_index"]
            y = graph["y"].unsqueeze(0)
            out = model(x, edge_index)
            pred = out.argmax(dim=1).item()

            text, true_label = _decode(graph, idx_to_word, dataset.idx_to_label)
            pred_label = dataset.idx_to_label[pred]
            mark = "✓" if pred == y.item() else "✗"

            print(f"  {mark} Product  : {text}")
            print(f"    True     : {true_label}")
            print(f"    Predicted: {pred_label}")
            print()


if __name__ == "__main__":
    train()
