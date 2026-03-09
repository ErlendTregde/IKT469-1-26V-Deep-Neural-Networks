import os
import kagglehub
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import TextGNN
from train import (
    train,
    train_with_pseudo_labels,
    train_with_mean_teacher,
    train_with_pseudo_and_mean_teacher,
    evaluate,
)


def _load_raw():
    dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    df = pd.read_csv(os.path.join(dataset_path, "styles.csv"), on_bad_lines="skip")
    df = df[["productDisplayName", "masterCategory"]].dropna()
    return df


def _build_vocab(texts):
    vocab = {}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def _build_graph(text, label_idx, word_to_idx):
    words = text.lower().split()
    x = torch.tensor([word_to_idx[w] for w in words], dtype=torch.long)

    edges = []
    for i in range(len(words) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=torch.tensor(label_idx, dtype=torch.long))


def run_experiments(label_fracs=(0.30, 0.50, 0.70, 0.01), epochs=50):
    # "2"     = supervised CE only
    # "2+3"   = supervised + pseudo-labels every 5 epochs
    # "2+4"   = supervised + mean-teacher consistency
    # "2+3+4" = supervised + pseudo-labels (from teacher) + mean-teacher consistency
    modes = ["2", "2+3", "2+4", "2+3+4"]

    df = _load_raw()

    unique_labels = sorted(df["masterCategory"].unique()) #type: ignore
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    vocab = _build_vocab(df["productDisplayName"])
    vocab_size = len(vocab)

    # build all graphs fully labeled (labels hidden from training via split, used for eval)
    all_graphs = [
        _build_graph(row["productDisplayName"], label_to_idx[row["masterCategory"]], vocab)
        for _, row in df.iterrows()
    ]

    # fixed 80/20 train-pool / test split
    n = len(all_graphs)
    perm = torch.randperm(n).tolist()
    test_size = int(0.2 * n)
    test_graphs = [all_graphs[i] for i in perm[:test_size]]
    train_pool  = [all_graphs[i] for i in perm[test_size:]]
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    results = {}

    for frac in label_fracs:
        n_train   = len(train_pool)
        n_labeled = max(1, int(frac * n_train))

        sub_perm         = torch.randperm(n_train).tolist()
        labeled_graphs   = [train_pool[i] for i in sub_perm[:n_labeled]]
        unlabeled_graphs = [train_pool[i] for i in sub_perm[n_labeled:]]

        labeled_loader   = DataLoader(labeled_graphs,   batch_size=32, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_graphs, batch_size=32, shuffle=True)

        for mode in modes:
            print(f"\n{'='*60}")
            print(f"Label fraction: {frac:.0%}  |  Mode: {mode}")
            print(f"{'='*60}")

            gnn = TextGNN(
                vocab_size=vocab_size,
                embed_dim=64,
                hidden_dim=128,
                num_classes=num_classes,
            )

            match mode:
                case "2":
                    train(gnn, labeled_loader, epochs=epochs)

                case "2+3":
                    train_with_pseudo_labels(
                        gnn, labeled_graphs, labeled_loader, unlabeled_loader, epochs=epochs
                    )

                case "2+4":
                    train_with_mean_teacher(
                        gnn, labeled_loader, unlabeled_loader, epochs=epochs
                    )

                case "2+3+4":
                    train_with_pseudo_and_mean_teacher(
                        gnn, labeled_graphs, labeled_loader, unlabeled_loader, epochs=epochs
                    )

            acc = evaluate(gnn, test_loader)
            results[(frac, mode)] = acc
            print(f"  => Test accuracy: {acc:.4f}")

    # print comparison table
    col_w = 12
    print("\n" + "=" * 60)
    print("RESULTS — Test Accuracy")
    print("=" * 60)
    header = f"{'Frac':<8}" + "".join(f"{m:<{col_w}}" for m in modes)
    print(header)
    print("-" * len(header))
    for frac in label_fracs:
        row = f"{frac:<8.0%}" + "".join(f"{results[(frac, m)]:<{col_w}.4f}" for m in modes)
        print(row)


if __name__ == "__main__":
    run_experiments()
