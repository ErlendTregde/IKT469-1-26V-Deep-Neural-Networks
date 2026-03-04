import os
import kagglehub
import pandas as pd
import torch
from torch.utils.data import Dataset


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


def _build_graph(text, label, word_to_idx, label_to_idx):
    words = text.lower().split()

    # Nodes: each word as its vocab index
    x = torch.tensor([word_to_idx[w] for w in words], dtype=torch.long)  # [num_nodes]

    # Edges: consecutive words, both directions
    edges = []
    for i in range(len(words) - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, num_edges]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    y = torch.tensor(label_to_idx[label], dtype=torch.long)

    return {"x": x, "edge_index": edge_index, "y": y}


class FashionTextGraphDataset(Dataset):
    def __init__(self):
        df = _load_raw()
        self.word_to_idx = _build_vocab(df["productDisplayName"])
        self.label_to_idx = {label: i for i, label in enumerate(df["masterCategory"].unique())} # type: ignore
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        self.graphs = [
            _build_graph(row["productDisplayName"], row["masterCategory"], self.word_to_idx, self.label_to_idx)
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def get_dataset():
    return FashionTextGraphDataset()
