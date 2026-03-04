import torch
import torch.nn as nn

from a2 import MessagePassingLayer


class GNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gnn = MessagePassingLayer(embed_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x:          [num_nodes]       long  – vocab indices
        edge_index: [2, num_edges]    long
        returns:    [1, num_classes]  float – class logits
        """
        h = self.embedding(x)                   # [num_nodes, embed_dim]
        h = self.gnn(h, edge_index)             # [num_nodes, hidden_dim]

        # A3: global mean pooling → single graph-level vector
        h_graph = h.mean(dim=0, keepdim=True)   # [1, hidden_dim]

        return self.classifier(h_graph)          # [1, num_classes]
