import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x:          [num_nodes, in_dim]  float node features
        edge_index: [2, num_edges]       long, each column is (src, dst)
        returns:    [num_nodes, out_dim]
        """
        src, dst = edge_index[0], edge_index[1]   # both shape [num_edges]

        # 1. MESSAGE: each source node sends its features to its neighbour
        messages = x[src]  # [num_edges, in_dim]

        # 2. AGGREGATE: sum all incoming messages at each destination node
        agg = torch.zeros(x.size(0), x.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        # agg[v] = sum of features of all neighbours of v

        # 3. UPDATE: linear transform + non-linearity
        out = torch.relu(self.linear(agg))  # [num_nodes, out_dim]

        return out
