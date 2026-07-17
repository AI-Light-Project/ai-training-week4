# -*- coding: utf-8 -*-
"""GraphSAGE：邻居采样聚合的图神经网络。

通过聚合邻居特征学习节点表示，适合归纳式（inductive）场景。
forward(data) 接收 PyG Data，输出 [N, 2] logits。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data, return_embedding: bool = False):
        x, edge_index = data.x, data.edge_index
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
