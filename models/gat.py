# -*- coding: utf-8 -*-
"""GAT：基于注意力机制的图神经网络。

通过注意力权重自适应聚合邻居信息，能突出重要邻居的贡献。
多头注意力（heads）增强表达力。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, heads: int = 2):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            # 后续层输入维度 = hidden_dim * heads
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.classifier = nn.Linear(hidden_dim * heads, 2)

    def forward(self, data, return_embedding: bool = False):
        x, edge_index = data.x, data.edge_index
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
