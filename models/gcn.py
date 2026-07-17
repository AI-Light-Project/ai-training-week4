# -*- coding: utf-8 -*-
"""GCN：图卷积网络（同构图谱卷积模型）。

基于谱图理论的消息传递，通过归一化拉普拉斯算子聚合邻居信息。
是 GNN 的经典基线，适合传递式（transductive）场景。

forward(data) 接收 PyG Data，输出 [N, 2] logits。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data, return_embedding: bool = False):
        x, edge_index = data.x, data.edge_index
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
