# -*- coding: utf-8 -*-
"""MLP Baseline：只使用节点自身特征，不使用图结构。

作用：回答"仅凭节点特征能做到什么程度"，作为 non-graph 底座。
后续图模型应与它对比，判断 GNN 是否带来真实增益。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, data, return_embedding: bool = False):
        # data 可能是 PyG Data 或 (x, edge_index)；MLP 只用 x
        x = data.x if hasattr(data, 'x') else data[0]
        h = self.encode(x)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
