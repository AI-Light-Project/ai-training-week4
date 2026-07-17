# -*- coding: utf-8 -*-
"""RGCN：关系图卷积网络。

针对多关系图设计，每种边类型使用独立的权重矩阵做消息传递，
并通过基分解（basis decomposition）降低参数量。

与 HeteroSAGE 的区别：
- RGCN 使用 RGCNConv，内置基分解参数共享，适合关系类型较多的场景；
- HeteroSAGE 使用独立 SAGEConv，每条关系参数完全独立。

forward(data) 接收同构 Data（需含 edge_type 属性）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNModel(nn.Module):
    """RGCN：关系图卷积，按 edge_type 分权重聚合。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, num_relations: int = 12,
                 num_bases: int = 4):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        # 第一层：in_dim -> hidden_dim
        self.layers.append(RGCNConv(in_dim, hidden_dim, num_relations,
                                    num_bases=num_bases))
        # 后续层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations,
                                        num_bases=num_bases))
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data, return_embedding: bool = False):
        x, edge_index = data.x, data.edge_index
        edge_type = getattr(data, 'edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long,
                                    device=x.device)

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_type)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
