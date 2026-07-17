# -*- coding: utf-8 -*-
"""HeteroSAGE：异构图 SAGE 模型。

基于 PyG HeteroConv，按 edge_type 分通道做 SAGEConv 聚合，
再拼接/求和融合多种关系的语义信息。

与同构 GraphSAGE 的区别：
- 不同 edge_type 使用独立的 SAGEConv 参数，学习不同关系的语义；
- HeteroConv 自动管理多种边类型的消息传递。

forward(data) 接收 PyG HeteroData（由 s1.to_heterogeneous_data 转换）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroSAGEModel(nn.Module):
    """异构图 SAGE：按 edge_type 分通道聚合，多关系融合。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, edge_types: list = None):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # edge_types: [("user","type_0","user"), ...] 由 data.metadata() 获取
        if edge_types is None:
            edge_types = [('user', 'edge', 'user')]  # fallback

        self.layers = nn.ModuleList()
        # 第一层：HeteroConv 包装每种关系的 SAGEConv
        conv_dict_1 = {}
        for et in edge_types:
            conv_dict_1[et] = SAGEConv((in_dim, in_dim), hidden_dim)
        self.layers.append(HeteroConv(conv_dict_1, aggr='sum'))

        # 后续层
        for _ in range(num_layers - 1):
            conv_dict = {}
            for et in edge_types:
                conv_dict[et] = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
            self.layers.append(HeteroConv(conv_dict, aggr='sum'))

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data, return_embedding: bool = False):
        # HeteroData: x_dict = {"user": tensor}, edge_index_dict = {rel: edge_index}
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        h_dict = x_dict
        for i, layer in enumerate(self.layers):
            h_dict = layer(h_dict, edge_index_dict)
            for key in h_dict:
                h_dict[key] = F.relu(h_dict[key])
                h_dict[key] = F.dropout(h_dict[key], p=self.dropout, training=self.training)

        # 取 "user" 节点的最终表示
        h = h_dict['user']
        logits = self.classifier(h)
        if return_embedding:
            return logits, h
        return logits
