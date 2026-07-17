# -*- coding: utf-8 -*-
"""TCN：时间卷积网络（Temporal Convolutional Network）。

在时间维度上做因果膨胀卷积（causal dilated convolution），
捕捉节点特征在时间快照序列上的时序模式。

与 EvolveGCN 的区别：
- EvolveGCN 通过 GRU 演化图卷积权重，侧重图结构演化；
- TCN 通过时间卷积直接处理快照特征序列，侧重时序模式。

forward(data) 接收包含时间快照序列的特殊 Data 对象。
节点特征序列由 GCN 编码每个快照得到，再输入 TCN 做时序建模。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TemporalBlock(nn.Module):
    """单层因果膨胀卷积块（含残差连接）。"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = (kernel_size - 1) * dilation  # 因果卷积：左 padding

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.chomp = padding  # 截断右侧多余的 padding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        out = self.conv(x)
        if self.chomp > 0:
            out = out[..., :-self.chomp]  # 因果：截断右侧
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNModel(nn.Module):
    """TCN 时间感知模型：GCN 编码各快照 + TCN 做时序卷积。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, num_snapshots: int = 5,
                 kernel_size: int = 3):
        super().__init__()
        self.dropout = dropout
        self.num_snapshots = num_snapshots
        self.hidden_dim = hidden_dim

        # GCN 编码器：对每个时间快照做图卷积得到节点表示
        self.gcn_encoder = GCNConv(in_dim, hidden_dim)

        # TCN 层：在时间维度上做因果膨胀卷积
        tcn_layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # 指数膨胀：1, 2, 4, ...
            tcn_layers.append(TemporalBlock(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size, dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*tcn_layers)

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data, return_embedding: bool = False):
        snapshots = getattr(data, 'snapshots', None)
        if snapshots is None:
            snapshots = [data]

        T = len(snapshots)
        N = snapshots[0].x.size(0)
        edge_index = snapshots[0].edge_index  # 使用最后一个快照的图结构

        # 1) GCN 编码每个快照：得到 (N, hidden_dim) × T
        h_list = []
        for snap in snapshots:
            h_t = self.gcn_encoder(snap.x, snap.edge_index)
            h_t = F.relu(h_t)
            h_t = F.dropout(h_t, p=self.dropout, training=self.training)
            h_list.append(h_t)

        # 2) 堆叠为时间序列：(N, hidden_dim, T)
        h_seq = torch.stack(h_list, dim=2)  # (N, hidden_dim, T)

        # 3) TCN 时序卷积
        h_out = self.tcn(h_seq)  # (N, hidden_dim, T)

        # 4) 取最后一个时间步的输出（因果卷积，只看过去）
        h_final = h_out[..., -1]  # (N, hidden_dim)

        logits = self.classifier(h_final)
        if return_embedding:
            return logits, h_final
        return logits
