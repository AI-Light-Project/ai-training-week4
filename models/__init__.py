# -*- coding: utf-8 -*-
"""模型工厂：统一构建各模型实例。

每个模型独立一个文件，便于修改和扩展。新增模型只需：
1. 在 models/ 下新增 {name}.py 实现模型类；
2. 在 build_model 中增加一个分支。

模型分类：
- 非图模型：mlp / xgboost / lightgbm
- 同构图模型：gcn / graphsage / gat
- 异构图模型：heterosage / rgcn（需要 edge_type）
- 时间感知模型：evolvegcn / tcn（需要时间快照）
"""
from __future__ import annotations

import torch.nn as nn
from torch_geometric.data import Data

from .mlp import MLPModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .gcn import GCNModel
from .graphsage import GraphSAGEModel
from .gat import GATModel
from .hetero_sage import HeteroSAGEModel
from .rgcn import RGCNModel
from .evolve_gcn import EvolveGCNModel
from .tcn import TCNModel


def build_model(model_name: str, data: Data, hidden_dim: int = 64,
                num_layers: int = 2, dropout: float = 0.3, heads: int = 2,
                xgb_rounds: int = 400, lgb_rounds: int = 400,
                num_relations: int = 12) -> nn.Module:
    """根据名称构建模型。

    所有模型统一输出 [num_nodes, 2] 的 logits。
    支持模型类型：
    - 非图: mlp / xgboost / lightgbm
    - 同构图: gcn / graphsage / gat
    - 异构图: heterosage / rgcn（需要 data.edge_type 或 HeteroData）
    - 时间感知: evolvegcn / tcn（需要 data.snapshots）
    """
    name = model_name.lower()
    # 兼容 HeteroData：从节点存储中获取 x
    if hasattr(data, 'x'):
        in_dim = data.x.size(-1)
    elif hasattr(data, 'node_types'):
        # HeteroData: 取第一个节点类型的 x 维度
        for nt in data.node_types:
            if hasattr(data[nt], 'x') and data[nt].x is not None:
                in_dim = data[nt].x.size(-1)
                break
        else:
            raise ValueError('HeteroData 中未找到节点特征 x')
    else:
        raise ValueError('无法从 data 获取节点特征维度')

    # --- 非图模型 ---
    if name == 'mlp':
        return MLPModel(in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if name == 'xgboost':
        return XGBoostModel(n_estimators=xgb_rounds)
    if name == 'lightgbm':
        return LightGBMModel(n_estimators=lgb_rounds)

    # --- 同构图模型 ---
    if name == 'gcn':
        return GCNModel(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if name == 'graphsage':
        return GraphSAGEModel(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if name == 'gat':
        return GATModel(in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, heads=heads)

    # --- 异构图模型 ---
    if name == 'heterosage':
        # 需要 HeteroData：从 data 获取 edge_types
        if hasattr(data, 'edge_index_dict'):
            edge_types = list(data.edge_index_dict.keys())
        else:
            edge_types = [('user', 'edge', 'user')]
        return HeteroSAGEModel(in_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                               dropout=dropout, edge_types=edge_types)
    if name == 'rgcn':
        # 需要 edge_type：计算关系数
        if hasattr(data, 'edge_type') and data.edge_type is not None:
            num_rel = int(data.edge_type.max()) + 1
        else:
            num_rel = num_relations
        return RGCNModel(in_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                         dropout=dropout, num_relations=num_rel)

    # --- 时间感知模型 ---
    if name == 'evolvegcn':
        return EvolveGCNModel(in_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                              dropout=dropout, num_snapshots=getattr(data, 'num_snapshots', 5))
    if name == 'tcn':
        return TCNModel(in_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                        dropout=dropout, num_snapshots=getattr(data, 'num_snapshots', 5))

    raise ValueError(f'未知模型: {model_name}。当前支持: mlp / lightgbm / xgboost / '
                     f'gcn / graphsage / gat / heterosage / rgcn / evolvegcn / tcn')


__all__ = [
    'build_model',
    'MLPModel', 'XGBoostModel', 'LightGBMModel',
    'GCNModel', 'GraphSAGEModel', 'GATModel',
    'HeteroSAGEModel', 'RGCNModel',
    'EvolveGCNModel', 'TCNModel',
]
