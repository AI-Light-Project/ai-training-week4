# -*- coding: utf-8 -*-
"""DGraphFin 大作业通用工具。

包含：随机种子、mask 转换、评估指标、持久化 IO、matplotlib 配置。
各阶段脚本共享这些函数，避免重复实现。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------- matplotlib 配置（必须在导入 pyplot 前设置）----------
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合批量保存图片
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

import config  # noqa: E402


# ============================================================
# 随机种子
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError:
            # GPU 显存不足时跳过 CUDA 种子设置
            pass


# ============================================================
# Mask / 数据标准化（复用自 baseline）
# ============================================================
def index_to_mask(index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[index.long()] = True
    return mask


def ensure_bool_masks(data: Data) -> Data:
    num_nodes = data.num_nodes
    for name in ['train_mask', 'val_mask', 'test_mask']:
        if not hasattr(data, name):
            raise ValueError(f'Data 对象缺少 `{name}`。')
        value = getattr(data, name)
        if value.dtype != torch.bool:
            setattr(data, name, index_to_mask(value, num_nodes))
    return data


def keep_binary_label_masks(data: Data) -> Data:
    """DGraphFin 原始标签含 0/1/2/3，只在 y=0 / y=1 节点上训练与评估。

    标签 2/3 是背景节点，不参与 loss，但保留在图结构中作为消息传递桥梁。
    绝不能删除背景节点，否则会破坏图连通性和 GNN 感受野。
    """
    binary_mask = (data.y == 0) | (data.y == 1)
    for name in ['train_mask', 'val_mask', 'test_mask']:
        setattr(data, name, getattr(data, name) & binary_mask)
    return data


def normalize_data(data: Data) -> Data:
    """统一数据类型并生成布尔 mask。"""
    if data.x is None:
        raise ValueError('Data 必须包含节点特征 `x`。')
    if data.edge_index is None:
        raise ValueError('Data 必须包含图结构 `edge_index`。')
    if data.y is None:
        raise ValueError('Data 必须包含标签 `y`。')

    data.x = data.x.float()
    data.y = data.y.view(-1).long()
    data.edge_index = data.edge_index.long()
    data = ensure_bool_masks(data)
    data = keep_binary_label_masks(data)
    return data


# ============================================================
# 评估指标（AUC / AP / Recall@K / Precision@K）
# ============================================================
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    ks: Iterable[int] = config.TOPK_LIST) -> Dict[str, float]:
    """计算风控场景核心指标。

    - ROC-AUC：整体排序能力；
    - Average Precision：类别不平衡下更有参考价值；
    - Recall@K / Precision@K：模拟每天只能审核 Top-K 个高风险用户。
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    metrics: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_score))
        metrics['average_precision'] = float(average_precision_score(y_true, y_score))
    else:
        metrics['roc_auc'] = float('nan')
        metrics['average_precision'] = float('nan')

    order = np.argsort(-y_score)
    positives = max(1, int(y_true.sum()))
    for k in ks:
        k_eff = min(k, len(order))
        topk = order[:k_eff]
        metrics[f'recall@{k}'] = float(y_true[topk].sum() / positives)
        metrics[f'precision@{k}'] = float(y_true[topk].mean()) if k_eff > 0 else 0.0
    return metrics


# ============================================================
# 持久化 IO
# ============================================================
def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _json_default(o):
    """JSON 序列化兜底：处理 numpy / torch 标量。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_features(x: torch.Tensor, meta: dict, data_source: str, feature_name: str) -> None:
    """保存特征张量与元信息（特征名、维度构成等）。"""
    torch.save(x.detach().cpu(), config.feature_path(data_source, feature_name))
    save_json(meta, config.feature_meta_path(data_source, feature_name))


def load_features(data_source: str, feature_name: str) -> torch.Tensor:
    return torch.load(config.feature_path(data_source, feature_name), map_location='cpu',
                      weights_only=False)


def save_model(state_dict: dict, data_source: str, model_name: str, feature_name: str) -> None:
    torch.save(state_dict, config.model_path(data_source, model_name, feature_name))


def save_predictions(node_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray,
                     data_source: str, model_name: str, feature_name: str) -> None:
    """保存测试集预测结果，供后续网页做风险分数分布、Top-K 清单等可视化。"""
    path = config.predictions_path(data_source, model_name, feature_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        node_ids=node_ids.astype(np.int64),
        y_true=y_true.astype(np.int64),
        y_score=y_score.astype(np.float32),
    )


def save_history(history: pd.DataFrame, data_source: str, model_name: str, feature_name: str) -> None:
    """保存每轮训练 loss / 指标，用于绘制训练曲线。"""
    save_json(history.to_dict(orient='records'),
              config.history_path(data_source, model_name, feature_name))


# ============================================================
# 预测接口
# ============================================================
def _get_node_attr(data, attr: str):
    """兼容 Data 和 HeteroData 的节点属性获取。"""
    if hasattr(data, attr):
        return getattr(data, attr)
    # HeteroData: data['user'].y, data['user'].test_mask 等
    if hasattr(data, 'node_types'):
        for nt in data.node_types:
            node_store = data[nt]
            if hasattr(node_store, attr):
                return getattr(node_store, attr)
    return None


@torch.no_grad()
def predict_on_mask(model, data: Data, mask: torch.Tensor, device: torch.device,
                    return_embedding: bool = False):
    """在指定 mask 上预测，返回 (node_ids, y_true, y_score[, embedding])。

    统一处理：
    - nn.Module（MLP/GCN/GraphSAGE/GAT/HeteroSAGE/RGCN/EvolveGCN/TCN）
    - XGBoost / LightGBM 包装器
    - 同构 Data 和 HeteroData
    """
    if hasattr(model, 'eval'):
        model.eval()
    emb = None

    # 获取节点 y 和 mask（兼容 HeteroData）
    y = _get_node_attr(data, 'y')
    if y is None:
        raise ValueError('无法从 data 获取节点标签 y')
    y = y.detach().cpu()  # 统一在 CPU 上做索引

    if hasattr(model, 'predict_proba_xgb'):  # XGBoost
        data = data.cpu()
        mask = mask.cpu()
        x = _get_node_attr(data, 'x')
        score = model.predict_proba_xgb(x)
        if not isinstance(score, torch.Tensor):
            score = torch.from_numpy(np.asarray(score))
    elif hasattr(model, 'predict_proba_lgb'):  # LightGBM
        data = data.cpu()
        mask = mask.cpu()
        x = _get_node_attr(data, 'x')
        score = model.predict_proba_lgb(x)
        if not isinstance(score, torch.Tensor):
            score = torch.from_numpy(np.asarray(score))
    else:
        # nn.Module：把模型与数据对齐到同一设备
        # 大图优化：GPU OOM 时自动回退到 CPU
        try:
            model = model.to(device)
            data_gpu = data.to(device)
            mask_gpu = mask.to(device)
            try:
                logits = model(data_gpu)
            except TypeError:
                x = _get_node_attr(data_gpu, 'x')
                ei = _get_node_attr(data_gpu, 'edge_index')
                logits = model(x, ei)
            score = logits.softmax(dim=-1)[:, 1]
            if return_embedding:
                try:
                    _, emb = model(data_gpu, return_embedding=True)
                except Exception:
                    emb = None
            # 评估完立即释放 GPU 显存
            del data_gpu, mask_gpu, logits
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and device.type == 'cuda':
                # GPU OOM，回退到 CPU
                print(f'  [OOM 回退] predict_on_mask 切换到 CPU')
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                model = model.cpu()
                data_cpu = data.cpu()
                mask_cpu_eval = mask.cpu()
                try:
                    logits = model(data_cpu)
                except TypeError:
                    x = _get_node_attr(data_cpu, 'x')
                    ei = _get_node_attr(data_cpu, 'edge_index')
                    logits = model(x, ei)
                score = logits.softmax(dim=-1)[:, 1]
                del data_cpu, mask_cpu_eval, logits
            else:
                raise

    mask_cpu = mask.detach().cpu()
    node_ids = mask_cpu.nonzero(as_tuple=False).view(-1)
    y_true = y[mask_cpu].detach().cpu().numpy()
    if isinstance(score, torch.Tensor):
        y_score_np = score[mask_cpu].detach().cpu().numpy()
    else:
        y_score_np = np.asarray(score)[mask_cpu.detach().cpu().numpy()]

    if return_embedding:
        return (node_ids.cpu().numpy(), y_true, y_score_np, emb)
    return (node_ids.cpu().numpy(), y_true, y_score_np)
