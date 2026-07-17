# -*- coding: utf-8 -*-
"""阶段三：模型预测结果评估与融合。

职责：
1. 在测试集上评估每个模型，保存 metrics / predictions；
2. 多模型概率加权融合（权重 = 各模型验证集 AP 归一化）；
3. 汇总所有模型 × 特征的对比表 summary.csv。

融合思路（任务C 改进方向之一）：
- MLP 看节点自身特征，GNN 看邻居结构，二者互补；
- 用验证集 AP 衡量每个模型可靠性，归一化为权重，对预测概率加权平均。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import config
import utils
from models import build_model
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel


# ============================================================
# 测试集评估
# ============================================================
def plot_model_score_distribution(data_source: str, model_name: str,
                                  feature_name: str) -> str:
    """绘制单个模型在测试集上的风险分数分布（正常 vs 异常）。

    参考 baseline notebook cell 37 # 4 的可视化风格：
    对每个模型输出两张直方图叠加（Normal y=0 蓝色、Fraud y=1 红色），
    展示模型对正常/异常节点的区分能力。

    Args:
        data_source: 'sample' | 'full'
        model_name: 模型名（mlp/graphsage/gat 等）
        feature_name: 特征集名（raw/full 等）

    Returns:
        保存的图片路径
    """
    import matplotlib.pyplot as plt
    from utils import plt as _plt

    pred_path = config.predictions_path(data_source, model_name, feature_name)
    if not pred_path.exists():
        return ''
    npz = np.load(pred_path)
    y_true = npz['y_true'].astype(np.int32)
    y_score = npz['y_score'].astype(np.float64)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_score[y_true == 0], bins=40, alpha=0.65, label='Normal y=0',
            color='steelblue', density=True, edgecolor='white', linewidth=0.3)
    if (y_true == 1).any():
        ax.hist(y_score[y_true == 1], bins=40, alpha=0.65, label='Fraud y=1',
                color='crimson', density=True, edgecolor='white', linewidth=0.3)
    ax.set_title(f'{model_name.upper()} / {feature_name} Test Risk Score Distribution')
    ax.set_xlabel('Predicted Risk Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_dir = config.out_root(data_source) / 'results' / 'score_distributions'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{model_name}_{feature_name}_score_dist.png'
    plt.savefig(out_path, dpi=120)
    plt.close()
    return str(out_path)


def evaluate_on_test(model, data, model_name: str, feature_name: str,
                     data_source: str, device: torch.device = None) -> dict:
    """在测试集评估，持久化 metrics 与 predictions。"""
    if device is None:
        device = torch.device('cpu')
    test_mask = utils._get_node_attr(data, 'test_mask')
    node_ids, y_true, y_score = utils.predict_on_mask(model, data, test_mask, device)
    metrics = utils.compute_metrics(y_true, y_score, ks=config.TOPK_LIST)

    # 从训练历史取验证集 AP，作为融合权重依据
    try:
        hist = utils.load_json(config.history_path(data_source, model_name, feature_name))
        val_aps = [r.get('val_average_precision', r.get('val_ap', 0.0)) for r in hist
                   if r.get('val_average_precision', r.get('val_ap')) is not None
                   and not np.isnan(r.get('val_average_precision', r.get('val_ap', 0.0)))]
        val_ap = float(max(val_aps)) if val_aps else 0.0
    except Exception:
        val_ap = 0.0

    metrics['model'] = model_name
    metrics['feature'] = feature_name
    metrics['data_source'] = data_source
    metrics['num_test_nodes'] = int(len(y_true))
    metrics['val_average_precision'] = val_ap

    utils.save_json(metrics, config.metrics_path(data_source, model_name, feature_name))
    utils.save_predictions(node_ids, y_true, y_score, data_source, model_name, feature_name)

    print(f'  [{model_name}/{feature_name}] 测试集 '
          f'AUC={metrics["roc_auc"]:.4f} AP={metrics["average_precision"]:.4f} '
          f'Recall@100={metrics["recall@100"]:.4f} Prec@100={metrics["precision@100"]:.4f} '
          f'| 预测已保存')
    return metrics


# ============================================================
# 多种融合策略
# ============================================================
def _fuse_ap_weighted(scores: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """AP 加权融合：权重 = 验证集 AP 归一化。"""
    fused = np.zeros_like(scores[0])
    for w, s in zip(weights, scores):
        fused += w * s
    return fused


def _fuse_mean(scores: List[np.ndarray]) -> np.ndarray:
    """等权算术平均：所有模型同等重要。"""
    return np.mean(np.vstack(scores), axis=0)


def _fuse_max(scores: List[np.ndarray]) -> np.ndarray:
    """最大值融合：取所有模型预测的最大值（风控保守策略，宁误报不漏报）。"""
    return np.max(np.vstack(scores), axis=0)


def _fuse_rank(scores: List[np.ndarray]) -> np.ndarray:
    """倒数排名融合 RRF：基于预测排名聚合，消除概率尺度差异。

    每个节点在每个模型中的排名 r（1=最高风险），融合分 = sum_m 1/(k+r_m)。
    返回归一化到 [0,1] 的风险分数（越高越异常）。
    """
    k = 60  # RRF 常数，平衡头部与尾部
    n = len(scores[0])
    rrf = np.zeros(n, dtype=np.float64)
    for s in scores:
        # 降序排名：风险最高(r=1) → 风险最低(r=n)
        ranks = np.argsort(np.argsort(-s)) + 1
        rrf += 1.0 / (k + ranks)
    # 归一化到 [0,1]（min-max，保持单调性）
    rrf_min, rrf_max = rrf.min(), rrf.max()
    if rrf_max > rrf_min:
        rrf = (rrf - rrf_min) / (rrf_max - rrf_min)
    return rrf


def _fuse_geomean(scores: List[np.ndarray]) -> np.ndarray:
    """几何平均：对低概率敏感，要求所有模型一致认为高风险才得高分。

    fused = exp(mean(log(score)))，等价于 prod(score)^(1/M)。
    对 score 做 epsilon 截断避免 log(0)。
    """
    eps = 1e-8
    log_scores = [np.log(np.clip(s, eps, 1.0)) for s in scores]
    fused = np.exp(np.mean(np.vstack(log_scores), axis=0))
    return fused


# method -> 融合函数
_FUSE_DISPATCH = {
    'ap_weighted': _fuse_ap_weighted,
    'mean': _fuse_mean,
    'max': _fuse_max,
    'rank': _fuse_rank,
    'geomean': _fuse_geomean,
}


# ============================================================
# 模型融合（多方法支持）
# ============================================================
def fuse_models(data_source: str, combo_name: str, model_names: List[str],
                feature_name: str, method: str = 'ap_weighted') -> dict:
    """对多个模型的测试集预测概率做融合。

    支持 5 种融合策略（见 config.FUSION_METHODS）：
    - ap_weighted: 验证集 AP 归一化加权（默认）
    - mean: 等权算术平均
    - max: 最大值（保守高召回）
    - rank: 倒数排名融合 RRF（尺度无关）
    - geomean: 几何平均（低概率敏感）

    权重 alpha_m = val_ap_m / sum_j(val_ap_j)（仅 ap_weighted 使用）。
    """
    if method not in _FUSE_DISPATCH:
        print(f'  [融合 {combo_name}] 未知方法 {method}，跳过')
        return {}

    scores = []
    weights = []
    y_true_ref = None
    node_ids_ref = None
    for mn in model_names:
        pred_path = config.predictions_path(data_source, mn, feature_name)
        if not pred_path.exists():
            print(f'  [融合 {combo_name}/{method}] 跳过：缺少 {mn} 预测文件')
            return {}
        npz = np.load(pred_path)
        scores.append(npz['y_score'].astype(np.float64))
        if y_true_ref is None:
            y_true_ref = npz['y_true']
            node_ids_ref = npz['node_ids']

        m = utils.load_json(config.metrics_path(data_source, mn, feature_name))
        weights.append(max(float(m.get('val_average_precision', 0.0)), 1e-6))

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    # 根据方法计算融合概率
    fuse_fn = _FUSE_DISPATCH[method]
    if method == 'ap_weighted':
        fused = fuse_fn(scores, weights)
    else:
        fused = fuse_fn(scores)

    metrics = utils.compute_metrics(y_true_ref, fused, ks=config.TOPK_LIST)
    metrics['combo'] = combo_name
    metrics['feature'] = feature_name
    metrics['data_source'] = data_source
    metrics['models'] = ','.join(model_names)
    metrics['method'] = method
    metrics['weights'] = {mn: float(w) for mn, w in zip(model_names, weights)}

    utils.save_json(metrics, config.fusion_metrics_path(
        data_source, combo_name, feature_name, method))
    np.savez_compressed(config.fusion_predictions_path(
        data_source, combo_name, feature_name, method),
        node_ids=node_ids_ref, y_true=y_true_ref, y_score=fused.astype(np.float32))

    if method == 'ap_weighted':
        weight_str = ', '.join(f'{mn}:{w:.3f}' for mn, w in zip(model_names, weights))
        print(f'  [融合 {combo_name}/{feature_name}][{method}] 权重[{weight_str}] '
              f'AUC={metrics["roc_auc"]:.4f} AP={metrics["average_precision"]:.4f} '
              f'Recall@100={metrics["recall@100"]:.4f}')
    else:
        print(f'  [融合 {combo_name}/{feature_name}][{method}] '
              f'AUC={metrics["roc_auc"]:.4f} AP={metrics["average_precision"]:.4f} '
              f'Recall@100={metrics["recall@100"]:.4f}')
    return metrics


# ============================================================
# 汇总对比表
# ============================================================
def build_summary(data_source: str) -> pd.DataFrame:
    """汇总所有单模型 + 融合的测试指标到 summary.csv。"""
    res_dir = config.out_root(data_source) / 'results'
    rows = []

    # 单模型 metrics
    for mp in sorted(res_dir.glob('*_metrics.json')):
        name = mp.stem  # {model}_{feature}_metrics
        if name.startswith('fusion_'):
            continue
        try:
            m = utils.load_json(mp)
            rows.append({
                'type': 'single',
                'model': m.get('model', ''),
                'feature': m.get('feature', ''),
                'roc_auc': m.get('roc_auc'),
                'average_precision': m.get('average_precision'),
                **{k: m.get(k) for k in [f'recall@{k}' for k in config.TOPK_LIST]},
                **{k: m.get(k) for k in [f'precision@{k}' for k in config.TOPK_LIST]},
                'val_average_precision': m.get('val_average_precision'),
            })
        except Exception as e:
            print(f'  读取 {mp} 失败: {e}')

    # 融合 metrics（含 ap_weighted 旧命名 + 其他方法新命名）
    for fp in sorted(res_dir.glob('fusion_*_metrics.json')):
        try:
            m = utils.load_json(fp)
            method = m.get('method', 'ap_weighted')
            combo = m.get('combo', '')
            rows.append({
                'type': 'fusion',
                'model': combo,
                'method': method,
                'feature': m.get('feature', ''),
                'roc_auc': m.get('roc_auc'),
                'average_precision': m.get('average_precision'),
                **{k: m.get(k) for k in [f'recall@{k}' for k in config.TOPK_LIST]},
                **{k: m.get(k) for k in [f'precision@{k}' for k in config.TOPK_LIST]},
                'val_average_precision': None,
            })
        except Exception as e:
            print(f'  读取 {fp} 失败: {e}')

    df = pd.DataFrame(rows)
    cols = ['type', 'model', 'method', 'feature', 'roc_auc', 'average_precision',
            'recall@20', 'recall@50', 'recall@100',
            'precision@20', 'precision@50', 'precision@100', 'val_average_precision']
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(config.summary_path(data_source), index=False, encoding='utf-8-sig')
    print(f'\n[S3] 汇总表已保存：{config.summary_path(data_source)}')
    if not df.empty:
        print(df.to_string(index=False))
    return df


# ============================================================
# 从磁盘加载已训练模型（供网页/复用，无需重训）
# ============================================================
def load_trained_model(model_name: str, feature_name: str, data_source: str,
                       in_dim: int):
    """从磁盘加载训练好的模型。"""
    name = model_name.lower()
    if name == 'xgboost':
        model = XGBoostModel(n_estimators=config.XGB_ROUNDS)
    elif name == 'lightgbm':
        model = LightGBMModel(n_estimators=config.LGB_ROUNDS)
    else:
        model = build_model(name, _dummy_data(in_dim, name))
    state = torch.load(config.model_path(data_source, model_name, feature_name),
                       map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    return model


def _dummy_data(in_dim: int, model_name: str = ''):
    """构造仅含形状信息的 dummy Data，用于 build_model。"""
    from torch_geometric.data import Data, HeteroData
    name = model_name.lower()
    if name == 'heterosage':
        het = HeteroData()
        het['user'].x = torch.zeros(1, in_dim)
        het['user', 'edge', 'user'].edge_index = torch.zeros(2, 1, dtype=torch.long)
        return het
    # RGCN 需要 edge_type，设置 num_relations=12（DGraphFin 有 12 种边类型）
    num_rel = 12 if name == 'rgcn' else 1
    d = Data(x=torch.zeros(1, in_dim),
             edge_index=torch.zeros(2, 1, dtype=torch.long),
             edge_type=torch.zeros(1, dtype=torch.long))
    # 对于 RGCN，设置 edge_type 范围覆盖所有关系
    if name == 'rgcn':
        d.edge_type = torch.arange(num_rel, dtype=torch.long) % num_rel
        d.edge_index = torch.zeros(2, num_rel, dtype=torch.long)
    return d


if __name__ == '__main__':
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else 'sample'
    build_summary(ds)
