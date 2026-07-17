# -*- coding: utf-8 -*-
"""阶段六：ROC 曲线可视化与 TPR/FPR 数据保存。

职责：
1. 加载所有已训练模型的测试集预测结果（y_true, y_score）；
2. 计算每个模型的 ROC 曲线（fpr, tpr, thresholds）；
3. 按特征集 / 模型类型分组绘制 ROC 曲线；
4. 保存 TPR/FPR 原始数据为 JSON，供后续展示使用。

输出：
- output/{data_source}/roc/roc_by_feature_{feature}.png  按特征集分组
- output/{data_source}/roc/roc_by_category_{category}.png 按模型类型分组
- output/{data_source}/roc/roc_data.json  所有模型的 TPR/FPR 数据
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

import config
import utils
from utils import plt


# 模型分类
MODEL_CATEGORIES = {
    '非图模型': ['mlp', 'lightgbm', 'xgboost'],
    '同构图': ['gcn', 'graphsage', 'gat'],
    '异构图': ['heterosage', 'rgcn'],
    '时序图': ['evolvegcn', 'tcn'],
}

# 模型显示名称
MODEL_DISPLAY = {
    'mlp': 'MLP', 'lightgbm': 'LightGBM', 'xgboost': 'XGBoost',
    'gcn': 'GCN', 'graphsage': 'GraphSAGE', 'gat': 'GAT',
    'heterosage': 'HeteroSAGE', 'rgcn': 'RGCN',
    'evolvegcn': 'EvolveGCN', 'tcn': 'TCN',
}

# 特征集显示名称
FEATURE_DISPLAY = {
    'raw': 'Raw (17维)', 'structural': 'Structural (32维)',
    'temporal': 'Temporal (24维)', 'topology': 'Topology (40维)',
    'full': 'Full (97维)', 'important': 'Important (25维)',
}


def _load_all_predictions(data_source: str) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """加载所有已保存的预测结果。返回 {(model, feature): (y_true, y_score)}。"""
    pred_dir = config.out_root(data_source) / 'results'
    results = {}
    if not pred_dir.exists():
        return results
    for f in sorted(pred_dir.glob('*_predictions.npz')):
        # 文件名格式: {model}_{feature}_predictions.npz
        base = f.name.replace('_predictions.npz', '')
        parts = base.rsplit('_', 1)
        if len(parts) != 2:
            continue
        mname, fname = parts
        if fname not in config.FEATURE_NAMES:
            continue
        try:
            data = np.load(f, allow_pickle=False)
            y_true = data['y_true']
            y_score = data['y_score']
            results[(mname, fname)] = (y_true, y_score)
        except Exception as e:
            print(f'  [跳过] 加载 {base} 失败: {e}')
    return results


def _compute_roc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """计算 ROC 曲线数据。"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc': float(roc_auc),
    }


def _plot_roc_group(rocs: Dict[str, dict], title: str, save_path: Path,
                    highlight: str = None):
    """绘制一组 ROC 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    # 随机分类器基线
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')
    for name, roc_data in rocs.items():
        fpr = np.array(roc_data['fpr'])
        tpr = np.array(roc_data['tpr'])
        auc_val = roc_data['auc']
        lw = 2.5 if highlight and name == highlight else 1.5
        alpha = 1.0 if highlight and name == highlight else 0.7
        display_name = MODEL_DISPLAY.get(name, name)
        ax.plot(fpr, tpr, lw=lw, alpha=alpha,
                label=f'{display_name} (AUC={auc_val:.4f})')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_stage6(data_source: str, feature_names: list = None):
    """执行阶段六：ROC 可视化 + TPR/FPR 数据保存。"""
    print(f'\n[S6] ROC 可视化  [dataset={data_source}]')
    roc_dir = config.out_root(data_source) / 'roc'
    roc_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有预测
    predictions = _load_all_predictions(data_source)
    print(f'  加载了 {len(predictions)} 个 (model, feature) 预测结果')

    if not predictions:
        print('  [警告] 无预测数据，跳过')
        return

    # 计算所有 ROC
    all_rocs = {}  # {(model, feature): roc_data}
    for (mname, fname), (y_true, y_score) in predictions.items():
        roc_data = _compute_roc(y_true, y_score)
        all_rocs[(mname, fname)] = roc_data
        print(f'  {mname:12s}/{fname:12s}: AUC={roc_data["auc"]:.4f}')

    # 保存全部 TPR/FPR 数据为 JSON
    roc_json = {}
    for (mname, fname), roc_data in all_rocs.items():
        key = f'{mname}_{fname}'
        roc_json[key] = {
            'model': mname,
            'feature': fname,
            'model_display': MODEL_DISPLAY.get(mname, mname),
            'feature_display': FEATURE_DISPLAY.get(fname, fname),
            'fpr': roc_data['fpr'],
            'tpr': roc_data['tpr'],
            'thresholds': roc_data['thresholds'],
            'auc': roc_data['auc'],
            'num_test_nodes': int(len(predictions[(mname, fname)][0])),
        }
    roc_path = roc_dir / 'roc_data.json'
    with open(roc_path, 'w', encoding='utf-8') as f:
        json.dump(roc_json, f, ensure_ascii=False, indent=2)
    print(f'  ROC 数据已保存: {roc_path}')

    feature_names = feature_names or config.FEATURE_NAMES

    # 1) 按特征集分组绘制（每个特征集一张图，展示所有模型）
    for fname in feature_names:
        rocs_in_feat = {}
        for (mname, fn), roc_data in all_rocs.items():
            if fn == fname:
                rocs_in_feat[mname] = roc_data
        if not rocs_in_feat:
            continue
        title = f'{data_source} - {FEATURE_DISPLAY.get(fname, fname)} ROC 曲线'
        save_path = roc_dir / f'roc_by_feature_{fname}.png'
        _plot_roc_group(rocs_in_feat, title, save_path)
        print(f'  已保存: {save_path.name}')

    # 2) 按模型类型分组绘制（每类一张图，展示该类所有模型在 full/important 上的表现）
    for cat_name, model_list in MODEL_CATEGORIES.items():
        rocs_in_cat = {}
        for (mname, fn), roc_data in all_rocs.items():
            if mname in model_list and fn in ('full', 'important', 'topology'):
                key = f'{MODEL_DISPLAY.get(mname, mname)} [{fn}]'
                rocs_in_cat[key] = roc_data
        if not rocs_in_cat:
            continue
        title = f'{data_source} - {cat_name} ROC 对比'
        save_path = roc_dir / f'roc_by_category_{cat_name}.png'
        _plot_roc_group(rocs_in_cat, title, save_path)
        print(f'  已保存: {save_path.name}')

    # 3) 总览图：所有模型在 full 特征集上的 ROC 对比
    rocs_full = {}
    for (mname, fn), roc_data in all_rocs.items():
        if fn == 'full':
            rocs_full[mname] = roc_data
    if rocs_full:
        title = f'{data_source} - Full 特征集 全模型 ROC 对比'
        save_path = roc_dir / 'roc_overview_full.png'
        _plot_roc_group(rocs_full, title, save_path)
        print(f'  已保存: {save_path.name}')

    # 4) GNN 对比图：同构/异构/时序在 raw/important/full 上的 ROC
    gnn_models = ['gcn', 'graphsage', 'gat', 'heterosage', 'rgcn', 'evolvegcn', 'tcn']
    for feat in ['raw', 'important', 'full']:
        rocs_gnn = {}
        for (mname, fn), roc_data in all_rocs.items():
            if mname in gnn_models and fn == feat:
                rocs_gnn[mname] = roc_data
        if not rocs_gnn:
            continue
        title = f'{data_source} - GNN 对比 ({FEATURE_DISPLAY.get(feat, feat)})'
        save_path = roc_dir / f'roc_gnn_{feat}.png'
        _plot_roc_group(rocs_gnn, title, save_path)
        print(f'  已保存: {save_path.name}')

    print(f'  [S6] 完成，共生成 {len(list(roc_dir.glob("*.png")))} 张图')


if __name__ == '__main__':
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else 'sample'
    run_stage6(ds)
