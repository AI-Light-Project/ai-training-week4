# -*- coding: utf-8 -*-
"""DGraphFin 大作业全局配置。

统一管理数据路径、训练超参数、设备选择和输出目录。
所有阶段的脚本都从这里读取配置，保证一致性。
"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根目录：0717-proj/
PROJECT_DIR = Path(__file__).resolve().parent

# 数据目录
DATA_DIR = PROJECT_DIR / 'data'
SAMPLE_PT = DATA_DIR / 'dgraphfin_sample' / 'dgraphfin_sample.pt'
FULL_NPZ = DATA_DIR / 'DGraphFin' / 'dgraphfin.npz'

# 输出根目录：output/，下分 sample / full 两个数据集目录
OUTPUT_DIR = PROJECT_DIR / 'output'

# ---------- 训练超参数 ----------
SEED = 42
EPOCHS = 30              # 图模型训练轮数（含早停，最大30轮）
MLP_EPOCHS = 30          # MLP 同样30轮
XGB_ROUNDS = 400         # XGBoost 迭代轮数
LGB_ROUNDS = 400         # LightGBM 迭代轮数
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
GAT_HEADS = 2
LR = 0.005
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 1
EARLY_STOP_PATIENCE = 7  # 早停耐心：验证集 AP 连续7轮不提升则终止

# Recall@K / Precision@K 的 K 值列表（模拟每天只能审核 Top-K 个用户）
TOPK_LIST = (20, 50, 100)
DEFAULT_TOPK = 100

# ---------- 特征工程配置 ----------
# feature_name -> 是否需要图结构 / 是否需要边时间
FEATURE_NAMES = ['raw', 'structural', 'temporal', 'topology', 'full', 'important']

# important 特征集：基于 s5 特征重要性分析的累积贡献度阈值选择
# 策略：XGBoost+LightGBM composite_score 降序（permutation 60% + 树内置 40%），
#       归一化为贡献度权重后累积，选取累积贡献度首次 >= 80% 的最小特征子集。
# 当前结果（基于 full 107维 importance_full.json）：
#   - 总特征数: 107
#   - 选中特征数: 37
#   - 最终累积贡献度: 80.33%
#   - 维度约简率: 65.42%
# 重新选择命令：.venv/Scripts/python.exe select_important_features.py
IMPORTANT_FEATURE_DIMS = [88, 2, 86, 84, 69, 46, 17, 1, 44, 47, 52, 45, 98, 15, 19, 100, 87, 106, 97, 101, 7, 3, 11, 92, 18, 96, 39, 91, 48, 93, 95, 105, 99, 104, 37, 65, 103]

# node2vec 嵌入维度
NODE2VEC_DIM = 16

# ---------- 模型列表 ----------
# 同构图模型 + 非图模型
MODEL_NAMES = ['mlp', 'lightgbm', 'xgboost', 'gcn', 'graphsage', 'gat']
# 异构图模型（需要 edge_type）
HETERO_MODEL_NAMES = ['heterosage', 'rgcn']
# 时间感知模型（需要 edge_time 快照）
TEMPORAL_MODEL_NAMES = ['evolvegcn', 'tcn']
# 全部模型
ALL_MODEL_NAMES = MODEL_NAMES + HETERO_MODEL_NAMES + TEMPORAL_MODEL_NAMES

# 融合使用的模型组合（这些模型都输出 test 概率，便于加权融合）
#
# 四类模型特性互补设计：
#   非图模型      : mlp / lightgbm / xgboost
#                  —— 仅看节点自身特征，捕捉属性层面的非线性关系；
#                  —— LightGBM/XGBoost 对表格特征强、训练快、可解释。
#   同构GNN       : gcn / graphsage / gat
#                  —— 聚合一阶邻居信息，建模局部结构；
#                  —— GraphSAGE 采样可扩展，GAT 注意力可聚焦关键邻居。
#   异构GNN       : heterosage / rgcn
#                  —— 显式建模边类型（关系）语义；
#                  —— RGCN 按关系分解权重，对多关系图表达力强。
#   时序GNN       : evolvegcn / tcn
#                  —— 在时间快照上建模节点表示演化；
#                  —— EvolveGCN 用 GRU+GCN 追踪时序动态，TCN 用因果卷积。
#
# 互补原则：每类模型从不同视角审视节点风险（属性/结构/关系/时间），
#           跨类组合可同时覆盖这些视角，预期优于同类组合。
#
# 组合设计（每类选 1 个代表模型以控制组合规模）：
#   非图=lightgbm  同构=graphsage  异构=rgcn  时序=evolvegcn
#   （在 sample full/important/topology/raw 特征集上 10 个模型均有 predictions）
FUSION_COMBOS = {
    # --- 原有同类组合（向后兼容） ---
    'mlp+graphsage': ['mlp', 'graphsage'],
    'mlp+gat': ['mlp', 'gat'],
    'graphsage+gat': ['graphsage', 'gat'],
    'mlp+graphsage+gat': ['mlp', 'graphsage', 'gat'],

    # --- 跨类 2 模型（6 个：4 类中任取 2 类的代表模型） ---
    'lightgbm+graphsage': ['lightgbm', 'graphsage'],       # 非图+同构
    'lightgbm+rgcn': ['lightgbm', 'rgcn'],                 # 非图+异构
    'lightgbm+evolvegcn': ['lightgbm', 'evolvegcn'],       # 非图+时序
    'graphsage+rgcn': ['graphsage', 'rgcn'],               # 同构+异构
    'graphsage+evolvegcn': ['graphsage', 'evolvegcn'],     # 同构+时序
    'rgcn+evolvegcn': ['rgcn', 'evolvegcn'],               # 异构+时序

    # --- 跨类 3 模型（4 个：4 类中任取 3 类） ---
    'lightgbm+graphsage+rgcn': ['lightgbm', 'graphsage', 'rgcn'],            # 非图+同构+异构
    'lightgbm+graphsage+evolvegcn': ['lightgbm', 'graphsage', 'evolvegcn'],  # 非图+同构+时序
    'lightgbm+rgcn+evolvegcn': ['lightgbm', 'rgcn', 'evolvegcn'],            # 非图+异构+时序
    'graphsage+rgcn+evolvegcn': ['graphsage', 'rgcn', 'evolvegcn'],          # 同构+异构+时序

    # --- 全类型 4 模型（2 个：四类各取 1 个代表） ---
    'lightgbm+graphsage+rgcn+evolvegcn': ['lightgbm', 'graphsage', 'rgcn', 'evolvegcn'],
    'lightgbm+graphsage+rgcn+tcn': ['lightgbm', 'graphsage', 'rgcn', 'tcn'],
}

# 融合方法（不同类型的概率聚合策略）
#   ap_weighted : 验证集 AP 归一化加权（现有默认，考虑模型可靠性）
#   mean        : 等权算术平均（简单基线，所有模型同等重要）
#   max         : 取最大预测概率（风控保守策略，宁误报不漏报，高召回）
#   rank        : 倒数排名融合 RRF（尺度无关，消除概率分布差异）
#   geomean     : 几何平均（对低概率敏感，要求所有模型一致认为高风险）
FUSION_METHODS = ['ap_weighted', 'mean', 'max', 'rank', 'geomean']


def get_device(data_source: str) -> 'torch.device':
    """根据数据来源选择设备。

    sample 与 full 均优先使用 GPU（RTX 4060 8GB 显存足够）。
    若 GPU 不可用则回退 CPU。
    """
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_data_path(data_source: str) -> Path:
    """返回指定数据来源的文件路径。"""
    if data_source == 'sample':
        return SAMPLE_PT
    if data_source == 'full':
        return FULL_NPZ
    raise ValueError(f"data_source 必须是 'sample' 或 'full'，当前为: {data_source}")


def out_root(data_source: str) -> Path:
    """返回指定数据集的输出根目录：output/{sample|full}/。"""
    assert data_source in ('sample', 'full'), data_source
    root = OUTPUT_DIR / data_source
    (root / 'features').mkdir(parents=True, exist_ok=True)
    (root / 'models').mkdir(parents=True, exist_ok=True)
    (root / 'results').mkdir(parents=True, exist_ok=True)
    (root / 'interpret').mkdir(parents=True, exist_ok=True)
    return root


def feature_path(data_source: str, feature_name: str) -> Path:
    """特征张量保存路径：output/{ds}/features/{feature_name}.pt。"""
    return out_root(data_source) / 'features' / f'{feature_name}.pt'


def feature_meta_path(data_source: str, feature_name: str) -> Path:
    return out_root(data_source) / 'features' / f'{feature_name}_meta.json'


def model_path(data_source: str, model_name: str, feature_name: str) -> Path:
    """模型保存路径：output/{ds}/models/{model}_{feature}.pt。"""
    return out_root(data_source) / 'models' / f'{model_name}_{feature_name}.pt'


def model_config_path(data_source: str, model_name: str, feature_name: str) -> Path:
    return out_root(data_source) / 'models' / f'{model_name}_{feature_name}_config.json'


def metrics_path(data_source: str, model_name: str, feature_name: str) -> Path:
    """测试指标保存路径：output/{ds}/results/{model}_{feature}_metrics.json。"""
    return out_root(data_source) / 'results' / f'{model_name}_{feature_name}_metrics.json'


def history_path(data_source: str, model_name: str, feature_name: str) -> Path:
    """训练历史（每轮 loss / 指标）保存路径，用于绘制曲线。"""
    return out_root(data_source) / 'results' / f'{model_name}_{feature_name}_history.json'


def predictions_path(data_source: str, model_name: str, feature_name: str) -> Path:
    """测试集预测结果保存路径：node_ids / y_true / y_score，供后续网页可视化。"""
    return out_root(data_source) / 'results' / f'{model_name}_{feature_name}_predictions.npz'


def summary_path(data_source: str) -> Path:
    return out_root(data_source) / 'results' / 'summary.csv'


def fusion_metrics_path(data_source: str, combo_name: str, feature_name: str,
                        method: str = 'ap_weighted') -> Path:
    """融合指标保存路径。

    ap_weighted 方法使用旧命名（向后兼容）；其他方法带 method 后缀。
    """
    if method == 'ap_weighted':
        return out_root(data_source) / 'results' / f'fusion_{combo_name}_{feature_name}_metrics.json'
    return out_root(data_source) / 'results' / f'fusion_{combo_name}_{method}_{feature_name}_metrics.json'


def fusion_predictions_path(data_source: str, combo_name: str, feature_name: str,
                            method: str = 'ap_weighted') -> Path:
    """融合预测保存路径（命名规则同 fusion_metrics_path）。"""
    if method == 'ap_weighted':
        return out_root(data_source) / 'results' / f'fusion_{combo_name}_{feature_name}_predictions.npz'
    return out_root(data_source) / 'results' / f'fusion_{combo_name}_{method}_{feature_name}_predictions.npz'


def data_stats_path(data_source: str) -> Path:
    return out_root(data_source) / 'results' / 'data_stats.json'


def feature_importance_dir(data_source: str) -> Path:
    """特征重要性分析输出目录：output/{ds}/feature_importance/。"""
    d = out_root(data_source) / 'feature_importance'
    d.mkdir(parents=True, exist_ok=True)
    return d
