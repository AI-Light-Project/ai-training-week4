# -*- coding: utf-8 -*-
"""阶段五：特征重要性分析与筛选（任务C 进阶）。

职责：
1. 用 XGBoost + LightGBM 两个决策树模型在每个特征集上训练；
2. 三种方法量化特征重要性：
   - 树内置重要性（gain / weight / cover）—— 分裂贡献；
   - Permutation Importance —— 打乱某特征后 AP 下降幅度（模型无关，最接近真实贡献）；
   - 特征相关性冗余分析 —— 高相关特征对可删其一；
3. 输出特征保留/删除建议报告，指导后续特征工程。

决策树解释性好，适合做特征筛选依据。GNN 不做这一步因为其特征交叉不可解释。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import config
import utils
from utils import plt


# ============================================================
# 特征名映射：把 dim 索引翻译成人类可读名
# ============================================================
def _degree_names(dim: int) -> List[str]:
    """degree 组：in/out/total/in*out + 各 edge_type 出度计数。"""
    names = ['in_degree', 'out_degree', 'total_degree', 'in_x_out_degree']
    num_types = dim - 4
    names += [f'out_degree_type{t}' for t in range(num_types)]
    return names


def _temporal_names(dim: int) -> List[str]:
    """temporal 组：边时间统计。"""
    base = ['edge_count', 'edge_time_mean', 'edge_time_std',
            'edge_time_min', 'edge_time_max', 'edge_time_span', 'recent_ratio']
    return base[:dim] + [f'temporal_{i}' for i in range(len(base), dim)]


def _neighbor_agg_names(dim: int) -> List[str]:
    """neighbor_agg 组：邻居特征均值 + 最大值。"""
    half = dim // 2
    return [f'neighbor_mean_{i}' for i in range(half)] + \
           [f'neighbor_max_{i}' for i in range(dim - half)]


def feature_names_from_meta(meta: dict) -> List[str]:
    """根据 meta.composition 生成每个维度的可读名称。"""
    # important 特征集：从 full 中选 top N 维度，名称映射回 full 的特征名
    meta_name = meta.get('name') or meta.get('feature_name', '')
    if meta_name == 'important' and 'selected_dims' in meta:
        # 构建 full 的特征名（与 full 的 composition 一致）
        full_composition = ['raw(17)', 'degree(16)', 'temporal(7)', 'neighbor_agg(34)',
                           'centrality(4)', 'community(3)', 'node2vec(16)']
        full_names = _names_from_composition(full_composition)
        return [full_names[d] if d < len(full_names) else f'imp_{d}'
                for d in meta['selected_dims']]
    return _names_from_composition(meta['composition'])


def _names_from_composition(composition: list) -> List[str]:
    """从 composition 列表生成特征名。"""
    names: List[str] = []
    for comp in composition:
        group, dim_str = comp.split('(')
        dim = int(dim_str.rstrip(')'))
        if group == 'raw':
            names += [f'raw_{i}' for i in range(dim)]
        elif group == 'degree':
            names += _degree_names(dim)
        elif group == 'temporal':
            names += _temporal_names(dim)
        elif group == 'neighbor_agg':
            names += _neighbor_agg_names(dim)
        elif group == 'centrality':
            base = ['pagerank', 'betweenness', 'closeness', 'clustering']
            names += base[:dim] + [f'centrality_{i}' for i in range(len(base), dim)]
        elif group == 'community':
            base = ['community_id', 'community_size', 'community_density']
            names += base[:dim] + [f'community_{i}' for i in range(len(base), dim)]
        elif group == 'neighbor_label':
            base = ['labeled_neighbor_ratio', 'known_illicit_neighbor_ratio']
            names += base[:dim] + [f'neighbor_label_{i}' for i in range(len(base), dim)]
        elif group == 'node2vec':
            names += [f'node2vec_{i}' for i in range(dim)]
        else:
            names += [f'{group}_{i}' for i in range(dim)]
    return names


# ============================================================
# 训练 XGBoost + LightGBM
# ============================================================
def _train_xgboost(x_train, y_train, x_val, y_val) -> dict:
    """训练 XGBoost，返回模型 + 内置重要性。"""
    import xgboost as xgb
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=config.XGB_ROUNDS, max_depth=6, learning_rate=0.1,
        objective='binary:logistic', eval_metric='aucpr', tree_method='hist',
        n_jobs=-1, scale_pos_weight=spw, random_state=config.SEED, verbosity=0,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    booster = model.get_booster()
    # 内置重要性：gain（增益）、weight（分裂次数）、cover（覆盖样本数）
    imp_gain = booster.get_score(importance_type='gain')
    imp_weight = booster.get_score(importance_type='weight')
    imp_cover = booster.get_score(importance_type='cover')
    return {
        'model': model, 'booster': booster,
        'imp_gain': imp_gain, 'imp_weight': imp_weight, 'imp_cover': imp_cover,
    }


def _train_lightgbm(x_train, y_train, x_val, y_val) -> dict:
    """训练 LightGBM，返回模型 + 内置重要性。"""
    import lightgbm as lgb
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    model = lgb.LGBMClassifier(
        n_estimators=config.XGB_ROUNDS, max_depth=6, learning_rate=0.1,
        objective='binary', metric='average_precision', n_jobs=-1,
        scale_pos_weight=spw, random_state=config.SEED, verbosity=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
              callbacks=[lgb.log_evaluation(0)])
    booster = model.booster_
    imp_split = booster.feature_importance(importance_type='split')
    imp_gain = booster.feature_importance(importance_type='gain')
    return {'model': model, 'booster': booster,
            'imp_split': imp_split, 'imp_gain': imp_gain}


def _xgb_imp_to_array(imp_dict: dict, num_features: int) -> np.ndarray:
    """XGBoost 的 get_score 返回 {f0: val, ...} 字典，转为数组。"""
    arr = np.zeros(num_features, dtype=np.float64)
    for k, v in imp_dict.items():
        idx = int(k.lstrip('f'))
        if idx < num_features:
            arr[idx] = v
    return arr


# ============================================================
# Permutation Importance（模型无关）
# ============================================================
def _permutation_importance(model, x_val, y_val, num_features: int,
                            n_repeats: int = 3, seed: int = 42) -> np.ndarray:
    """打乱每个特征后 AP 下降幅度，返回每特征的平均下降值。"""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    # 基线 AP
    proba = model.predict_proba(x_val)[:, 1]
    baseline_ap = average_precision_score(y_val, proba)

    drops = np.zeros((num_features, n_repeats), dtype=np.float64)
    for f in range(num_features):
        x_orig = x_val[:, f].copy()
        for r in range(n_repeats):
            x_val[:, f] = rng.permutation(x_orig)
            proba_perm = model.predict_proba(x_val)[:, 1]
            ap_perm = average_precision_score(y_val, proba_perm)
            drops[f, r] = baseline_ap - ap_perm  # 正值=重要，负值=打乱反而变好
        x_val[:, f] = x_orig  # 恢复
    return drops.mean(axis=1)


# ============================================================
# 相关性冗余分析
# ============================================================
def _correlation_redundancy(x: np.ndarray, feature_names: List[str],
                            threshold: float = 0.9) -> Tuple[np.ndarray, List[dict]]:
    """计算特征间 Pearson 相关性，找出高相关冗余对。"""
    corr = np.corrcoef(x.T)  # (F, F)
    # 用 nan 填充常数特征的相关（std=0 → corr=nan）
    corr = np.nan_to_num(corr, nan=0.0)
    redundant_pairs = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                redundant_pairs.append({
                    'feat_a': feature_names[i], 'feat_b': feature_names[j],
                    'corr': float(corr[i, j]),
                })
    return corr, redundant_pairs


# ============================================================
# 综合评分 + 保留/删除建议
# ============================================================
def _normalize(arr: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0, 1]。"""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _build_selection_report(feature_names: List[str],
                            xgb_gain: np.ndarray, xgb_weight: np.ndarray,
                            lgb_gain: np.ndarray,
                            perm_xgb: np.ndarray, perm_lgb: np.ndarray,
                            corr: np.ndarray,
                            redundant_pairs: List[dict]) -> dict:
    """综合多种方法给出特征保留/删除建议。

    评分逻辑：
    - 归一化每种重要性到 [0,1]，取均值得到 composite score；
    - Permutation 权重更高（最接近真实贡献），树内置作补充；
    - 冗余特征（与更高分特征 |corr|>0.9）标记为 redundant；
    - 建议：composite 高且不冗余 → keep；composite 低或冗余 → drop；中间 → consider。
    """
    n = len(feature_names)
    # 归一化各方法
    norm_xgb_gain = _normalize(xgb_gain)
    norm_lgb_gain = _normalize(lgb_gain)
    norm_perm_xgb = _normalize(np.maximum(perm_xgb, 0))  # 负值截断为0
    norm_perm_lgb = _normalize(np.maximum(perm_lgb, 0))

    # composite = permutation 占 60%，树内置占 40%
    composite = (0.3 * norm_perm_xgb + 0.3 * norm_perm_lgb +
                 0.2 * norm_xgb_gain + 0.2 * norm_lgb_gain)

    # 冗余标记：找每个特征是否有更高 composite 的冗余伙伴
    redundant_map = {name: False for name in feature_names}
    for pair in redundant_pairs:
        a_idx = feature_names.index(pair['feat_a'])
        b_idx = feature_names.index(pair['feat_b'])
        # composite 低的那个标记为冗余
        if composite[a_idx] >= composite[b_idx]:
            redundant_map[feature_names[b_idx]] = True
        else:
            redundant_map[feature_names[a_idx]] = True

    # 排名
    order = np.argsort(-composite)
    rank = np.zeros(n, dtype=int)
    for r, idx in enumerate(order):
        rank[idx] = r + 1

    # 建议
    threshold_drop = np.percentile(composite, 25)  # 底部25%
    feature_details = []
    for i, name in enumerate(feature_names):
        if composite[i] <= threshold_drop:
            rec = 'drop'
            reason = f'composite={composite[i]:.4f} 处于底部25%，多方法一致性低'
        elif redundant_map[name]:
            rec = 'drop'
            reason = f'与更高分特征 |corr|>0.9 冗余'
        elif composite[i] >= np.percentile(composite, 60):
            rec = 'keep'
            reason = f'composite={composite[i]:.4f} 处于顶部40%'
        else:
            rec = 'consider'
            reason = f'composite={composite[i]:.4f} 中等'

        feature_details.append({
            'dim': i, 'name': name, 'rank': int(rank[i]),
            'composite_score': float(composite[i]),
            'xgb_gain': float(xgb_gain[i]),
            'xgb_weight': float(xgb_weight[i]),
            'lgb_gain': float(lgb_gain[i]),
            'perm_xgb': float(perm_xgb[i]),
            'perm_lgb': float(perm_lgb[i]),
            'redundant': redundant_map[name],
            'recommendation': rec,
            'reason': reason,
        })

    keep = [f for f in feature_details if f['recommendation'] == 'keep']
    drop = [f for f in feature_details if f['recommendation'] == 'drop']
    consider = [f for f in feature_details if f['recommendation'] == 'consider']

    return {
        'feature_details': feature_details,
        'summary': {
            'num_keep': len(keep), 'num_drop': len(drop),
            'num_consider': len(consider),
            'keep_names': [f['name'] for f in keep],
            'drop_names': [f['name'] for f in drop],
            'consider_names': [f['name'] for f in consider],
        },
        'redundant_pairs': redundant_pairs,
    }


# ============================================================
# 可视化
# ============================================================
def _plot_importance(feature_names: List[str], composite: List[dict],
                     out_path):
    """绘制 composite score Top-25 水平柱状图。"""
    sorted_f = sorted(composite, key=lambda x: x['composite_score'], reverse=True)
    top = sorted_f[:25]
    fig, ax = plt.subplots(figsize=(9, 8))
    names = [f['name'] for f in top][::-1]
    scores = [f['composite_score'] for f in top][::-1]
    colors = ['#2ecc71' if f['recommendation'] == 'keep' else
              '#e74c3c' if f['recommendation'] == 'drop' else '#f39c12'
              for f in top][::-1]
    ax.barh(names, scores, color=colors)
    ax.set_xlabel('Composite Importance Score (归一化)')
    ax.set_title('特征重要性 Top-25（绿=保留 红=删除 橙=考虑）')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _plot_correlation(corr: np.ndarray, feature_names: List[str], out_path):
    """绘制相关性热力图。"""
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(np.abs(corr), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=5)
    ax.set_yticklabels(feature_names, fontsize=5)
    fig.colorbar(im, ax=ax, label='|Pearson r|')
    ax.set_title('特征间相关性矩阵（|r|）')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ============================================================
# 主流程
# ============================================================
def analyze_feature_set(data_source: str, feature_name: str,
                        do_permutation: bool = True) -> Optional[dict]:
    """对单个特征集做完整重要性分析。"""
    print(f'\n[S5] 特征重要性分析: {feature_name} (dataset={data_source})')

    # 加载特征 + masks
    meta = utils.load_json(config.feature_meta_path(data_source, feature_name))
    if not meta:
        print(f'  找不到 {feature_name} 的 meta，跳过')
        return None
    x = utils.load_features(data_source, feature_name).cpu().numpy().astype(np.float32)
    base = torch.load(config.out_root(data_source) / 'features' / '_base_data.pt',
                      map_location='cpu', weights_only=False)
    y = base.y.cpu().numpy().astype(np.int32)
    train_mask = base.train_mask.cpu().numpy().astype(bool)
    val_mask = base.val_mask.cpu().numpy().astype(bool)

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    num_features = x.shape[1]
    feat_names = feature_names_from_meta(meta)
    print(f'  特征维度={num_features} train={len(y_train)} val={len(y_val)}')

    # 训练 XGBoost + LightGBM
    print('  训练 XGBoost...')
    xgb_res = _train_xgboost(x_train, y_train, x_val, y_val)
    print('  训练 LightGBM...')
    lgb_res = _train_lightgbm(x_train, y_train, x_val, y_val)

    # 内置重要性
    xgb_gain = _xgb_imp_to_array(xgb_res['imp_gain'], num_features)
    xgb_weight = _xgb_imp_to_array(xgb_res['imp_weight'], num_features)
    lgb_gain = np.asarray(lgb_res['imp_gain'], dtype=np.float64)

    # Permutation importance（最耗时，仅在指定时做）
    if do_permutation:
        print(f'  Permutation importance (XGBoost, {num_features}特征 x 3重复)...')
        perm_xgb = _permutation_importance(xgb_res['model'], x_val, y_val,
                                            num_features, n_repeats=3, seed=config.SEED)
        print(f'  Permutation importance (LightGBM, {num_features}特征 x 3重复)...')
        perm_lgb = _permutation_importance(lgb_res['model'], x_val, y_val,
                                           num_features, n_repeats=3, seed=config.SEED)
    else:
        perm_xgb = np.zeros(num_features)
        perm_lgb = np.zeros(num_features)

    # 相关性冗余分析
    print('  相关性冗余分析...')
    corr, redundant_pairs = _correlation_redundancy(
        x_train, feat_names, threshold=0.9)
    print(f'  发现 {len(redundant_pairs)} 对高相关(>0.9)冗余特征')

    # 综合报告
    report = _build_selection_report(
        feat_names, xgb_gain, xgb_weight, lgb_gain,
        perm_xgb, perm_lgb, corr, redundant_pairs)
    report['feature_name'] = feature_name
    report['data_source'] = data_source
    report['num_features'] = num_features
    report['feature_names'] = feat_names

    # 保存结果
    out_dir = config.feature_importance_dir(data_source)
    utils.save_json(report, out_dir / f'importance_{feature_name}.json')
    print(f'  报告已保存: importance_{feature_name}.json')

    # 可视化
    _plot_importance(feat_names, report['feature_details'],
                     out_dir / f'importance_plot_{feature_name}.png')
    _plot_correlation(corr, feat_names, out_dir / f'correlation_{feature_name}.png')
    print(f'  图已保存: importance_plot_{feature_name}.png, correlation_{feature_name}.png')

    # 打印摘要
    s = report['summary']
    print(f'  === 特征筛选建议 ===')
    print(f'  保留(keep): {s["num_keep"]} 个')
    print(f'  考虑(consider): {s["num_consider"]} 个')
    print(f'  删除(drop): {s["num_drop"]} 个')
    if s['drop_names']:
        print(f'  建议删除: {s["drop_names"][:10]}{"..." if len(s["drop_names"]) > 10 else ""}')
    return report


def run_stage5(data_source: str, feature_names=None):
    """对所有特征集做重要性分析，full 做 permutation，其余只做内置+相关性。"""
    utils.set_seed(config.SEED)
    feature_names = feature_names or config.FEATURE_NAMES
    print('\n' + '=' * 60)
    print(f'阶段五：特征重要性分析  [dataset={data_source}]')
    print('=' * 60)

    all_reports = {}
    for fname in feature_names:
        # full 做完整 permutation（特征最多，最有筛选价值）；其余只做内置+相关性
        do_perm = (fname == 'full') or (fname == 'topology')
        report = analyze_feature_set(data_source, fname, do_permutation=do_perm)
        if report is not None:
            all_reports[fname] = report

    # 汇总跨特征集的洞察
    _print_cross_insights(all_reports)
    return all_reports


def _print_cross_insights(reports: dict):
    """打印跨特征集的洞察（哪些特征组整体更重要）。"""
    print('\n' + '=' * 60)
    print('跨特征集洞察')
    print('=' * 60)
    for fname, rep in reports.items():
        details = rep['feature_details']
        # 按特征组前缀聚合平均 composite
        groups = {}
        for d in details:
            prefix = d['name'].split('_')[0] if '_' in d['name'] else d['name']
            groups.setdefault(prefix, []).append(d['composite_score'])
        print(f'\n  [{fname}] 各特征组平均重要性:')
        for g in sorted(groups, key=lambda k: -np.mean(groups[k])):
            scores = groups[g]
            print(f'    {g:20s}: mean={np.mean(scores):.4f} max={max(scores):.4f} '
                  f'({len(scores)}维)')


if __name__ == '__main__':
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else 'sample'
    run_stage5(ds)
