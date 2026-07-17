# -*- coding: utf-8 -*-
"""基于累积贡献度阈值选择 important 特征集。

策略：
1. 读取 s5 生成的 importance_full.json（含每个特征的 composite_score）；
2. 按 composite_score 降序排序；
3. 计算累积贡献度（归一化后累加）；
4. 选取累积贡献度首次 >= 阈值(80%)的最小特征子集；
5. 输出选中的 dim 索引列表，更新 config.IMPORTANT_FEATURE_DIMS。

注意：composite_score 已经是归一化到 [0,1] 的得分，
但不同特征的得分之和可能不为1，需重新归一化得到"贡献度权重"。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def select_by_cumulative_contribution(importance_json: Path,
                                      threshold: float = 0.80) -> dict:
    """基于累积贡献度选择特征。

    Returns:
        dict 包含:
        - selected_dims: 选中特征的 dim 索引列表（按 composite 降序）
        - selected_names: 选中特征名
        - selected_scores: 选中特征的 composite_score
        - cumulative_contrib: 累积贡献度列表
        - threshold: 使用阈值
        - total_features: 总特征数
        - selected_count: 选中特征数
        - final_cumulative: 最终累积贡献度
    """
    with open(importance_json, encoding='utf-8') as f:
        report = json.load(f)

    details = report['feature_details']
    # 提取 (dim, name, composite_score)
    items = [(d['dim'], d['name'], d['composite_score']) for d in details]
    # 按 composite_score 降序
    items.sort(key=lambda x: -x[2])

    dims = np.array([it[0] for it in items])
    names = [it[1] for it in items]
    scores = np.array([it[2] for it in items])

    # 归一化为贡献度权重（和为1）
    total = scores.sum()
    if total <= 0:
        raise ValueError('composite_score 总和非正，无法计算贡献度')
    weights = scores / total
    cumulative = np.cumsum(weights)

    # 选取首次 >= threshold 的最小子集
    # 至少选1个特征；若所有特征加起来都不到阈值（理论上不会），选全部
    idx_cutoff = int(np.searchsorted(cumulative, threshold)) + 1
    idx_cutoff = max(1, min(idx_cutoff, len(items)))

    selected_dims = dims[:idx_cutoff].tolist()
    selected_names = names[:idx_cutoff]
    selected_scores = scores[:idx_cutoff].tolist()
    final_cumulative = float(cumulative[idx_cutoff - 1])

    return {
        'selected_dims': selected_dims,
        'selected_names': selected_names,
        'selected_scores': selected_scores,
        'cumulative_contrib': cumulative.tolist(),
        'threshold': threshold,
        'total_features': len(items),
        'selected_count': idx_cutoff,
        'final_cumulative': final_cumulative,
    }


def format_report(result: dict) -> str:
    """格式化报告字符串。"""
    lines = []
    lines.append('=' * 70)
    lines.append('important 特征集选择报告（基于累积贡献度）')
    lines.append('=' * 70)
    lines.append(f"总特征数: {result['total_features']}")
    lines.append(f"阈值: {result['threshold']*100:.0f}%")
    lines.append(f"选中特征数: {result['selected_count']}")
    lines.append(f"最终累积贡献度: {result['final_cumulative']*100:.2f}%")
    lines.append(f"维度约简率: {(1 - result['selected_count']/result['total_features'])*100:.2f}%")
    lines.append('')
    lines.append('选中特征详情（按 composite_score 降序）:')
    lines.append(f"{'排名':<5}{'dim':<6}{'特征名':<30}{'score':<12}{'累积贡献':<12}")
    lines.append('-' * 70)
    for i, (dim, name, score) in enumerate(zip(
            result['selected_dims'], result['selected_names'],
            result['selected_scores'])):
        cum = result['cumulative_contrib'][i]
        lines.append(f"{i+1:<5}{dim:<6}{name:<30}{score:<12.4f}{cum*100:<12.2f}%")
    lines.append('')
    lines.append(f"Python 列表格式（用于 config.IMPORTANT_FEATURE_DIMS）:")
    lines.append(f"IMPORTANT_FEATURE_DIMS = {result['selected_dims']}")
    return '\n'.join(lines)


def main():
    import sys
    proj_root = Path(__file__).resolve().parent
    importance_json = proj_root / 'output' / 'sample' / 'feature_importance' / 'importance_full.json'

    if not importance_json.exists():
        print(f'找不到 {importance_json}，请先运行 stage 5')
        sys.exit(1)

    threshold = 0.80
    result = select_by_cumulative_contribution(importance_json, threshold)
    print(format_report(result))

    # 保存结果到 JSON
    out_path = proj_root / 'output' / 'sample' / 'feature_importance' / 'important_selection_80pct.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n选择结果已保存: {out_path}')


if __name__ == '__main__':
    main()
