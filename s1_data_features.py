# -*- coding: utf-8 -*-
"""阶段一：数据理解与特征提取（任务A + 特征工程）。

职责：
1. 加载数据（sample 课堂采样 / full 全量），标准化为 PyG Data；
2. 任务A 数据理解：统计基础信息、边类型/时间分布、正常/异常节点度分布，输出业务洞察；
3. 特征工程：构造 raw / structural / temporal / full 四种特征集并持久化。

红线：
- 绝不用测试集标签构造特征（如邻居异常比例）；
- 保留背景节点（标签2/3）维护图连通性，只做二分类 mask。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import config
import utils
from utils import plt


# ============================================================
# 数据加载
# ============================================================
def load_sample_data(sample_pt: Path) -> Data:
    if not sample_pt.exists():
        raise FileNotFoundError(f'找不到采样数据: {sample_pt}')
    try:
        obj = torch.load(sample_pt, map_location='cpu', weights_only=False)
    except TypeError:
        obj = torch.load(sample_pt, map_location='cpu')

    if isinstance(obj, Data):
        data = obj
    elif isinstance(obj, dict) and 'data' in obj:
        data = obj['data']
    else:
        raise TypeError('sample_pt 必须是 PyG Data 或含 key `data` 的 dict。')
    return utils.normalize_data(data)


def load_full_npz(npz_path: Path) -> Data:
    if not npz_path.exists():
        raise FileNotFoundError(f'找不到全量 npz: {npz_path}')
    with np.load(npz_path, allow_pickle=False) as loader:
        required = {'x', 'y', 'edge_index', 'edge_type', 'edge_timestamp',
                    'train_mask', 'valid_mask', 'test_mask'}
        missing = sorted(required - set(loader.files))
        if missing:
            raise KeyError(f'npz 缺少键: {missing}。可用键: {loader.files}')

        x = torch.from_numpy(loader['x']).float()
        y = torch.from_numpy(loader['y']).view(-1).long()
        N = x.size(0)
        edge_index = torch.from_numpy(loader['edge_index']).long()
        if edge_index.dim() != 2:
            raise ValueError(f'edge_index 应为 2D，得到 shape={tuple(edge_index.shape)}')
        if edge_index.size(0) != 2 and edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()

        # mask 在 npz 中可能是布尔数组（长度=N）或节点索引数组（长度< N）。
        # 这里直接透传原始 long 张量，由 utils.normalize_data -> ensure_bool_masks
        # 统一通过 index_to_mask 转换为布尔 mask，避免重复转换导致错误。
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index.contiguous(),
            edge_type=torch.from_numpy(loader['edge_type']).view(-1).long(),
            edge_time=torch.from_numpy(loader['edge_timestamp']).view(-1).long(),
            train_mask=torch.from_numpy(loader['train_mask']).view(-1).long(),
            val_mask=torch.from_numpy(loader['valid_mask']).view(-1).long(),
            test_mask=torch.from_numpy(loader['test_mask']).view(-1).long(),
        )
    return utils.normalize_data(data)


def load_data(data_source: str) -> Data:
    if data_source == 'sample':
        return load_sample_data(config.SAMPLE_PT)
    if data_source == 'full':
        return load_full_npz(config.FULL_NPZ)
    raise ValueError(f"data_source 必须是 'sample' 或 'full'，当前为: {data_source}")


# ============================================================
# 任务A：数据理解与统计
# ============================================================
def split_summary(data: Data) -> pd.DataFrame:
    rows = []
    for split_name, mask_name in [('train', 'train_mask'), ('val', 'val_mask'), ('test', 'test_mask')]:
        mask = getattr(data, mask_name)
        labels = data.y[mask].cpu()
        counts = torch.bincount(labels, minlength=2)
        total = int(mask.sum())
        rows.append({
            'split': split_name,
            'num_nodes': total,
            'label_0': int(counts[0]),
            'label_1': int(counts[1]),
            'positive_ratio': float(counts[1] / max(total, 1)),
        })
    return pd.DataFrame(rows)


def task_a_analysis(data: Data, data_source: str) -> dict:
    """任务A：统计基础信息 + 分布 + 业务洞察，保存 JSON 与图片。"""
    stats: dict = {}
    stats['data_source'] = data_source
    stats['num_nodes'] = int(data.num_nodes)
    stats['num_edges'] = int(data.edge_index.size(1))
    stats['num_features'] = int(data.x.size(-1))
    stats['all_label_counts'] = torch.bincount(
        data.y.cpu(), minlength=int(data.y.max()) + 1).tolist()

    stats['split_summary'] = split_summary(data).to_dict(orient='records')

    # 边类型分布
    edge_type = getattr(data, 'edge_type', None)
    if edge_type is not None:
        et = edge_type.cpu()
        uniq, counts = torch.unique(et, return_counts=True)
        stats['edge_type_counts'] = {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}

    # 边时间统计
    edge_time = getattr(data, 'edge_time', None)
    if edge_time is not None:
        eti = edge_time.cpu().float()
        stats['edge_time'] = {
            'min': int(eti.min()), 'max': int(eti.max()),
            'mean': float(eti.mean()), 'median': float(eti.median()),
            'q25': float(torch.quantile(eti, 0.25)),
            'q75': float(torch.quantile(eti, 0.75)),
        }

    # 度分布（按正常/异常节点对比）
    row, col = data.edge_index
    N = data.num_nodes
    out_degree = torch.bincount(row, minlength=N)
    in_degree = torch.bincount(col, minlength=N)
    total_degree = in_degree + out_degree

    binary = (data.y == 0) | (data.y == 1)
    deg_by_label = {}
    for lab in [0, 1]:
        m = binary & (data.y == lab)
        degs = total_degree[m].cpu().numpy()
        deg_by_label[f'label_{lab}'] = {
            'num_nodes': int(m.sum()),
            'in_degree_mean': float(in_degree[m].float().mean()),
            'out_degree_mean': float(out_degree[m].float().mean()),
            'total_degree_mean': float(degs.mean()) if len(degs) else 0.0,
            'total_degree_median': float(np.median(degs)) if len(degs) else 0.0,
        }
    stats['degree_by_label'] = deg_by_label

    # 业务洞察（自动生成结论）
    insights = []
    pos_ratio = stats['split_summary'][0]['positive_ratio']
    insights.append(f"正样本（异常）占比仅 {pos_ratio:.4f}，类别极度不平衡，禁止只看 Accuracy。")
    if deg_by_label.get('label_1') and deg_by_label.get('label_0'):
        out1 = deg_by_label['label_1']['out_degree_mean']
        out0 = deg_by_label['label_0']['out_degree_mean']
        in1 = deg_by_label['label_1']['in_degree_mean']
        if out1 > in1:
            insights.append(
                f"异常节点平均出度({out1:.2f})高于入度({in1:.2f})，"
                "说明异常用户倾向于将更多人设为联系人，但较少被他人关联（团伙发散特征）。")
        if out1 > out0:
            insights.append(
                f"异常节点平均出度({out1:.2f})高于正常节点({out0:.2f})，"
                "出度可作为风控线索。")
    if edge_type is not None:
        # 异常源节点各类型边占比 vs 正常源节点，找出风险关系类型
        src = row
        src_label = data.y[src]
        fraud_src = (src_label == 1)
        if fraud_src.sum() > 0:
            num_types = int(edge_type.max()) + 1
            fraud_type_counts = torch.bincount(edge_type[fraud_src], minlength=num_types)
            normal_type_counts = torch.bincount(edge_type[src_label == 0], minlength=num_types)
            fraud_ratio = fraud_type_counts.float() / max(int(fraud_type_counts.sum()), 1)
            normal_ratio = normal_type_counts.float() / max(int(normal_type_counts.sum()), 1)
            diff = (fraud_ratio - normal_ratio)
            top_t = int(diff.argmax())
            # 保存原始对比数据供后续展示（按 edge_type 排序）
            edge_type_risk = []
            for t in range(num_types):
                edge_type_risk.append({
                    'edge_type': int(t),
                    'fraud_count': int(fraud_type_counts[t]),
                    'normal_count': int(normal_type_counts[t]),
                    'fraud_ratio': float(fraud_ratio[t]),
                    'normal_ratio': float(normal_ratio[t]),
                    'diff': float(diff[t]),
                })
            # 按 diff 降序排序（风险关系类型优先）
            edge_type_risk.sort(key=lambda x: x['diff'], reverse=True)
            stats['edge_type_risk_analysis'] = {
                'description': '异常源节点 vs 正常源节点各 edge_type 边占比对比',
                'fields': {
                    'fraud_ratio': '异常源节点该类型边数 / 异常源节点总边数',
                    'normal_ratio': '正常源节点该类型边数 / 正常源节点总边数',
                    'diff': 'fraud_ratio - normal_ratio，正值表示异常偏向该类型',
                },
                'sorted_by': 'diff 降序',
                'top_risk_type': int(top_t),
                'per_type': edge_type_risk,
            }
            insights.append(
                f"edge_type={top_t} 在异常节点中占比({fraud_ratio[top_t]:.4f})"
                f"显著高于正常节点({normal_ratio[top_t]:.4f})，可能对应高风险关系类型。")
    if edge_time is not None:
        insights.append(
            "edge_time 非简单类别，应关注关系'何时发生'及是否集中爆发（团伙集中注册信号）。")
    insights.append(
        "背景节点(标签2/3)必须保留以维护图连通性，作为远距离风险信号传递桥梁，不可删除。")
    stats['insights'] = insights

    # 保存 JSON
    utils.save_json(stats, config.data_stats_path(data_source))

    # 保存图片
    _plot_task_a(data, data_source, stats, total_degree, binary, edge_type, edge_time)
    print('[任务A] 统计完成，业务洞察：')
    for s in insights:
        print('  -', s)
    return stats


def _plot_task_a(data, data_source, stats, total_degree, binary, edge_type, edge_time):
    res_dir = config.out_root(data_source) / 'results'
    N = data.num_nodes

    # 1) 边类型分布柱状图
    if edge_type is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        keys = sorted(stats['edge_type_counts'].keys())
        vals = [stats['edge_type_counts'][k] for k in keys]
        ax.bar([str(k) for k in keys], vals)
        ax.set_xlabel('edge_type')
        ax.set_ylabel('边数')
        ax.set_title(f'{data_source} 边类型分布')
        plt.tight_layout()
        plt.savefig(res_dir / 'data_edge_type_dist.png', dpi=120)
        plt.close()

    # 2) 边时间分布直方图
    if edge_time is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(edge_time.cpu().numpy(), bins=50, color='steelblue', edgecolor='white')
        ax.set_xlabel('edge_time')
        ax.set_ylabel('边数')
        ax.set_title(f'{data_source} 边时间分布')
        plt.tight_layout()
        plt.savefig(res_dir / 'data_edge_time_dist.png', dpi=120)
        plt.close()

    # 3) 度分布对比（正常 vs 异常，log 尺度）
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, color in [(0, 'steelblue'), (1, 'crimson')]:
        m = binary & (data.y == lab)
        degs = total_degree[m].cpu().numpy()
        ax.hist(degs, bins=40, range=(0, max(degs.max(), 1)), alpha=0.6,
                label=f'y={lab} ({"正常" if lab == 0 else "异常"})', color=color, density=True)
    ax.set_xlabel('total_degree')
    ax.set_ylabel('密度')
    ax.set_title(f'{data_source} 正常/异常节点度分布对比')
    ax.legend()
    plt.tight_layout()
    plt.savefig(res_dir / 'data_degree_by_label.png', dpi=120)
    plt.close()

    # 4) 异常率随度分桶变化
    deg_np = total_degree.cpu().numpy()
    y_bin = data.y.cpu().numpy()
    m_bin = binary.cpu().numpy()
    df = pd.DataFrame({'deg': deg_np, 'y': y_bin})
    df = df[m_bin]
    df['deg_bin'] = pd.cut(df['deg'], bins=[-1, 0, 1, 2, 3, 5, 10, np.inf],
                           labels=['0', '1', '2', '3', '4-5', '6-10', '>10'])
    agg = df.groupby('deg_bin', observed=True).agg(num=('y', 'size'),
                                                   fraud_rate=('y', 'mean')).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(agg['deg_bin'].astype(str), agg['fraud_rate'], color='darkorange')
    ax.set_xlabel('total_degree 分桶')
    ax.set_ylabel('异常率')
    ax.set_title(f'{data_source} 不同度数桶的异常率')
    for i, v in enumerate(agg['fraud_rate']):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(res_dir / 'data_fraud_rate_by_degree.png', dpi=120)
    plt.close()


# ============================================================
# 特征工程
# ============================================================
def _standardize(x: torch.Tensor, train_mask: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """用 train 节点的均值/方差标准化（避免 val/test 泄漏）。"""
    xt = x.float()
    mu = xt[train_mask].mean(dim=0)
    std = xt[train_mask].std(dim=0)
    std = std.clamp(min=1e-6)
    xs = (xt - mu) / std
    xs = torch.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    return xs, {'mean': mu.tolist(), 'std': std.tolist()}


def _degree_features(data: Data) -> torch.Tensor:
    """入度/出度/总度 + 各 edge_type 入度与出度计数。

    DGraphFin 的 edge_type 从 1 开始编号（1~11 共 11 种），
    用 torch.unique 而非 range(max+1)，避免多生成全 0 的空列。
    """
    row, col = data.edge_index
    N = data.num_nodes
    out_deg = torch.bincount(row, minlength=N).float().view(-1, 1)
    in_deg = torch.bincount(col, minlength=N).float().view(-1, 1)
    total_deg = (in_deg + out_deg)

    feats = [in_deg, out_deg, total_deg, in_deg * out_deg]
    # 各 edge_type 入度与出度计数
    et = getattr(data, 'edge_type', None)
    if et is not None:
        et = et.long()
        uniq_types = torch.unique(et)
        for t in uniq_types.tolist():
            tm = (et == t).float()
            # 出度：该节点作为源点发出该类型边的数量
            feats.append(torch.bincount(row, minlength=N, weights=tm).view(-1, 1))
            # 入度：该节点作为终点接收该类型边的数量
            feats.append(torch.bincount(col, minlength=N, weights=tm).view(-1, 1))
    return torch.cat(feats, dim=1)


def _temporal_features(data: Data) -> torch.Tensor:
    """基于 edge_time 的节点级时间特征。

    刻画"关系何时发生""是否集中爆发"：
    - 每个节点作为源/目的的边时间统计：min/max/mean/std/span/count
    - 近期活跃度：最近时间窗内的边数占比
    """
    N = data.num_nodes
    et = getattr(data, 'edge_time', None)
    if et is None:
        return torch.zeros(N, 1)
    et = et.float()
    row, col = data.edge_index

    def scatter_stats(idx):
        """对节点 idx 维度做 edge_time 的 sum/sum_sq/count/min/max。"""
        count = torch.bincount(idx, minlength=N).float()
        s = torch.bincount(idx, minlength=N, weights=et).float()
        sq = torch.bincount(idx, minlength=N, weights=et * et).float()
        mn = torch.full((N,), float(et.max()) + 1, dtype=torch.float)
        mx = torch.full((N,), float(et.min()) - 1, dtype=torch.float)
        mn.scatter_reduce_(0, idx, et, reduce='amin', include_self=False)
        mx.scatter_reduce_(0, idx, et, reduce='amax', include_self=False)
        return count, s, sq, mn, mx

    cnt_r, s_r, sq_r, mn_r, mx_r = scatter_stats(row)
    cnt_c, s_c, sq_c, mn_c, mx_c = scatter_stats(col)

    cnt = cnt_r + cnt_c
    s = s_r + s_c
    sq = sq_r + sq_c
    mean = s / cnt.clamp(min=1)
    var = (sq / cnt.clamp(min=1)) - mean * mean
    std = var.clamp(min=0).sqrt()
    mn = torch.minimum(mn_r, mn_c)
    mx = torch.maximum(mx_r, mx_c)
    span = mx - mn

    t_max = float(et.max())
    t_min = float(et.min())
    span_total = max(t_max - t_min, 1.0)
    # 近期活跃度：最近 1/4 时间窗内边数占比
    recent_thr = t_max - span_total / 4
    recent_mask = (et >= recent_thr).float()
    recent_cnt_r = torch.bincount(row, minlength=N, weights=recent_mask).float()
    recent_cnt_c = torch.bincount(col, minlength=N, weights=recent_mask).float()
    recent_ratio = (recent_cnt_r + recent_cnt_c) / cnt.clamp(min=1)

    feats = [cnt.view(-1, 1), mean.view(-1, 1), std.view(-1, 1),
             mn.view(-1, 1), mx.view(-1, 1), span.view(-1, 1), recent_ratio.view(-1, 1)]
    return torch.cat(feats, dim=1)


def _neighbor_agg_features(data: Data) -> torch.Tensor:
    """邻居原始特征的聚合：均值 / 最大值（刻画局部风险环境）。

    用无向边聚合，避免方向偏置。
    """
    N, D = data.x.shape
    row, col = data.edge_index
    # 无向化：把 (row,col) 和 (col,row) 都加入
    src = torch.cat([row, col])
    dst = torch.cat([col, row])
    x = data.x.float()

    sum_x = torch.zeros(N, D)
    sum_x.index_add_(0, src, x[src])
    cnt = torch.bincount(src, minlength=N).float().clamp(min=1).view(-1, 1)
    mean_x = sum_x / cnt

    max_x = torch.full((N, D), float('-inf'))
    max_x.scatter_reduce_(0, src.view(-1, 1).expand(-1, D), x[src], reduce='amax', include_self=False)
    max_x = torch.where(torch.isinf(max_x), torch.zeros_like(max_x), max_x)

    return torch.cat([mean_x, max_x], dim=1)


def build_features(data: Data, feature_name: str) -> Tuple[Data, dict]:
    """构造指定特征集，返回新的 Data 和元信息。

    feature_name:
      - raw: 原始 17 维（标准化）
      - structural: raw + 度/各 edge_type 度
      - temporal: raw + 时间统计
      - topology: raw + 中心性 + 社区 + node2vec 图嵌入
      - full: 最全特征集 = raw + degree + temporal + neighbor_agg
              + centrality + community + node2vec（包含所有模块）
      - important: 从 full(97维) 中按 s5 特征重要性选 top 25 维度（派生特征精选）
    """
    data = data.clone()
    raw = data.x.float()
    parts = [raw]
    composition = ['raw(17)']

    # 结构度特征
    if feature_name in ('structural', 'full', 'important'):
        deg_f = _degree_features(data)
        parts.append(deg_f)
        composition.append(f'degree({deg_f.size(1)})')
    # 时间特征
    if feature_name in ('temporal', 'full', 'important'):
        time_f = _temporal_features(data)
        parts.append(time_f)
        composition.append(f'temporal({time_f.size(1)})')
    # 邻居聚合特征（full + important 需要）
    if feature_name in ('full', 'important'):
        nb_f = _neighbor_agg_features(data)
        parts.append(nb_f)
        composition.append(f'neighbor_agg({nb_f.size(1)})')
    # 图拓扑特征（topology 单独 + full/important 全集）
    if feature_name in ('topology', 'full', 'important'):
        cen_f = _centrality_features(data, _ctx_data_source)
        parts.append(cen_f)
        composition.append(f'centrality({cen_f.size(1)})')
        comm_f = _community_features(data, _ctx_data_source)
        parts.append(comm_f)
        composition.append(f'community({comm_f.size(1)})')
        n2v_f = _node2vec_features(data, _ctx_data_source)
        parts.append(n2v_f)
        composition.append(f'node2vec({n2v_f.size(1)})')

    # important: 从 full 中按 s5 特征重要性选 top 25 维度
    if feature_name == 'important':
        full_x = torch.cat(parts, dim=1)
        dims = config.IMPORTANT_FEATURE_DIMS
        sel_x = full_x[:, dims]
        xs, norm_meta = _standardize(sel_x, data.train_mask)
        data.x = xs
        meta = {
            'name': 'important',
            'feature_dim': xs.size(1),
            'dim': xs.size(1),
            'composition': [f'important_top{len(dims)}(from full)'],
            'selected_dims': dims,
            'source': 'full',
            'normalization': norm_meta,
        }
        return data, meta

    x_new = torch.cat(parts, dim=1)
    x_new, norm_meta = _standardize(x_new, data.train_mask)
    data.x = x_new

    meta = {
        'feature_name': feature_name,
        'data_source': None,  # 由调用方填充
        'num_nodes': int(data.num_nodes),
        'feature_dim': int(x_new.size(1)),
        'composition': composition,
        'total_dim': int(x_new.size(1)),
        'normalization': norm_meta,
    }
    return data, meta


# ============================================================
# 图拓扑特征（中心性 / 社区 / 邻居标签 / node2vec）
# 参考实战1：PageRank / Betweenness / Closeness / Clustering / 社区发现
# ============================================================
# build_features 无法直接拿到 data_source，用模块级上下文变量传递（由 run_stage1 设置）
_ctx_data_source: str = 'sample'


def _cache_path(data_source: str, name: str):
    """图拓扑特征的缓存路径（计算昂贵，持久化避免重复）。"""
    return config.out_root(data_source) / 'features' / f'_{name}.pt'


def _centrality_features(data: Data, data_source: str) -> torch.Tensor:
    """中心性特征（4维）：PageRank / Betweenness / Closeness / Clustering。

    - PageRank：有向图上的资金/关联重要性（可扩展，大图也跑）；
    - Betweenness：近似（k=100 采样），大图(>10万节点)跳过；
    - Closeness / Clustering：无向图，大图跳过。

    业务含义：PageRank 高=重要资金汇聚点；Betweenness 高=中转桥接节点；
    Closeness 高=网络核心位置；Clustering 高=局部团伙闭环。
    """
    cp = _cache_path(data_source, 'centrality')
    if cp.exists():
        return torch.load(cp, map_location='cpu', weights_only=False)

    import networkx as nx
    N = data.num_nodes
    ei = data.edge_index.cpu().numpy()
    scalable_only = N > 100000  # 大图只跑可扩展项

    print(f'    [centrality] 构建 networkx 图 (N={N}, E={ei.shape[1]})...')
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))

    print('    [centrality] PageRank...')
    pr = nx.pagerank(G, alpha=0.85, tol=1e-6)
    pr_arr = np.array([pr.get(i, 0.0) for i in range(N)], dtype=np.float32)

    if scalable_only:
        print('    [centrality] 大图模式：跳过 Betweenness/Closeness/Clustering')
        bt_arr = np.zeros(N, dtype=np.float32)
        cl_arr = np.zeros(N, dtype=np.float32)
        cc_arr = np.zeros(N, dtype=np.float32)
    else:
        G_und = G.to_undirected()
        print(f'    [centrality] Betweenness (近似 k={min(100, N)})...')
        bt = nx.betweenness_centrality(G_und, k=min(100, N),
                                       seed=config.SEED, normalized=True)
        bt_arr = np.array([bt.get(i, 0.0) for i in range(N)], dtype=np.float32)
        print('    [centrality] Closeness...')
        cl = nx.closeness_centrality(G_und)
        cl_arr = np.array([cl.get(i, 0.0) for i in range(N)], dtype=np.float32)
        print('    [centrality] Clustering...')
        cc = nx.clustering(G_und)
        cc_arr = np.array([cc.get(i, 0.0) for i in range(N)], dtype=np.float32)

    feats = torch.stack([
        torch.from_numpy(pr_arr),
        torch.from_numpy(bt_arr),
        torch.from_numpy(cl_arr),
        torch.from_numpy(cc_arr),
    ], dim=1)  # (N, 4)
    torch.save(feats, cp)
    print(f'    [centrality] 完成 {feats.shape}，已缓存')
    return feats


def _community_features(data: Data, data_source: str) -> torch.Tensor:
    """社区特征（3维）：community_id / community_size / community_density。

    用 greedy_modularity 社区发现，刻画节点所属团伙的规模和紧密程度。
    大图(>10万节点)跳过（greedy_modularity 复杂度高）。
    """
    cp = _cache_path(data_source, 'community')
    if cp.exists():
        return torch.load(cp, map_location='cpu', weights_only=False)

    N = data.num_nodes
    if N > 100000:
        print('    [community] 大图模式：跳过社区发现')
        feats = torch.zeros(N, 3, dtype=torch.float32)
        torch.save(feats, cp)
        return feats

    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    ei = data.edge_index.cpu().numpy()
    G_und = nx.Graph()
    G_und.add_nodes_from(range(N))
    G_und.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))

    print(f'    [community] greedy_modularity 社区发现 (N={N})...')
    communities = list(greedy_modularity_communities(G_und))
    print(f'    [community] 发现 {len(communities)} 个社区')

    comm_id = np.zeros(N, dtype=np.float32)
    comm_size = np.zeros(N, dtype=np.float32)
    comm_density = np.zeros(N, dtype=np.float32)
    for cid, comm in enumerate(communities):
        members = list(comm)
        sub = G_und.subgraph(members)
        density = float(nx.density(sub)) if len(members) > 1 else 0.0
        for m in members:
            comm_id[m] = cid
            comm_size[m] = len(members)
            comm_density[m] = density

    feats = torch.stack([
        torch.from_numpy(comm_id),
        torch.from_numpy(comm_size),
        torch.from_numpy(comm_density),
    ], dim=1)
    torch.save(feats, cp)
    print(f'    [community] 完成 {feats.shape}，已缓存')
    return feats


def _neighbor_label_features(data: Data, data_source: str) -> torch.Tensor:
    """邻居标签特征（2维，泄露安全版）。

    仅用 train+val 标签计算，绝不用 test 标签：
    - labeled_neighbor_ratio: 邻居中已知标签(train+val)的比例；
    - known_illicit_neighbor_ratio: 邻居中已知异常(train+val, y=1)的比例。

    注意：此特征在真实上线时可能不可用（新用户无已知邻居标签），
    默认不放入 topology 特征集，单独保存供实验对比。
    """
    cp = _cache_path(data_source, 'neighbor_label')
    if cp.exists():
        return torch.load(cp, map_location='cpu', weights_only=False)

    N = data.num_nodes
    y = data.y.cpu().numpy()
    # 已知标签 = train 或 val 节点（test 标签绝不用）
    known_mask = (data.train_mask | data.val_mask).cpu().numpy()
    known_illicit = known_mask & (y == 1)

    ei = data.edge_index.cpu().numpy()
    # 无向化：每个节点的邻居 = 出现在 src 或 dst 的对端
    all_neighbors = np.concatenate([ei[0], ei[1]])
    nb_weights = np.ones(len(all_neighbors), dtype=np.float64)

    num_nb = np.bincount(all_neighbors, minlength=N).astype(np.float64)
    num_known = np.bincount(all_neighbors, minlength=N,
                            weights=known_mask[all_neighbors].astype(np.float64))
    num_illicit = np.bincount(all_neighbors, minlength=N,
                              weights=known_illicit[all_neighbors].astype(np.float64))

    labeled_ratio = (num_known / np.maximum(num_nb, 1)).astype(np.float32)
    illicit_ratio = (num_illicit / np.maximum(num_nb, 1)).astype(np.float32)

    feats = torch.stack([
        torch.from_numpy(labeled_ratio),
        torch.from_numpy(illicit_ratio),
    ], dim=1)
    torch.save(feats, cp)
    print(f'    [neighbor_label] 完成 {feats.shape}（仅用 train+val 标签，已缓存）')
    return feats


def _node2vec_features(data: Data, data_source: str,
                       embedding_dim: int = None, epochs: int = 100) -> torch.Tensor:
    """node2vec / DeepWalk 图嵌入特征，纯 PyTorch 实现。

    embedding_dim 默认从 config.NODE2VEC_DIM 读取（16维），保持计算效率。
    PyG Node2Vec 需要 pyg-lib（Windows 难装），这里用纯 PyTorch 实现：
    1. 向量化随机游走（不依赖 pyg-lib 的 C++ random_walk）；
    2. SkipGram + 负采样训练嵌入。

    捕捉图结构相似性：功能相似的节点嵌入相近。sample/full 均可扩展。
    """
    if embedding_dim is None:
        embedding_dim = config.NODE2VEC_DIM
    cp = _cache_path(data_source, 'node2vec')
    if cp.exists():
        return torch.load(cp, map_location='cpu', weights_only=False)

    device = config.get_device(data_source)
    N = data.num_nodes
    # 大图降级策略：减少游走数/长度/epoch，避免内存爆炸（370万节点）
    # - 默认: walks_per_node=10, walk_length=20, epochs=100
    # - 大图(>100k): walks_per_node=2, walk_length=10, epochs=30
    if N > 100000:
        walk_length = 10
        walks_per_node = 2
        epochs = min(epochs, 30)
        print(f'    [node2vec] 大图模式(N={N}>100k)：'
              f'walks_per_node=2, walk_length=10, epochs={epochs}')
    else:
        walk_length = 20
        walks_per_node = 10
    context_size = 5  # 窗口半宽
    num_neg = 5  # 每个正样本的负采样数

    # 构建 CSR 邻接表（按源节点排序），在 CPU 上构建（避免占用 GPU 显存）
    row, col = data.edge_index
    sort_idx = torch.argsort(row)
    row_s = row[sort_idx]
    col_s_cpu = col[sort_idx]
    counts = torch.bincount(row_s, minlength=N)
    ptr_cpu = torch.zeros(N + 1, dtype=torch.long)
    ptr_cpu[1:] = torch.cumsum(counts, dim=0)

    print(f'    [node2vec] 生成随机游走 (N={N}, walks/node={walks_per_node}, '
          f'len={walk_length})...')
    # 随机游走在 GPU 上生成（向量化计算快），完成后立即转回 CPU
    ptr = ptr_cpu.to(device)
    col_s = col_s_cpu.to(device)
    num_walks = N * walks_per_node
    walks = torch.zeros(num_walks, walk_length, dtype=torch.long, device=device)
    walks[:, 0] = torch.arange(N, device=device).repeat(walks_per_node)
    num_edges = col_s.size(0)
    for step in range(1, walk_length):
        cur = walks[:, step - 1]
        starts = ptr[cur]
        ends = ptr[cur + 1]
        has_nb = ends > starts
        degs = (ends - starts).clamp(min=1)
        rand_off = (torch.rand(num_walks, device=device) * degs.float()).long()
        safe_idx = (starts + rand_off).clamp(max=num_edges - 1)
        nxt = col_s[safe_idx]
        walks[:, step] = torch.where(has_nb, nxt, cur)
    # 释放 CSR 邻接表显存（训练阶段不需要）
    del ptr, col_s
    torch.cuda.empty_cache()

    # 构建 SkipGram 训练对（在 CPU 上，避免占用 GPU 显存）
    print('    [node2vec] 构建 SkipGram 训练对...')
    walks_cpu = walks.cpu()
    del walks
    torch.cuda.empty_cache()
    centers, contexts = [], []
    for offset in range(1, context_size + 1):
        centers.append(walks_cpu[:, :-offset].reshape(-1))
        contexts.append(walks_cpu[:, offset:].reshape(-1))
    centers = torch.cat(centers)  # (num_pairs,)
    contexts = torch.cat(contexts)
    del walks_cpu
    # 过滤掉 center==context 的无效对
    valid = centers != contexts
    centers = centers[valid]
    contexts = contexts[valid]
    num_pairs = centers.shape[0]
    print(f'    [node2vec] {num_pairs} 训练对，训练 SkipGram (dim={embedding_dim}, '
          f'epochs={epochs}, device={device})...')

    # SkipGram 嵌入层
    emb = torch.nn.Embedding(N, embedding_dim).to(device)
    torch.nn.init.xavier_uniform_(emb.weight)
    optimizer = torch.optim.Adam(emb.parameters(), lr=0.01)
    # 增大 batch_size 减少 kernel launch 开销
    batch_size = 20480
    # 训练对保留在 CPU，按 batch 传 GPU（减少常驻显存）
    log_every = max(1, epochs // 6)  # 打印 6 次日志
    sync_every = 50  # 每 50 steps 同步一次 loss，避免每 step 同步
    import time as _time
    t0 = _time.time()
    for epoch in range(epochs):
        perm = torch.randperm(num_pairs)  # CPU 上生成，避免 GPU 显存
        total_loss = 0.0
        sync_count = 0
        steps = 0
        for i in range(0, num_pairs, batch_size):
            idx = perm[i:i + batch_size]
            # 从 CPU 拷贝到 GPU（仅当前 batch）
            c = centers[idx].to(device, non_blocking=True)
            pos = contexts[idx].to(device, non_blocking=True)
            neg = torch.randint(0, N, (len(idx), num_neg), device=device)
            pos_score = (emb(c) * emb(pos)).sum(dim=1)
            pos_loss = -torch.nn.functional.logsigmoid(pos_score).mean()
            neg_score = torch.bmm(emb(c).unsqueeze(1),
                                  emb(neg).transpose(1, 2)).squeeze(1)
            neg_loss = -torch.nn.functional.logsigmoid(-neg_score).mean()
            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每 sync_every steps 才同步一次 loss（减少 CPU-GPU 同步开销）
            if (steps + 1) % sync_every == 0:
                total_loss += loss.item() * sync_every
                sync_count += 1
            steps += 1
        # 处理剩余未同步的 loss
        if sync_count == 0:
            total_loss = loss.item() * steps
        else:
            remainder = steps % sync_every
            if remainder > 0:
                total_loss += loss.item() * remainder
        avg_loss = total_loss / max(steps, 1)
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = _time.time() - t0
            print(f'    [node2vec] epoch {epoch+1}/{epochs} '
                  f'loss={avg_loss:.4f} elapsed={elapsed:.1f}s '
                  f'({elapsed/(epoch+1):.1f}s/epoch)')

    embeddings = emb.weight.detach().cpu()  # (N, embedding_dim)
    torch.save(embeddings, cp)
    print(f'    [node2vec] 完成 {embeddings.shape}，已缓存')
    return embeddings


# ============================================================
# 异构图转换 & 时间快照（供异构/时间感知模型使用）
# ============================================================
def to_heterogeneous_data(data: Data) -> 'HeteroData':
    """将同构 Data 转为 HeteroData，按 edge_type 拆分为多种关系。

    DGraphFin 节点统一为"user"类型，边按 edge_type 拆分为
    ("user", "type_{t}", "user") 多种关系。
    保留所有边类型信息，不丢失任何边。
    """
    from torch_geometric.data import HeteroData
    het = HeteroData()
    het['user'].x = data.x
    if hasattr(data, 'y'):
        het['user'].y = data.y
    for name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(data, name):
            het['user'][name] = getattr(data, name)

    et = getattr(data, 'edge_type', None)
    if et is None:
        # 无 edge_type，作为单一关系
        het['user', 'edge', 'user'].edge_index = data.edge_index
    else:
        num_types = int(et.max()) + 1
        for t in range(num_types):
            mask = (et == t)
            if mask.sum() == 0:
                continue
            ei_t = data.edge_index[:, mask]
            het['user', f'type_{t}', 'user'].edge_index = ei_t
    return het


def build_temporal_snapshots(data: Data, num_snapshots: int = 5) -> list:
    """按 edge_time 将图切分为时间快照序列，供时间感知模型使用。

    将边按时间排序后均匀分为 num_snapshots 个时间桶，
    每个快照包含截至当前时间桶的所有边（累积图）。
    返回 [Data_0, Data_1, ..., Data_{T-1}]，每个 Data 共享节点特征和标签，
    但 edge_index 不同（只含对应时间段的边）。
    """
    et = getattr(data, 'edge_time', None)
    if et is None:
        # 无时间信息，返回同一图的 T 份拷贝
        return [data for _ in range(num_snapshots)]

    et = et.cpu().long()
    N = data.num_nodes
    # 按时间排序，均匀分桶
    sorted_idx = torch.argsort(et)
    bucket_size = max(len(sorted_idx) // num_snapshots, 1)
    snapshots = []
    for t in range(num_snapshots):
        start = 0
        end = (t + 1) * bucket_size if t < num_snapshots - 1 else len(sorted_idx)
        # 累积：包含所有时间 <= 当前桶的边
        edge_indices = sorted_idx[:end]
        snap = data.clone()
        snap.edge_index = data.edge_index[:, edge_indices]
        if hasattr(data, 'edge_type'):
            snap.edge_type = data.edge_type[edge_indices]
        snap.edge_time = et[edge_indices]
        snapshots.append(snap)
    return snapshots


# ============================================================
# 深度图分析：连通分量 / Top-K 中心性 / 社区审计 / 时间窗口
# ============================================================
def deep_graph_analysis(data: Data, data_source: str) -> dict:
    """深度图分析：4 个子分析，结果统一保存到 graph_analysis.json。

    - 连通分量分析（WCC）：识别团伙边界
    - Top-K 中心性异常率：高中心性节点是否更易异常
    - 全图社区逐社区审计：定位高风险社区
    - 时间窗口标签分布：异常爆发时段
    """
    res_dir = config.out_root(data_source) / 'results'
    result = {'data_source': data_source, 'num_nodes': int(data.num_nodes)}

    # 分析1：连通分量
    try:
        print('  [深度图分析] 1/4 连通分量分析...')
        result['connected_components'] = _analyze_connected_components(data, data_source)
    except Exception as e:
        print(f'  [深度图分析] 连通分量分析失败: {e}')
        result['connected_components'] = {'error': str(e)}

    # 分析2：Top-K 中心性异常率
    try:
        print('  [深度图分析] 2/4 Top-K 中心性异常率...')
        result['topk_centrality'] = _analyze_topk_centrality(data, data_source)
    except Exception as e:
        print(f'  [深度图分析] Top-K 中心性分析失败: {e}')
        result['topk_centrality'] = {'error': str(e)}

    # 分析3：社区审计
    try:
        print('  [深度图分析] 3/4 社区审计...')
        result['communities'] = _analyze_communities(data, data_source)
    except Exception as e:
        print(f'  [深度图分析] 社区审计失败: {e}')
        result['communities'] = {'error': str(e)}

    # 分析4：时间窗口标签分布
    try:
        print('  [深度图分析] 4/4 时间窗口标签分布...')
        result['temporal_labels'] = _analyze_temporal_labels(data, data_source)
    except Exception as e:
        print(f'  [深度图分析] 时间窗口分析失败: {e}')
        result['temporal_labels'] = {'error': str(e)}

    save_path = res_dir / 'graph_analysis.json'
    utils.save_json(result, save_path)
    print(f'  [深度图分析] 完成，结果保存到 {save_path}')
    return result


def _analyze_connected_components(data: Data, data_source: str) -> dict:
    """分析1：弱连通分量（WCC）分析。

    sample 用 networkx，full 用 scipy.sparse.csgraph（高效并查集）。
    识别图的连通结构，定位包含最多异常节点的团伙。
    """
    N = data.num_nodes
    y = data.y.cpu().numpy()
    ei = data.edge_index.cpu().numpy()
    res_dir = config.out_root(data_source) / 'results'

    if N <= 100000:
        # sample：networkx 无向图
        import networkx as nx
        print(f'    [WCC] 构建 networkx 无向图 (N={N})...')
        G_und = nx.Graph()
        G_und.add_nodes_from(range(N))
        G_und.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        components = list(nx.connected_components(G_und))
        num_comp = len(components)
        comp_labels = np.zeros(N, dtype=np.int64)
        for cid, comp in enumerate(components):
            arr = np.fromiter(comp, dtype=np.int64)
            comp_labels[arr] = cid
        print(f'    [WCC] 共 {num_comp} 个弱连通分量')
    else:
        # full：scipy.sparse.csgraph 高效并查集
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components as sp_cc
        print(f'    [WCC] 构建 CSR 邻接矩阵 (N={N})...')
        adj = csr_matrix((np.ones(len(ei[0]), dtype=np.int8), (ei[0], ei[1])),
                         shape=(N, N))
        print('    [WCC] 计算弱连通分量...')
        num_comp, comp_labels = sp_cc(csgraph=adj, directed=False,
                                      connection='weak', return_labels=True)
        comp_labels = comp_labels.astype(np.int64)
        print(f'    [WCC] 共 {num_comp} 个弱连通分量')

    # 分量规模统计
    comp_sizes = np.bincount(comp_labels, minlength=num_comp)
    largest_comp_size = int(comp_sizes.max())
    largest_comp_ratio = float(largest_comp_size / max(N, 1))

    # 向量化统计每个分量的标签分布（argsort 分组，避免逐分量 np.where）
    order = np.argsort(comp_labels, kind='stable')
    sorted_y = y[order]
    sorted_labels = comp_labels[order]
    boundaries = np.searchsorted(sorted_labels, np.arange(num_comp + 1))

    comp_stats = []
    for cid in range(num_comp):
        s, e = int(boundaries[cid]), int(boundaries[cid + 1])
        if s == e:
            continue
        y_mem = sorted_y[s:e]
        num_labeled = int(((y_mem == 0) | (y_mem == 1)).sum())
        num_illicit = int((y_mem == 1).sum())
        comp_stats.append({
            'component_id': int(cid),
            'size': int(e - s),
            'num_labeled': num_labeled,
            'num_illicit': num_illicit,
            'illicit_rate': float(num_illicit / max(num_labeled, 1)),
        })

    # 按 num_illicit 降序取 Top-10
    comp_stats.sort(key=lambda x: x['num_illicit'], reverse=True)
    top10 = comp_stats[:10]

    # 画图：分量规模分布双对数图
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes_sorted = np.sort(comp_sizes)[::-1]
    ranks = np.arange(1, len(sizes_sorted) + 1)
    ax.loglog(ranks, sizes_sorted, 'o-', markersize=4, alpha=0.7,
              color='steelblue', label='分量规模')
    for i in range(min(10, len(sizes_sorted))):
        ax.annotate(f'#{i+1}\nsize={int(sizes_sorted[i])}',
                    (ranks[i], sizes_sorted[i]),
                    textcoords='offset points', xytext=(8, -8),
                    fontsize=7, color='crimson')
    ax.set_xlabel('分量排名 (log)')
    ax.set_ylabel('分量规模 (log)')
    ax.set_title(f'{data_source} 弱连通分量规模分布 (共{num_comp}个)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(res_dir / 'wcc_size_distribution.png', dpi=120)
    plt.close()

    return {
        'num_components': int(num_comp),
        'largest_component_size': largest_comp_size,
        'largest_component_ratio': largest_comp_ratio,
        'top10_components': top10,
    }


def _analyze_topk_centrality(data: Data, data_source: str) -> dict:
    """分析2：Top-K 中心性异常率。

    基于 PageRank / Betweenness 取 Top-50，对比异常率与全局基线。
    中心性特征从缓存加载（_centrality_features 已有缓存机制）。
    """
    y = data.y.cpu().numpy()
    labeled_mask = (y == 0) | (y == 1)
    global_illicit_rate = float((y == 1).sum() / max(int(labeled_mask.sum()), 1))
    res_dir = config.out_root(data_source) / 'results'

    # 中心性特征（从缓存加载或计算）
    cen = _centrality_features(data, data_source).cpu().numpy()
    pagerank = cen[:, 0]
    betweenness = cen[:, 1]
    K = 50

    def topk_stats(score):
        topk_idx = np.argsort(-score)[:K]
        y_topk = y[topk_idx]
        num_labeled = int(((y_topk == 0) | (y_topk == 1)).sum())
        num_illicit = int((y_topk == 1).sum())
        return {
            'illicit_rate': float(num_illicit / max(num_labeled, 1)),
            'num_labeled': num_labeled,
            'num_illicit': num_illicit,
        }

    pr_stats = topk_stats(pagerank)
    # betweenness 在 full 上全为 0（大图跳过），检测后跳过
    bt_all_zero = bool(np.all(betweenness == 0))
    if bt_all_zero:
        bt_stats = {
            'illicit_rate': 0.0, 'num_labeled': 0, 'num_illicit': 0,
            'skipped': '大图未计算 betweenness',
        }
    else:
        bt_stats = topk_stats(betweenness)

    # 画图：全局 vs Top-50 PageRank vs Top-50 Betweenness
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['全局', 'Top-50\nPageRank']
    values = [global_illicit_rate, pr_stats['illicit_rate']]
    colors = ['steelblue', 'crimson']
    if not bt_all_zero:
        labels.append('Top-50\nBetweenness')
        values.append(bt_stats['illicit_rate'])
        colors.append('darkorange')
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    ax.set_ylabel('异常率 (illicit_rate)')
    ax.set_title(f'{data_source} Top-K 中心性节点异常率对比')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f'{v:.4f}',
                ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(res_dir / 'topk_centrality_illicit_rate.png', dpi=120)
    plt.close()

    return {
        'global_illicit_rate': global_illicit_rate,
        'pagerank_top50': pr_stats,
        'betweenness_top50': bt_stats,
    }


def _analyze_communities(data: Data, data_source: str) -> dict:
    """分析3：全图社区逐社区审计。

    sample：用 _community_features 获取全图社区，逐社区审计。
    full：greedy_modularity 不可行，从最大连通分量采样 2 万节点子图做社区发现。
    """
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    N = data.num_nodes
    y = data.y.cpu().numpy()
    ei = data.edge_index.cpu().numpy()
    res_dir = config.out_root(data_source) / 'results'

    if N <= 100000:
        # sample：全图社区（从缓存加载）
        comm_feat = _community_features(data, data_source).cpu().numpy()
        comm_ids = comm_feat[:, 0].astype(np.int64)
        comm_density_per_node = comm_feat[:, 2]
        num_communities = int(comm_ids.max() + 1) if len(comm_ids) > 0 else 0
        note = 'sample为全图社区'

        order = np.argsort(comm_ids, kind='stable')
        sorted_y = y[order]
        sorted_ids = comm_ids[order]
        boundaries = np.searchsorted(sorted_ids, np.arange(num_communities + 1))

        comm_stats = []
        for cid in range(num_communities):
            s, e = int(boundaries[cid]), int(boundaries[cid + 1])
            if s == e:
                continue
            y_mem = sorted_y[s:e]
            num_labeled = int(((y_mem == 0) | (y_mem == 1)).sum())
            num_illicit = int((y_mem == 1).sum())
            size = int(e - s)
            density = float(comm_density_per_node[order[s]])
            # order[s:e] 给出该社区全部节点在原始 data 中的 id
            member_ids = order[s:e].tolist()
            comm_stats.append({
                'community_id': int(cid),
                'size': size,
                'num_labeled': num_labeled,
                'num_illicit': num_illicit,
                'illicit_rate': float(num_illicit / max(num_labeled, 1)),
                'density': density,
                'node_ids': member_ids,
            })
    else:
        # full：从最大连通分量 BFS 采样 2 万节点子图（保留局部连通性）
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components as sp_cc
        from collections import deque
        print('    [社区] 构建 CSR 邻接矩阵定位最大连通分量...')
        adj = csr_matrix((np.ones(len(ei[0]), dtype=np.int8), (ei[0], ei[1])),
                         shape=(N, N))
        num_comp, comp_labels = sp_cc(csgraph=adj, directed=False,
                                      connection='weak', return_labels=True)
        comp_labels = comp_labels.astype(np.int64)
        comp_sizes = np.bincount(comp_labels, minlength=num_comp)
        largest_comp_id = int(comp_sizes.argmax())
        largest_members = np.where(comp_labels == largest_comp_id)[0]
        print(f'    [社区] 最大连通分量: {len(largest_members)} 节点')

        # 无向化邻接矩阵用于 BFS
        adj_undirected = (adj + adj.T).tocsr()

        rng = np.random.RandomState(config.SEED)
        sample_size = min(20000, len(largest_members))

        # 选 BFS 种子：优先从最大连通分量中的异常节点(y=1)出发
        largest_set_mask = np.zeros(N, dtype=bool)
        largest_set_mask[largest_members] = True
        illicit_seeds = np.where(largest_set_mask & (y == 1))[0]
        if len(illicit_seeds) > 0:
            seed = int(illicit_seeds[rng.randint(len(illicit_seeds))])
        else:
            seed = int(largest_members[rng.randint(len(largest_members))])

        # BFS 采样：从种子出发沿边扩展，保留局部连通性
        visited = np.zeros(N, dtype=bool)
        visited[seed] = True
        queue = deque([seed])
        num_visited = 1
        while queue and num_visited < sample_size:
            cur = queue.popleft()
            neighbors = adj_undirected.indices[
                adj_undirected.indptr[cur]:adj_undirected.indptr[cur + 1]]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(int(nb))
                    num_visited += 1
                    if num_visited >= sample_size:
                        break
        sampled = np.where(visited)[0]
        print(f'    [社区] BFS 采样: {len(sampled)} 节点 (种子={seed}, y={int(y[seed])})')

        # 用布尔 mask 快速筛选子图边
        in_sample = np.zeros(N, dtype=bool)
        in_sample[sampled] = True
        s_mask = in_sample[ei[0]] & in_sample[ei[1]]

        G_sub = nx.Graph()
        G_sub.add_nodes_from(sampled.tolist())
        G_sub.add_edges_from(zip(ei[0][s_mask].tolist(), ei[1][s_mask].tolist()))
        print(f'    [社区] 采样子图: {G_sub.number_of_nodes()} 节点, '
              f'{G_sub.number_of_edges()} 边')

        communities = list(greedy_modularity_communities(G_sub))
        num_communities = len(communities)
        print(f'    [社区] 发现 {num_communities} 个社区')

        comm_stats = []
        for cid, comm in enumerate(communities):
            members = np.fromiter(comm, dtype=np.int64)
            y_mem = y[members]
            num_labeled = int(((y_mem == 0) | (y_mem == 1)).sum())
            num_illicit = int((y_mem == 1).sum())
            size = int(len(members))
            density = float(nx.density(G_sub.subgraph(list(comm)))) if size > 1 else 0.0
            comm_stats.append({
                'community_id': int(cid),
                'size': size,
                'num_labeled': num_labeled,
                'num_illicit': num_illicit,
                'illicit_rate': float(num_illicit / max(num_labeled, 1)),
                'density': density,
                'node_ids': members.tolist(),
            })
        note = f'full为采样子图({sample_size}节点，来自最大连通分量)'

    # 排序：(illicit_rate 降序, size 降序)
    comm_stats.sort(key=lambda x: (x['illicit_rate'], x['size']), reverse=True)
    top15 = comm_stats[:15]

    # 导出 top5 社区完整节点列表（用于后续可视化/分析）
    top5 = comm_stats[:5]
    top5_export = {
        'data_source': data_source,
        'sort_key': '(illicit_rate 降序, size 降序)',
        'note': note,
        'num_communities': int(num_communities),
        'top5_communities': [
            {
                'rank': i + 1,
                'community_id': c['community_id'],
                'size': c['size'],
                'num_labeled': c['num_labeled'],
                'num_illicit': c['num_illicit'],
                'illicit_rate': c['illicit_rate'],
                'density': c['density'],
                'node_ids': c['node_ids'],
            } for i, c in enumerate(top5)
        ],
    }
    top5_path = res_dir / 'top5_communities.json'
    utils.save_json(top5_export, top5_path)
    print(f'  [top5_communities] 已导出: {top5_path} '
          f'({len(top5_export["top5_communities"])} 社区)')

    # 画图：Top-15 社区 size 和 illicit_rate 双轴图
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_labels = [f'C{c["community_id"]}' for c in top15]
    sizes = [c['size'] for c in top15]
    rates = [c['illicit_rate'] for c in top15]
    x_pos = np.arange(len(top15))
    ax1.bar(x_pos, sizes, color='steelblue', alpha=0.7, label='社区规模')
    ax1.set_xlabel('社区 (按异常率降序)')
    ax1.set_ylabel('规模', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')

    ax2 = ax1.twinx()
    ax2.plot(x_pos, rates, 'o-', color='crimson', label='异常率',
             linewidth=2, markersize=6)
    ax2.set_ylabel('异常率', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    plt.title(f'{data_source} Top-15 社区审计 (规模 & 异常率)')
    fig.tight_layout()
    plt.savefig(res_dir / 'community_audit.png', dpi=120)
    plt.close()

    return {
        'num_communities': int(num_communities),
        'top15': top15,
        'note': note,
    }


def _analyze_temporal_labels(data: Data, data_source: str) -> dict:
    """分析4：时间窗口标签分布。

    把边时间映射到节点（取关联边 edge_time 的 min 作为首次出现时间），
    按时间步分桶统计标签 0/1/2/3 分布，定位异常爆发时段。
    """
    et = getattr(data, 'edge_time', None)
    if et is None:
        return {'error': '数据无 edge_time 字段'}
    N = data.num_nodes
    y = data.y.cpu().numpy()
    res_dir = config.out_root(data_source) / 'results'

    # 每个节点的首次出现时间 = 关联边 edge_time 的 min（作为源或目的）
    et_t = et.cpu().float()
    row, col = data.edge_index
    node_time = torch.full((N,), float('inf'), dtype=torch.float)
    node_time.scatter_reduce_(0, row, et_t, reduce='amin', include_self=False)
    node_time.scatter_reduce_(0, col, et_t, reduce='amin', include_self=False)
    node_time_np = node_time.cpu().numpy()
    valid = node_time_np != float('inf')
    if not valid.any():
        return {'error': '无有效节点时间'}
    node_time_np[~valid] = -1
    node_time_np = node_time_np.astype(np.int64)

    t_min = int(node_time_np[valid].min())
    t_max = int(node_time_np[valid].max())

    # 分桶：每 20 个时间步一桶
    bucket_size = 20
    num_buckets = (t_max - t_min) // bucket_size + 1

    buckets = []
    for b in range(num_buckets):
        bs = t_min + b * bucket_size
        be = bs + bucket_size - 1
        if b == num_buckets - 1:
            be = t_max
        mask = (node_time_np >= bs) & (node_time_np <= be)
        y_b = y[mask]
        buckets.append({
            'bucket_start': int(bs),
            'bucket_end': int(be),
            'label_0': int((y_b == 0).sum()),
            'label_1': int((y_b == 1).sum()),
            'label_2': int((y_b == 2).sum()),
            'label_3': int((y_b == 3).sum()),
        })

    # 找异常率最高的桶（基于已标注节点 y=0/1）
    peak_bucket = None
    peak_rate = -1.0
    for b in buckets:
        labeled = b['label_0'] + b['label_1']
        if labeled > 0:
            rate = b['label_1'] / labeled
            if rate > peak_rate:
                peak_rate = rate
                peak_bucket = b

    # 画图：堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(buckets))
    l0 = np.array([b['label_0'] for b in buckets])
    l1 = np.array([b['label_1'] for b in buckets])
    l2 = np.array([b['label_2'] for b in buckets])
    l3 = np.array([b['label_3'] for b in buckets])
    ax.bar(x, l0, label='y=0 正常', color='steelblue')
    ax.bar(x, l1, bottom=l0, label='y=1 异常', color='crimson')
    ax.bar(x, l2, bottom=l0 + l1, label='y=2 背景', color='lightgray')
    ax.bar(x, l3, bottom=l0 + l1 + l2, label='y=3 背景', color='darkgray')
    ax.set_xlabel('时间桶 (每20时间步)')
    ax.set_ylabel('节点数')
    ax.set_title(f'{data_source} 时间窗口标签分布堆叠图')
    step = max(1, len(x) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f'{b["bucket_start"]}' for b in buckets][::step],
                       rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(res_dir / 'temporal_label_distribution.png', dpi=120)
    plt.close()

    return {
        'time_buckets': buckets,
        'peak_illicit_bucket': {
            'bucket_start': peak_bucket['bucket_start'],
            'bucket_end': peak_bucket['bucket_end'],
            'illicit_rate': float(peak_rate),
        } if peak_bucket is not None else None,
    }


def run_stage1(data_source: str, feature_names=None) -> Tuple[Data, Dict[str, Data]]:
    """执行阶段一：加载 + 任务A统计 + 构造并保存所有特征集。"""
    utils.set_seed(config.SEED)
    print(f'\n[S1] 加载数据: {data_source}')
    data = load_data(data_source)
    print(f'  num_nodes={data.num_nodes} num_edges={data.edge_index.size(1)} '
          f'x={tuple(data.x.shape)}')

    # 任务A
    print('[S1] 任务A 数据理解与统计...')
    task_a_analysis(data, data_source)

    # 深度图分析（连通分量/中心性TopK/社区审计/时间窗口）
    try:
        deep_graph_analysis(data, data_source)
    except Exception as e:
        print(f'  [深度图分析] 失败（不影响主流程）: {e}')

    # 特征工程
    feature_names = feature_names or config.FEATURE_NAMES
    global _ctx_data_source
    _ctx_data_source = data_source  # 供 build_features 内的图拓扑特征函数使用
    feature_datasets: Dict[str, Data] = {}
    for fname in feature_names:
        d_feat, meta = build_features(data, fname)
        meta['data_source'] = data_source
        utils.save_features(d_feat.x, meta, data_source, fname)
        feature_datasets[fname] = d_feat
        print(f'  特征 {fname}: dim={meta.get("feature_dim", meta.get("dim", "?"))} composition={meta["composition"]} 已保存')

    # 邻居标签特征（泄露安全版，单独保存，默认不放入 topology）
    # 用户要求"提取但可以不用"，这里提取持久化供实验对比
    try:
        _neighbor_label_features(data, data_source)
    except Exception as e:
        print(f'  [neighbor_label] 提取失败（不影响主流程）: {e}')

    # 保存原始 data（含图结构）供后续阶段复用，避免重复加载
    torch.save(data.cpu(), config.out_root(data_source) / 'features' / '_base_data.pt')
    return data, feature_datasets


if __name__ == '__main__':
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else 'sample'
    run_stage1(ds)
