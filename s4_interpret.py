# -*- coding: utf-8 -*-
"""阶段四：可解释性分析（进阶挑战）。

职责：
1. t-SNE 可视化 GNN 最后一层 embedding，观察异常点是否从正常点中分离；
2. GNNExplainer 对 Top-1 高风险用户做节点级解释（重要特征 / 重要边）；
3. k_hop 子图绘制：直观展示高风险用户的局部图结构证据；
4. 社区发现：在高风险子图上检测紧密团伙。

这些内容贴合"风控审核清单"业务场景，把模型判断翻译成图结构证据。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config
import utils
from utils import plt
from models.xgboost_model import XGBoostModel


class _ExplainerModel(nn.Module):
    """GNNExplainer 适配包装类。

    PyG Explainer 调用 `model(x, edge_index)` 传张量，但项目里的 GNN
    forward 期望 Data 对象（forward(data, return_embedding=False)）。
    这里把 (x, edge_index) 重新打包成 Data 再调用原始 GNN。
    """

    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn

    def forward(self, x, edge_index):
        from torch_geometric.data import Data
        return self.gnn(Data(x=x, edge_index=edge_index))


def _get_embedding(model, data, device):
    """获取 GNN 倒数第二层 embedding；XGBoost 无 embedding 返回 None。"""
    if isinstance(model, XGBoostModel):
        return None
    model = model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        try:
            _, emb = model(data, return_embedding=True)
        except Exception:
            return None
    return emb.detach().cpu()


# ============================================================
# 1. t-SNE 可视化 embedding
# ============================================================
def plot_tsne(model, data, data_source: str, model_name: str, feature_name: str,
              device: torch.device, sample_size: int = 3000) -> Optional[str]:
    """对 GNN embedding 做 t-SNE 降维，按标签着色。"""
    emb = _get_embedding(model, data, device)
    if emb is None:
        print(f'  [t-SNE] {model_name} 无 embedding，跳过')
        return None

    from sklearn.manifold import TSNE

    # 在二分类节点上采样，避免全量 t-SNE 过慢
    binary = (data.y == 0) | (data.y == 1)
    idx = binary.nonzero(as_tuple=False).view(-1).cpu().numpy()
    if len(idx) > sample_size:
        rng = np.random.default_rng(config.SEED)
        idx = rng.choice(idx, sample_size, replace=False)
    emb_np = emb[idx].numpy()
    labels = data.y[idx].cpu().numpy()

    print(f'  [t-SNE] {model_name}/{feature_name} 降维 {emb_np.shape[0]} 个节点...')
    tsne = TSNE(n_components=2, random_state=config.SEED, init='pca',
                learning_rate='auto', perplexity=min(30, len(idx) - 1))
    coord = tsne.fit_transform(emb_np)

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, color, name in [(0, 'steelblue', '正常 y=0'), (1, 'crimson', '异常 y=1')]:
        m = labels == lab
        ax.scatter(coord[m, 0], coord[m, 1], s=8, alpha=0.5, c=color, label=name)
    ax.set_title(f'{model_name}/{feature_name} GNN embedding t-SNE')
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out = config.out_root(data_source) / 'interpret' / f'tsne_{model_name}_{feature_name}.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f'  [t-SNE] 已保存 {out.name}')
    return str(out)


# ============================================================
# 2. GNNExplainer 节点级解释
# ============================================================
def _node_indegree(edge_index: torch.Tensor, node_idx: int, num_nodes: int) -> int:
    """统计 node_idx 作为 target（入边）的次数。

    SAGEConv 默认 flow='source_to_target'：消息从 source 流向 target，
    target 节点聚合入边邻居。所以只有入度>0 的节点，其预测才依赖边。
    """
    ei = edge_index if edge_index.device.type == 'cpu' else edge_index.cpu()
    return int(((ei[1] == node_idx) & (ei[0] != node_idx)).sum().item())


def explain_node(model, data, node_idx: int, data_source: str, model_name: str,
                 feature_name: str, device: torch.device,
                 epochs: int = 300) -> Optional[dict]:
    """用 GNNExplainer 解释单个节点的预测（重要特征 + 重要边）。

    关键修复（针对 DGraphFin 有向图 + PyG 2.8.0）：
    1. return_type='raw'：模型返回 raw logits（不是 log_softmax），用 'log_probs' 会导致
       F.nll_loss 在 raw logits 上语义错误；
    2. 入度=0 的节点（无入边）跳过 edge_mask：SAGEConv 对这类节点的预测只依赖自身特征，
       edge_mask 对其预测无梯度，PyG 正则项会在空 mask 上 ent.mean()=nan 导致训练崩溃；
    3. node_mask 形状是 (N, F)，取目标节点行 nm[node_idx] 才是该节点的重要特征；
    4. epochs 提高到 300 让 mask 充分收敛。
    """
    if isinstance(model, XGBoostModel):
        print('  [GNNExplainer] XGBoost 不支持，跳过')
        return None
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
    except Exception as e:
        print(f'  [GNNExplainer] 无法导入 PyG explain 模块: {e}')
        return None

    model = model.to(device)
    data = data.to(device)
    model.eval()

    # 诊断：模型对目标节点的原始预测
    with torch.no_grad():
        wrapper_diag = _ExplainerModel(model).to(device)
        logits_all = wrapper_diag(data.x, data.edge_index)
        logit_node = logits_all[node_idx]
        probs_node = logit_node.softmax(dim=-1)
        pred_class = int(probs_node.argmax().item())
        print(f'  [GNNExplainer] 节点 {node_idx} 预测: probs={probs_node.cpu().numpy().tolist()}, '
              f'pred_class={pred_class}')

    # 检查入度：入度=0 的节点跳过 edge_mask 避免 nan
    indegree = _node_indegree(data.edge_index, node_idx, data.num_nodes)
    use_edge_mask = indegree > 0
    if not use_edge_mask:
        print(f'  [GNNExplainer] 节点 {node_idx} 入度=0（纯源节点），'
              f'预测只依赖自身特征，禁用 edge_mask 避免 nan')
    edge_mask_type = 'object' if use_edge_mask else None

    # 用包装类把 (x, edge_index) 调用适配到 forward(data) 的 GNN
    wrapper = _ExplainerModel(model).to(device)
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type=edge_mask_type,
        model_config=dict(mode='multiclass_classification', task_level='node',
                          return_type='raw'),
    )
    try:
        explanation = explainer(data.x, data.edge_index, index=node_idx)
    except Exception as e:
        print(f'  [GNNExplainer] 解释节点 {node_idx} 失败: {e}')
        return None

    result = {
        'node_idx': int(node_idx), 'model': model_name, 'feature': feature_name,
        'pred_class': pred_class, 'indegree': indegree,
        'edge_mask_enabled': use_edge_mask,
    }

    # 重要特征（top-10）：node_mask 形状 (N, F)，取目标节点行
    if explanation.node_mask is not None:
        nm = explanation.node_mask.detach().cpu()  # (N, F) 或 (1, F)
        result['node_mask_shape'] = list(nm.shape)
        if nm.dim() == 2 and nm.size(0) > 1:
            # (N, F)：取目标节点行
            target_nm = nm[node_idx]
            result['node_mask_mode'] = 'target_node_row'
        elif nm.dim() == 2:
            target_nm = nm[0]
            result['node_mask_mode'] = 'single_row'
        else:
            target_nm = nm.view(-1)
            result['node_mask_mode'] = 'flat'
        nm_np = target_nm.numpy()
        top_feat = np.argsort(-nm_np)[:10]
        result['top_features'] = [{'dim': int(i), 'score': float(nm_np[i])}
                                  for i in top_feat]
        result['node_mask_stats'] = {
            'min': float(nm_np.min()), 'max': float(nm_np.max()),
            'mean': float(nm_np.mean()), 'std': float(nm_np.std()),
        }

    # 重要边（top-10）：edge_mask 长度 = 全图边数
    if explanation.edge_mask is not None:
        em = explanation.edge_mask.detach().cpu().numpy()
        result['edge_mask_shape'] = list(em.shape)
        result['edge_mask_stats'] = {
            'min': float(em.min()), 'max': float(em.max()),
            'mean': float(em.mean()), 'std': float(em.std()),
            'num_nonzero': int((em > 1e-6).sum()), 'total_edges': int(len(em)),
        }
        top_edges_idx = np.argsort(-em)[:10]
        ei = data.edge_index.cpu().numpy()
        result['top_edges'] = [
            {'src': int(ei[0, i]), 'dst': int(ei[1, i]), 'score': float(em[i])}
            for i in top_edges_idx
        ]

    out = config.out_root(data_source) / 'interpret' / f'explainer_{model_name}_{feature_name}_node{node_idx}.json'
    utils.save_json(result, out)
    # 诊断输出
    nm_stats = result.get('node_mask_stats', {})
    em_stats = result.get('edge_mask_stats', {})
    print(f'  [GNNExplainer] node_mask shape={result.get("node_mask_shape")} '
          f'max={nm_stats.get("max", 0):.4f} std={nm_stats.get("std", 0):.6f}')
    if em_stats:
        print(f'  [GNNExplainer] edge_mask shape={result.get("edge_mask_shape")} '
              f'max={em_stats["max"]:.4f} nnz={em_stats["num_nonzero"]}/{em_stats["total_edges"]}')
    print(f'  [GNNExplainer] 节点 {node_idx} 解释已保存: {out.name}')
    return result


# ============================================================
# 3. k_hop 子图绘制
# ============================================================
def plot_subgraph(data, node_idx: int, data_source: str, model_name: str,
                  feature_name: str, hops: int = 2, top_k_extra: int = 15,
                  y_score: Optional[np.ndarray] = None) -> Optional[str]:
    """绘制中心节点 + k_hop 邻域（+ 可选 Top-K 相似高风险节点）的诱导子图。"""
    import networkx as nx
    from torch_geometric.utils import k_hop_subgraph

    # 防御性归位：子图操作全部在 CPU 上做，避免 edge_index/y 设备不一致
    data = data.cpu()

    subset, edge_index_sub, _, _ = k_hop_subgraph(
        node_idx, hops, data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
    ei = edge_index_sub.cpu().numpy()

    G = nx.Graph()
    nodes = subset.cpu().numpy()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(zip(ei[0], ei[1]))

    # 节点颜色：中心=金，异常=红，正常=蓝，背景=灰
    y = data.y.cpu().numpy()
    colors = []
    for n in nodes:
        if n == node_idx:
            colors.append('gold')
        elif y[n] == 1:
            colors.append('crimson')
        elif y[n] == 0:
            colors.append('steelblue')
        else:
            colors.append('lightgray')
    sizes = [300 if n == node_idx else 60 for n in nodes]

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=config.SEED)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.4, ax=ax)
    ax.set_title(f'{model_name}/{feature_name} 节点 {node_idx} 的 {hops}-hop 子图\n'
                 f'(金=中心 红=异常 蓝=正常 灰=背景)')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out = config.out_root(data_source) / 'interpret' / f'subgraph_{model_name}_{feature_name}_node{node_idx}.png'
    plt.savefig(out, dpi=120)
    plt.close()
    print(f'  [子图] 已保存 {out.name}')
    return str(out)


# ============================================================
# 4. 高风险社区发现
# ============================================================
def detect_high_risk_communities(data, y_score: np.ndarray, data_source: str,
                                 model_name: str, feature_name: str,
                                 top_k: int = 200) -> Optional[dict]:
    """取 Top-K 高风险节点做诱导子图，跑社区发现，检测紧密团伙。"""
    import networkx as nx
    from torch_geometric.utils import subgraph

    # 防御性归位：subgraph 要求节点 index 与 edge_index 在同一设备
    data = data.cpu()

    test_mask = data.test_mask.cpu().numpy()
    node_ids = test_mask.nonzero()[0]
    if len(node_ids) == 0:
        return None
    order = np.argsort(-y_score)[:min(top_k, len(y_score))]
    top_nodes = node_ids[order]

    sub_ei, _ = subgraph(torch.from_numpy(top_nodes).long(), data.edge_index,
                         relabel_nodes=True, num_nodes=data.num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(len(top_nodes)))
    ei = sub_ei.cpu().numpy()
    G.add_edges_from(zip(ei[0], ei[1]))

    if G.number_of_edges() == 0:
        print('  [社区发现] Top-K 高风险节点之间无边连接，无法做社区发现')
        return {'num_top_nodes': int(len(top_nodes)), 'num_edges': 0, 'communities': []}

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
    except Exception as e:
        print(f'  [社区发现] 失败: {e}')
        return None

    y = data.y.cpu().numpy()
    comm_info = []
    for cid, comm in enumerate(communities):
        members = top_nodes[list(comm)]
        labels = y[members]
        fraud = int((labels == 1).sum())
        comm_info.append({
            'community_id': cid, 'size': int(len(comm)),
            'fraud_hits': fraud, 'fraud_rate': float(fraud / max(len(comm), 1)),
        })
    comm_info.sort(key=lambda x: x['fraud_hits'], reverse=True)

    result = {
        'num_top_nodes': int(len(top_nodes)),
        'num_edges': int(G.number_of_edges()),
        'num_communities': len(communities),
        'communities': comm_info[:10],
        'top_fraud_community': comm_info[0] if comm_info else None,
    }
    out = config.out_root(data_source) / 'interpret' / f'community_{model_name}_{feature_name}.json'
    utils.save_json(result, out)
    print(f'  [社区发现] Top-{len(top_nodes)} 高风险节点形成 {len(communities)} 个社区，'
          f'最大团伙命中 {comm_info[0]["fraud_hits"]} 个异常（已保存 {out.name}）')

    # 绘制社区图
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, seed=config.SEED)
    cmap = plt.cm.tab20
    node_colors = [cmap(cid % 20) for cid in range(len(communities))
                   for _ in communities[cid]] if communities else []
    # 重新对齐到节点顺序
    color_map = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            color_map[n] = cmap(cid % 20)
    nc = [color_map.get(n, 'lightgray') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=40, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, ax=ax)
    ax.set_title(f'{model_name}/{feature_name} Top-{len(top_nodes)} 高风险节点社区发现')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(config.out_root(data_source) / 'interpret' / f'community_{model_name}_{feature_name}.png', dpi=120)
    plt.close()
    return result


# ============================================================
# 编排：对指定模型做全套解释
# ============================================================
def _pick_explainable_node(node_ids: np.ndarray, y_score: np.ndarray,
                           edge_index: torch.Tensor, num_nodes: int,
                           top_k: int = 30) -> int:
    """在 Top-K 高风险节点中选第一个入度>0 的节点。

    DGraphFin 是有向图，SAGEConv flow='source_to_target'：只有入度>0 的节点
    其预测才依赖边，GNNExplainer 才能学到有意义的 edge_mask。纯源节点（入度=0）
    的预测只依赖自身特征，edge_mask 会被禁用（但仍可解释 node_mask）。
    """
    order = np.argsort(-y_score)[:min(top_k, len(y_score))]
    ei_cpu = edge_index if edge_index.device.type == 'cpu' else edge_index.cpu()
    for i in order:
        nid = int(node_ids[i])
        indeg = int(((ei_cpu[1] == nid) & (ei_cpu[0] != nid)).sum().item())
        if indeg > 0:
            return nid
    # 退路：所有 Top-K 都是纯源节点，返回 Top-1（explain_node 会禁用 edge_mask）
    return int(node_ids[int(np.argmax(y_score))])


def run_stage4(model, data, data_source: str, model_name: str, feature_name: str,
               device: torch.device = None):
    """对单个模型执行全套可解释性分析。"""
    if device is None:
        device = torch.device('cpu')
    print(f'\n[S4] 可解释性分析: {model_name}/{feature_name}')

    # 1) t-SNE
    plot_tsne(model, data, data_source, model_name, feature_name, device)

    # 在 Top-K 高风险节点中选第一个有入度的（让 edge_mask 能学到东西）
    node_ids, y_true, y_score = utils.predict_on_mask(model, data, data.test_mask, device)
    top1_pos = _pick_explainable_node(node_ids, y_score, data.edge_index, data.num_nodes)
    top1_rank = int(np.where(np.argsort(-y_score) == np.argsort(-y_score)[0])[0][0]) if len(y_score) > 0 else 0
    print(f'  [S4] 选择解释目标: 节点 {top1_pos} (Top-K 高风险中首个有入度的)')

    # 2) GNNExplainer
    explain_node(model, data, top1_pos, data_source, model_name, feature_name, device)

    # 3) k_hop 子图
    plot_subgraph(data, top1_pos, data_source, model_name, feature_name, hops=2)

    # 4) 社区发现
    detect_high_risk_communities(data, y_score, data_source, model_name, feature_name)


if __name__ == '__main__':
    print('s4_interpret 需在 main.py 编排下调用（需已训练模型）。')
