# -*- coding: utf-8 -*-
"""阶段二：模型训练与评估函数（模型构建、训练、评估）。

职责：
- class_weights：处理正负样本极度不平衡；
- train_model：统一训练入口（nn.Module 走梯度循环，XGBoost 走 fit）；
- evaluate / predict：AUC / AP / Recall@K / Precision@K 评估。

训练逻辑：
- CrossEntropyLoss + 类别权重；
- 按验证集 Average Precision 选最佳模型（best val AP）；
- 每个 epoch 记录 train_loss / val_loss / val 指标，用于观察过拟合。
"""
from __future__ import annotations

import gc
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import config
import utils
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel


# ============================================================
# 类别权重（缓解正负样本极度不平衡）
# ============================================================
def class_weights(data) -> torch.Tensor:
    y = utils._get_node_attr(data, 'y')
    train_mask = utils._get_node_attr(data, 'train_mask')
    y_train = y[train_mask]
    counts = torch.bincount(y_train, minlength=2).float()
    return counts.sum() / (2.0 * counts.clamp(min=1.0))


@torch.no_grad()
def evaluate(model, data, mask: torch.Tensor, device: torch.device,
             ks=config.TOPK_LIST) -> Tuple[dict, np.ndarray, np.ndarray]:
    """在指定 mask 上评估，返回 (metrics, y_true, y_score)。

    大图优化：如果 data 不在 device 上，临时拷贝评估后释放，避免显存累积。
    """
    if hasattr(model, 'eval'):
        model.eval()
    # 判断 data 是否已在目标设备上
    x_ref = utils._get_node_attr(data, 'x')
    data_on_device = x_ref.is_cuda if hasattr(x_ref, 'is_cuda') else False
    if not data_on_device:
        data = data.to(device)
        mask = mask.to(device)
    y = utils._get_node_attr(data, 'y')
    if hasattr(model, 'predict_proba_xgb'):  # XGBoost
        x = utils._get_node_attr(data, 'x')
        score = model.predict_proba_xgb(x).detach().cpu().numpy()
    elif hasattr(model, 'predict_proba_lgb'):  # LightGBM
        x = utils._get_node_attr(data, 'x')
        score = model.predict_proba_lgb(x).detach().cpu().numpy()
    else:
        try:
            logits = model(data)
        except TypeError:
            x = utils._get_node_attr(data, 'x')
            ei = utils._get_node_attr(data, 'edge_index')
            logits = model(x, ei)
        score = logits.softmax(dim=-1)[:, 1].detach().cpu().numpy()
    true = y[mask].detach().cpu().numpy()
    mask_cpu = mask.detach().cpu()
    score_masked = score[mask_cpu.numpy()] if score.shape[0] == y.shape[0] else score
    metrics = utils.compute_metrics(true, score_masked, ks=ks)
    return metrics, true, score_masked


@torch.no_grad()
def evaluate_loss(model, data, mask: torch.Tensor, weights: torch.Tensor) -> float:
    if hasattr(model, 'eval'):
        model.eval()
    if isinstance(model, (XGBoostModel, LightGBMModel)):
        return float('nan')
    logits = model(data)
    y = utils._get_node_attr(data, 'y')
    return float(F.cross_entropy(logits[mask], y[mask], weight=weights).item())


# ============================================================
# 训练：nn.Module（含早停机制）
# ============================================================
def _train_nn(model: nn.Module, data, model_name: str, feature_name: str,
              data_source: str, device: torch.device,
              epochs: int) -> Tuple[nn.Module, pd.DataFrame, dict]:
    # 大图优化：data 拷贝到 GPU 训练，函数结束前释放
    data_gpu = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    weights = class_weights(data_gpu).to(device)

    best_state = None
    best_val_ap = -1.0
    best_val_metrics: dict = {}
    history = []
    patience_counter = 0  # 早停计数器

    pbar = tqdm(range(1, epochs + 1), desc=f'训练 {model_name}/{feature_name}', leave=False)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        logits = model(data_gpu)
        train_mask = utils._get_node_attr(data_gpu, 'train_mask')
        y = utils._get_node_attr(data_gpu, 'y')
        val_mask = utils._get_node_attr(data_gpu, 'val_mask')
        loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=weights)
        loss.backward()
        optimizer.step()

        val_loss = evaluate_loss(model, data_gpu, val_mask, weights)
        record = {'epoch': epoch, 'train_loss': float(loss.item()), 'val_loss': val_loss}

        if epoch == 1 or epoch % config.EVAL_EVERY == 0 or epoch == epochs:
            val_metrics, _, _ = evaluate(model, data_gpu, val_mask, device)
            record.update({f'val_{k}': v for k, v in val_metrics.items()})
            pbar.set_postfix(train_loss=f'{loss.item():.4f}', val_loss=f'{val_loss:.4f}',
                             val_ap=f'{val_metrics["average_precision"]:.4f}',
                             val_auc=f'{val_metrics["roc_auc"]:.4f}')
            cur_ap = val_metrics['average_precision']
            if not np.isnan(cur_ap) and cur_ap > best_val_ap:
                best_val_ap = cur_ap
                best_val_metrics = dict(val_metrics)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0  # 有提升，重置计数器
            else:
                patience_counter += 1
        history.append(record)

        # 早停：验证集 AP 连续 patience 轮不提升
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f'  [早停] epoch {epoch}: 验证集 AP 连续 {config.EARLY_STOP_PATIENCE} 轮未提升，提前终止')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.history = pd.DataFrame(history)
    # 大图优化：释放 GPU 显存
    del data_gpu, weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, model.history, {**best_val_metrics, 'best_val_ap': best_val_ap}


# ============================================================
# 训练：XGBoost / LightGBM（树模型，非 epoch 训练）
# ============================================================
def _train_xgb(model: XGBoostModel, data, model_name: str, feature_name: str,
               data_source: str, device: torch.device) -> Tuple[XGBoostModel, pd.DataFrame, dict]:
    print(f'  [XGBoost] fit on train nodes... ({data.train_mask.sum().item()} 样本)')
    model.fit(data.cpu() if hasattr(data, 'cpu') else data)
    val_metrics, _, _ = evaluate(model, data.cpu(), data.val_mask, torch.device('cpu'))
    history = pd.DataFrame([{
        'epoch': 1,
        'train_loss': float('nan'),
        'val_loss': float('nan'),
        **{f'val_{k}': v for k, v in val_metrics.items()},
    }])
    model.history = history
    return model, history, {**val_metrics, 'best_val_ap': float(val_metrics.get('average_precision', 0.0))}


def _train_lgb(model: LightGBMModel, data, model_name: str, feature_name: str,
               data_source: str, device: torch.device) -> Tuple[LightGBMModel, pd.DataFrame, dict]:
    print(f'  [LightGBM] fit on train nodes... ({data.train_mask.sum().item()} 样本)')
    model.fit(data.cpu() if hasattr(data, 'cpu') else data)
    val_metrics, _, _ = evaluate(model, data.cpu(), data.val_mask, torch.device('cpu'))
    history = pd.DataFrame([{
        'epoch': 1,
        'train_loss': float('nan'),
        'val_loss': float('nan'),
        **{f'val_{k}': v for k, v in val_metrics.items()},
    }])
    model.history = history
    return model, history, {**val_metrics, 'best_val_ap': float(val_metrics.get('average_precision', 0.0))}


# ============================================================
# 统一训练入口
# ============================================================
def train_model(model, data, model_name: str, feature_name: str,
                data_source: str, device: torch.device = None) -> Tuple[object, pd.DataFrame, dict]:
    """训练并持久化模型与训练历史。

    返回 (model, history_df, best_val_metrics)。
    best_val_metrics 含 'average_precision'，供 s3 融合确定权重。
    支持：nn.Module（MLP/GCN/GraphSAGE/GAT/HeteroSAGE/RGCN/EvolveGCN/TCN）
         + XGBoost + LightGBM 包装器。

    大图优化：GPU OOM 时自动回退到 CPU 训练。
    """
    if device is None:
        device = config.get_device(data_source)
    utils.set_seed(config.SEED)

    if isinstance(model, XGBoostModel):
        model, history, best_val = _train_xgb(model, data, model_name, feature_name, data_source, device)
        utils.save_model(model.state_dict(), data_source, model_name, feature_name)
    elif isinstance(model, LightGBMModel):
        model, history, best_val = _train_lgb(model, data, model_name, feature_name, data_source, device)
        utils.save_model(model.state_dict(), data_source, model_name, feature_name)
    else:
        epochs = config.MLP_EPOCHS if model_name.lower() == 'mlp' else config.EPOCHS
        try:
            model, history, best_val = _train_nn(model, data, model_name, feature_name,
                                                 data_source, device, epochs)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and device.type == 'cuda':
                # GPU OOM，回退到 CPU
                print(f'  [OOM 回退] {model_name}/{feature_name} GPU 显存不足，切换到 CPU')
                # 尝试清理 GPU 显存（可能也会失败，忽略错误）
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                gc.collect()
                cpu_device = torch.device('cpu')
                # 重建模型（确保在 CPU 上）
                from models import build_model
                # 临时强制 CPU 创建模型
                _orig_device = config.get_device
                config.get_device = lambda ds: cpu_device
                try:
                    model = build_model(
                        model_name, data,
                        hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                        dropout=config.DROPOUT, heads=config.GAT_HEADS,
                        xgb_rounds=config.XGB_ROUNDS, lgb_rounds=config.LGB_ROUNDS,
                    )
                finally:
                    config.get_device = _orig_device
                model, history, best_val = _train_nn(model, data, model_name, feature_name,
                                                     data_source, cpu_device, epochs)
            else:
                raise
        model = model.cpu()
        utils.save_model(model.state_dict(), data_source, model_name, feature_name)
        # 大图优化：强制 GC + 释放 GPU 显存
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

    # 保存训练历史（用于绘制 loss / 指标曲线）
    utils.save_history(history, data_source, model_name, feature_name)

    # 保存模型配置（便于复现 / 网页展示）
    is_tree = isinstance(model, (XGBoostModel, LightGBMModel))
    cfg = {
        'model_name': model_name, 'feature_name': feature_name, 'data_source': data_source,
        'hidden_dim': config.HIDDEN_DIM, 'num_layers': config.NUM_LAYERS,
        'dropout': config.DROPOUT, 'gat_heads': config.GAT_HEADS,
        'lr': config.LR, 'weight_decay': config.WEIGHT_DECAY,
        'epochs': 1 if is_tree else (config.MLP_EPOCHS if model_name.lower() == 'mlp' else config.EPOCHS),
        'early_stop_patience': config.EARLY_STOP_PATIENCE,
        'xgb_rounds': config.XGB_ROUNDS, 'lgb_rounds': config.LGB_ROUNDS,
        'seed': config.SEED,
    }
    utils.save_json(cfg, config.model_config_path(data_source, model_name, feature_name))

    print(f'  [{model_name}/{feature_name}] best val AP={best_val.get("average_precision", float("nan")):.4f} '
          f'val AUC={best_val.get("roc_auc", float("nan")):.4f} | 模型与历史已保存')
    return model, history, best_val


if __name__ == '__main__':
    # 自测：在 sample 上训练 MLP/raw
    import sys
    from models import build_model
    ds = sys.argv[1] if len(sys.argv) > 1 else 'sample'
    import s1_data_features as s1
    _, feat_sets = s1.run_stage1(ds, ['raw'])
    d = feat_sets['raw']
    m = build_model('mlp', d)
    train_model(m, d, 'mlp', 'raw', ds)
