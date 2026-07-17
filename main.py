# -*- coding: utf-8 -*-
"""DGraphFin 大作业主流程编排。

用法：
    # 在 sample 上跑全流程（默认）
    .venv/Scripts/python.exe main.py

    # 指定数据集 / 阶段 / 特征 / 模型
    .venv/Scripts/python.exe main.py --data sample --stages 1 2 3 4
    .venv/Scripts/python.exe main.py --data full --stages 1 2 3
    .venv/Scripts/python.exe main.py --features raw structural --models mlp graphsage

阶段：
  1 = 数据理解 + 特征工程（任务A + 特征持久化）
  2 = 模型训练（任务B：MLP/LightGBM/XGBoost/GCN/GraphSAGE/GAT/HeteroSAGE/RGCN/EvolveGCN/TCN）
  3 = 测试评估 + 融合 + 汇总
  4 = 可解释性（进阶：t-SNE / GNNExplainer / 子图 / 社区发现）
  5 = 特征重要性分析（XGBoost+LightGBM 多方法，指导特征筛选）
  6 = ROC 曲线可视化 + TPR/FPR 数据保存
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import torch

import config
import utils
from models import build_model
import s1_data_features as s1
import s2_train_eval as s2
import s3_eval_fusion as s3
import s4_interpret as s4
import s5_feature_importance as s5
import s6_visualization as s6


def run_stage1(data_source, feature_names):
    print('\n' + '=' * 60)
    print(f'阶段一：数据理解与特征提取  [dataset={data_source}]')
    print('=' * 60)
    base_data, feat_sets = s1.run_stage1(data_source, feature_names)
    return base_data, feat_sets


def run_stage2(data_source, feature_names, model_names, feat_sets, device):
    import gc as _gc
    print('\n' + '=' * 60)
    print(f'阶段二：模型训练  [dataset={data_source}]')
    print('=' * 60)
    trained = {}  # (feature, model) -> model
    val_metrics = {}  # (feature, model) -> best_val

    # 按模型类型预处理数据
    hetero_data_cache = {}  # (feature) -> HeteroData
    temporal_data_cache = {}  # (feature) -> Data with snapshots

    for fname in feature_names:
        data = feat_sets[fname]
        for mname in model_names:
            # 跳过已训练的模型（支持断点续跑）
            model_file = config.model_path(data_source, mname, fname)
            if model_file.exists():
                print(f'  [跳过] {mname}/{fname} 已训练，从磁盘加载')
                try:
                    model = build_model(
                        mname, data,
                        hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                        dropout=config.DROPOUT, heads=config.GAT_HEADS,
                        xgb_rounds=config.XGB_ROUNDS, lgb_rounds=config.LGB_ROUNDS,
                    )
                    model.load_state_dict(torch.load(model_file, map_location='cpu'))
                    model.eval()
                    trained[(fname, mname)] = model
                except Exception as e:
                    print(f'  [加载失败] {mname}/{fname}: {e}，将重新训练')
                else:
                    continue

            print(f'\n--- 训练 {mname} / 特征 {fname} ---')
            try:
                # 异构图模型：需要 HeteroData
                if mname.lower() in config.HETERO_MODEL_NAMES:
                    if mname.lower() == 'heterosage':
                        if fname not in hetero_data_cache:
                            hetero_data_cache[fname] = s1.to_heterogeneous_data(data)
                        train_data = hetero_data_cache[fname]
                    else:  # rgcn 用同构 Data + edge_type
                        train_data = data
                # 时间感知模型：需要时间快照
                elif mname.lower() in config.TEMPORAL_MODEL_NAMES:
                    if fname not in temporal_data_cache:
                        snapshots = s1.build_temporal_snapshots(data, num_snapshots=5)
                        # 把快照列表挂到 data 上
                        tdata = data.clone()
                        tdata.snapshots = snapshots
                        tdata.num_snapshots = 5
                        temporal_data_cache[fname] = tdata
                    train_data = temporal_data_cache[fname]
                else:
                    train_data = data

                # 大图优化：复杂模型（GAT/HeteroSAGE/RGCN/EvolveGCN/TCN）在 full 数据集上用 CPU
                # 避免显存不足导致 OOM
                model_device = device
                if (data_source == 'full' and
                    mname.lower() in ('gat', 'heterosage', 'rgcn', 'evolvegcn', 'tcn')):
                    model_device = torch.device('cpu')
                    print(f'  [大图策略] {mname} 使用 CPU 训练（避免 GPU OOM）')

                model = build_model(
                    mname, train_data,
                    hidden_dim=config.HIDDEN_DIM, num_layers=config.NUM_LAYERS,
                    dropout=config.DROPOUT, heads=config.GAT_HEADS,
                    xgb_rounds=config.XGB_ROUNDS, lgb_rounds=config.LGB_ROUNDS,
                )
                model, _, best_val = s2.train_model(
                    model, train_data, mname, fname, data_source, model_device)
                trained[(fname, mname)] = model
                val_metrics[(fname, mname)] = best_val
            except Exception as e:
                print(f'  [跳过] {mname}/{fname} 训练失败: {e}')
                traceback.print_exc()
            finally:
                # 大图优化：每个模型训练后强制 GC + 释放 GPU 显存
                _gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
    return trained, val_metrics


def run_stage3(data_source, feature_names, model_names, feat_sets, trained, device):
    import gc as _gc
    print('\n' + '=' * 60)
    print(f'阶段三：测试评估 + 融合 + 汇总  [dataset={data_source}]')
    print('=' * 60)

    hetero_data_cache = {}
    temporal_data_cache = {}

    for fname in feature_names:
        data = feat_sets[fname]
        for mname in model_names:
            # 跳过已评估的模型（支持断点续跑）
            metrics_file = config.metrics_path(data_source, mname, fname)
            if metrics_file.exists():
                # 已评估：补画 score-distribution（如果还没画过）
                dist_path = (config.out_root(data_source) / 'results' / 'score_distributions'
                             / f'{mname}_{fname}_score_dist.png')
                if not dist_path.exists():
                    try:
                        plot_path = s3.plot_model_score_distribution(data_source, mname, fname)
                        if plot_path:
                            print(f'    [score-dist] {plot_path}')
                    except Exception as e:
                        print(f'    [score-dist 失败] {mname}/{fname}: {e}')
                else:
                    print(f'  [跳过] {mname}/{fname} 已评估')
                continue

            model = trained.get((fname, mname))
            if model is None:
                print(f'  [跳过] {mname}/{fname} 未训练')
                continue
            # 为不同模型类型准备正确的数据格式
            if mname.lower() == 'heterosage':
                if fname not in hetero_data_cache:
                    hetero_data_cache[fname] = s1.to_heterogeneous_data(data)
                eval_data = hetero_data_cache[fname]
            elif mname.lower() in config.TEMPORAL_MODEL_NAMES:
                if fname not in temporal_data_cache:
                    snapshots = s1.build_temporal_snapshots(data, num_snapshots=5)
                    tdata = data.clone()
                    tdata.snapshots = snapshots
                    tdata.num_snapshots = 5
                    temporal_data_cache[fname] = tdata
                eval_data = temporal_data_cache[fname]
            else:
                eval_data = data
            try:
                # 大图优化：复杂模型评估也用 CPU
                eval_device = device
                if (data_source == 'full' and
                    mname.lower() in ('gat', 'heterosage', 'rgcn', 'evolvegcn', 'tcn')):
                    eval_device = torch.device('cpu')
                s3.evaluate_on_test(model, eval_data, mname, fname, data_source, eval_device)
                # 评估后绘制该模型的风险分数分布（参考 baseline notebook # 4）
                try:
                    plot_path = s3.plot_model_score_distribution(data_source, mname, fname)
                    if plot_path:
                        print(f'    [score-dist] {plot_path}')
                except Exception as e:
                    print(f'    [score-dist 失败] {mname}/{fname}: {e}')
            except Exception as e:
                print(f'  [评估失败] {mname}/{fname}: {e}')
            finally:
                # 大图优化：评估后强制 GC + 释放 GPU 显存
                _gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass

        # 融合（基于 predictions 文件存在性判断，避免模型加载失败导致融合被跳过）
        available = [m for m in model_names
                     if config.predictions_path(data_source, m, fname).exists()]
        for combo_name, combo_models in config.FUSION_COMBOS.items():
            if all(m in available for m in combo_models):
                for method in config.FUSION_METHODS:
                    # 断点续跑：跳过已融合的 method
                    fmp = config.fusion_metrics_path(data_source, combo_name, fname, method)
                    if fmp.exists():
                        print(f'  [跳过] 融合 {combo_name}/{method}/{fname} 已完成')
                        continue
                    try:
                        s3.fuse_models(data_source, combo_name, combo_models,
                                       fname, method=method)
                    except Exception as e:
                        print(f'  [融合失败] {combo_name}/{method}/{fname}: {e}')
                    finally:
                        _gc.collect()
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except RuntimeError:
                                pass

    s3.build_summary(data_source)


def run_stage4(data_source, feature_names, feat_sets, trained, device):
    print('\n' + '=' * 60)
    print(f'阶段四：可解释性分析  [dataset={data_source}]')
    print('=' * 60)
    # 选测试 AP 最高的 GNN 模型做解释
    best = None
    best_ap = -1
    for fname in feature_names:
        for mname in ['graphsage', 'gat']:
            mp = config.metrics_path(data_source, mname, fname)
            if not mp.exists():
                continue
            try:
                m = utils.load_json(mp)
                ap = float(m.get('average_precision', -1))
                if ap > best_ap and not np.isnan(ap):
                    best_ap = ap
                    best = (fname, mname)
            except Exception:
                pass
    if best is None:
        print('  未找到可解释的 GNN 模型，跳过阶段四')
        return
    fname, mname = best
    print(f'  选择最佳 GNN 模型做解释: {mname}/{fname} (test AP={best_ap:.4f})')
    model = trained.get((fname, mname))
    data = feat_sets[fname]
    if model is None:
        print('  模型不在内存，尝试从磁盘加载...')
        in_dim = data.x.size(-1)
        model = s3.load_trained_model(mname, fname, data_source, in_dim)
    s4.run_stage4(model, data, data_source, mname, fname, device)


def main():
    parser = argparse.ArgumentParser(description='DGraphFin 大作业主流程')
    parser.add_argument('--data', default='sample', choices=['sample', 'full'],
                        help='数据集：sample(5万) 或 full(370万)，默认 sample')
    parser.add_argument('--stages', nargs='+', default=['1', '2', '3', '4'],
                        help='要执行的阶段，默认 1 2 3 4')
    parser.add_argument('--features', nargs='+', default=None,
                        help='特征集，默认全部 raw/structural/temporal/full')
    parser.add_argument('--models', nargs='+', default=None,
                        help='模型，默认全部 mlp/lightgbm/xgboost/gcn/graphsage/gat/heterosage/rgcn/evolvegcn/tcn')
    args = parser.parse_args()

    data_source = args.data
    stages = [int(s) for s in args.stages]
    feature_names = args.features or config.FEATURE_NAMES
    model_names = args.models or config.ALL_MODEL_NAMES
    device = config.get_device(data_source)

    print(f'数据集: {data_source} | 设备: {device}')
    print(f'阶段: {stages} | 特征: {feature_names} | 模型: {model_names}')

    utils.set_seed(config.SEED)

    feat_sets = {}
    trained = {}
    val_metrics = {}

    if 1 in stages:
        _, feat_sets = run_stage1(data_source, feature_names)
    else:
        # 跳过阶段1时，从磁盘加载已保存的特征
        print('\n[跳过阶段1] 从磁盘加载特征...')
        import torch as _t
        base = _t.load(config.out_root(data_source) / 'features' / '_base_data.pt',
                       map_location='cpu', weights_only=False)
        for fname in feature_names:
            x = utils.load_features(data_source, fname)
            d = base.clone()
            d.x = x
            feat_sets[fname] = d

    if 2 in stages:
        trained, val_metrics = run_stage2(
            data_source, feature_names, model_names, feat_sets, device)

    if 3 in stages:
        if not trained:
            print('\n[阶段3] 模型未在内存，从磁盘加载...')
            for fname in feature_names:
                d = feat_sets[fname]
                for mname in model_names:
                    mp = config.model_path(data_source, mname, fname)
                    if mp.exists():
                        try:
                            trained[(fname, mname)] = s3.load_trained_model(
                                mname, fname, data_source, d.x.size(-1))
                        except Exception as e:
                            print(f'  加载 {mname}/{fname} 失败: {e}')
        run_stage3(data_source, feature_names, model_names, feat_sets, trained, device)

    if 4 in stages:
        run_stage4(data_source, feature_names, feat_sets, trained, device)

    if 5 in stages:
        s5.run_stage5(data_source, feature_names)

    if 6 in stages:
        s6.run_stage6(data_source, feature_names)

    print('\n' + '=' * 60)
    print('全部流程完成。输出目录：', config.out_root(data_source))
    print('=' * 60)


if __name__ == '__main__':
    main()
