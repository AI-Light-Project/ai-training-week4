# -*- coding: utf-8 -*-
"""XGBoost 包装器：非图对照组（任务B）。

只使用节点特征矩阵（不使用图结构）。当特征工程注入结构统计特征后，
XGBoost 能直接利用这些结构信息，是验证"结构特征是否有用"的利器。

为对齐 GNN 训练/评估流程，包装成与 nn.Module 类似接口：
- fit(data) 用 train_mask 训练；
- predict_proba_xgb(x) 返回所有节点属于正类（y=1）的概率张量。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class XGBoostModel:
    """XGBoost 二分类包装器。非 nn.Module，但提供兼容的 fit / 预测接口。"""

    def __init__(self, n_estimators: int = 400, max_depth: int = 6,
                 learning_rate: float = 0.1, n_jobs: int = -1,
                 scale_pos_weight: float = 1.0, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.model = None  # 真正的 xgboost 模型，在 fit 中创建

    def fit(self, data) -> 'XGBoostModel':
        import xgboost as xgb

        x = data.x.detach().cpu().numpy().astype(np.float32)
        y = data.y.detach().cpu().numpy().astype(np.int32)
        train_mask = data.train_mask.detach().cpu().numpy().astype(bool)

        x_train = x[train_mask]
        y_train = y[train_mask]

        # 处理类别不平衡：正样本（异常）极少，用 scale_pos_weight 加权
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = self.scale_pos_weight if self.scale_pos_weight != 1.0 else (neg / max(pos, 1))

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='binary:logistic',
            eval_metric='aucpr',
            tree_method='hist',
            n_jobs=self.n_jobs,
            scale_pos_weight=spw,
            random_state=self.seed,
            verbosity=0,
        )
        self.model.fit(x_train, y_train)
        return self

    def predict_proba_xgb(self, x: torch.Tensor) -> torch.Tensor:
        """返回所有节点属于正类（y=1）的概率，形状 [N]。

        统一用 Booster + DMatrix 预测：训练后 self.model 是 XGBClassifier
        （有 get_booster）；从磁盘加载后 self.model 是 Booster，二者都兼容。
        binary:logistic 目标下 booster.predict 直接返回 P(y=1)。
        """
        import xgboost as xgb
        arr = x.detach().cpu().numpy().astype(np.float32)
        booster = self.model.get_booster() if hasattr(self.model, 'get_booster') else self.model
        proba_pos = booster.predict(xgb.DMatrix(arr))  # P(y=1)
        proba_pos = np.asarray(proba_pos).astype(np.float32).reshape(-1)
        return torch.from_numpy(proba_pos).to(x.device)

    # ---------- 兼容 nn.Module 调用风格的接口 ----------
    def __call__(self, data, return_embedding: bool = False):
        # 训练阶段（s2）会调用 model(data) 取 logits 算 loss；
        # XGBoost 不走梯度训练循环，这里返回概率转成的伪 logits 以备不时之需。
        prob = self.predict_proba_xgb(data.x)
        prob = prob.clamp(min=1e-6, max=1 - 1e-6)
        logits = torch.stack([torch.log(1 - prob), torch.log(prob)], dim=1)
        if return_embedding:
            return logits, None
        return logits

    def eval(self):
        return self

    def state_dict(self) -> dict:
        """保存为 JSON（XGBoost 原生模型序列化）。"""
        if self.model is None:
            return {}
        return {'model_json': self.model.get_booster().save_raw(raw_format='json').decode('utf-8')}

    def load_state_dict(self, state: dict) -> 'XGBoostModel':
        import xgboost as xgb
        if not state or 'model_json' not in state:
            return self
        # 直接用 Booster 加载，避免 XGBClassifier 缺少 n_classes_ 等属性的问题
        booster = xgb.Booster()
        booster.load_model(bytearray(state['model_json'], 'utf-8'))
        self.model = booster
        return self

    def to(self, device):
        # XGBoost 在 CPU 上跑，device 仅用于对齐接口
        return self
