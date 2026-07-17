# -*- coding: utf-8 -*-
"""LightGBM 包装器：非图模型（任务B），从特征重要性工具扩展为完整可训练模型。

与 XGBoost 类似的接口，但使用 LightGBM 的直方图算法（leaf-wise 生长），
训练更快且内存更优。同时保留作为特征重要性分析工具的能力。

接口对齐：
- fit(data) 用 train_mask 训练；
- predict_proba_lgb(x) 返回所有节点属于正类（y=1）的概率张量。
"""
from __future__ import annotations

import numpy as np
import torch


class LightGBMModel:
    """LightGBM 二分类包装器。非 nn.Module，但提供兼容的 fit / 预测接口。"""

    def __init__(self, n_estimators: int = 400, max_depth: int = -1,
                 learning_rate: float = 0.1, num_leaves: int = 31,
                 n_jobs: int = -1, scale_pos_weight: float = 1.0,
                 seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth      # -1 表示无限制
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves    # LightGBM 核心参数
        self.n_jobs = n_jobs
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.model = None

    def fit(self, data) -> 'LightGBMModel':
        import lightgbm as lgb

        x = data.x.detach().cpu().numpy().astype(np.float32)
        y = data.y.detach().cpu().numpy().astype(np.int32)
        train_mask = data.train_mask.detach().cpu().numpy().astype(bool)

        x_train = x[train_mask]
        y_train = y[train_mask]

        # 处理类别不平衡
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = self.scale_pos_weight if self.scale_pos_weight != 1.0 else (neg / max(pos, 1))

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            objective='binary',
            metric='average_precision',
            boosting_type='gbdt',
            n_jobs=self.n_jobs,
            scale_pos_weight=spw,
            random_state=self.seed,
            verbosity=-1,
        )
        self.model.fit(x_train, y_train)
        return self

    def predict_proba_lgb(self, x: torch.Tensor) -> torch.Tensor:
        """返回所有节点属于正类（y=1）的概率，形状 [N]。"""
        arr = x.detach().cpu().numpy().astype(np.float32)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(arr)
            proba_pos = proba[:, 1] if proba.ndim == 2 else proba
        else:
            # Booster 直接 predict 返回概率
            proba_pos = self.model.predict(arr)
        proba_pos = np.asarray(proba_pos).astype(np.float32).reshape(-1)
        return torch.from_numpy(proba_pos).to(x.device)

    # ---------- 兼容 nn.Module 调用风格的接口 ----------
    def __call__(self, data, return_embedding: bool = False):
        prob = self.predict_proba_lgb(data.x)
        prob = prob.clamp(min=1e-6, max=1 - 1e-6)
        logits = torch.stack([torch.log(1 - prob), torch.log(prob)], dim=1)
        if return_embedding:
            return logits, None
        return logits

    def eval(self):
        return self

    def state_dict(self) -> dict:
        """保存为 JSON 字符串。"""
        if self.model is None:
            return {}
        booster = self.model.booster_ if hasattr(self.model, 'booster_') else self.model
        return {'model_json': booster.model_to_string()}

    def load_state_dict(self, state: dict) -> 'LightGBMModel':
        import lightgbm as lgb
        if not state or 'model_json' not in state:
            return self
        booster = lgb.Booster(model_str=state['model_json'])
        self.model = booster
        return self

    def to(self, device):
        return self
