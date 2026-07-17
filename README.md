# 0717-project

**0717-project** 是一个基于图神经网络的金融风控欺诈检测系统。

该项目旨在通过融合多种图神经网络模型与传统机器学习方法，构建一套高效、准确的欺诈检测解决方案，帮助金融机构识别异常交易行为并降低信用风险。

---

## 目录

- [功能一览](#功能一览)
- [特征工程模块](#特征工程模块)
- [模型训练模块](#模型训练模块)
- [模型融合模块](#模型融合模块)
- [可解释性分析模块](#可解释性分析模块)
- [特征重要性分析模块](#特征重要性分析模块)
- [可视化模块](#可视化模块)
- [技术栈说明](#技术栈说明)
- [开发指南](#开发指南)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 功能一览

| 功能 | 说明 |
| --- | --- |
| 特征工程 | 原始/结构/时间/拓扑特征提取与融合 |
| 模型训练 | 10种模型（MLP/XGBoost/GCN/GraphSAGE等）训练与评估 |
| 模型融合 | 5种融合策略（AP加权/平均/最大/RRF/几何平均） |
| 可解释性分析 | GNNExplainer、t-SNE可视化、子图分析 |
| 特征重要性 | XGBoost+LightGBM综合评分与特征筛选 |
| ROC曲线可视化 | 多模型ROC对比与TPR/FPR数据导出 |
| 深度图分析 | 连通分量、中心性、社区审计、时间窗口分析 |

---

## 特征工程模块

### 使用步骤

#### 1、数据加载与预处理
运行 `main.py` 程序，通过 `--data` 参数指定数据集（`sample` 或 `full`），系统自动加载 DGraphFin 数据集并进行标准化预处理。

```bash
python main.py --data sample --stages 1
```

**Note:** 数据集需提前放置于 `data/` 目录，格式支持 `.pt` 和 `.npz`。

#### 2、特征集生成
系统支持6种特征集自动生成，通过 `--features` 参数指定：

```bash
python main.py --data sample --stages 1 --features raw structural temporal topology full important
```

| 特征集 | 维度 | 说明 |
| --- | --- | --- |
| `raw` | 17 | 原始特征 |
| `structural` | 43 | 原始+度特征 |
| `temporal` | 24 | 原始+时间特征 |
| `topology` | 40 | 图拓扑特征 |
| `full` | 107 | 全特征集 |
| `important` | 37 | 精选特征（累积贡献度80%） |

**Tip:** 首次运行时系统会自动生成中间缓存文件（`_base_data.pt`、`_centrality.pt`等），后续运行可复用缓存加速。

#### 3、深度图分析
特征工程阶段自动触发深度图分析，包括：
- 连通分量分析
- Top-K 中心性节点异常率统计
- 社区审计（Top-5高风险社区识别）
- 时间窗口分析

**Important:** 分析结果保存于 `output/{dataset}/results/graph_analysis.json`。

---

## 模型训练模块

### 使用步骤

#### 1、启动训练
运行 `main.py` 指定 `--stages 2` 启动模型训练：

```bash
python main.py --data sample --stages 2 --features full --models mlp graphsage
```

**Tip:** 通过 `--models` 参数可指定训练特定模型，默认训练全部10个模型。

#### 2、模型选择
系统支持10种模型，分为4类：

| 类型 | 模型 | 特点 |
| --- | --- | --- |
| 非图模型 | MLP、LightGBM、XGBoost | 基于节点自身特征 |
| 同构GNN | GCN、GraphSAGE、GAT | 基于邻居结构信息 |
| 异构GNN | HeteroSAGE、RGCN | 支持多边类型 |
| 时序GNN | EvolveGCN、TCN | 支持时间演化 |

#### 3、大图策略
在 `full` 数据集上训练复杂模型（GAT/HeteroSAGE/RGCN/EvolveGCN/TCN）时，系统自动切换到CPU训练，避免GPU显存不足。

**Note:** 训练进度每轮打印一次，包含训练损失、验证集AUC和AP指标。

#### 4、模型保存
训练完成后模型自动保存至 `output/{dataset}/models/{model}_{feature}.pt`，同时保存训练历史至 `output/{dataset}/results/{model}_{feature}_history.json`。

---

## 模型融合模块

### 使用步骤

#### 1、启动融合
训练完成后运行 `--stages 3` 自动执行评估与融合：

```bash
python main.py --data sample --stages 3 --features full
```

#### 2、融合策略选择
系统支持5种融合策略：

| 策略 | 说明 | 适用场景 |
| --- | --- | --- |
| `ap_weighted` | 验证集AP归一化加权 | 考虑模型可靠性 |
| `mean` | 等权算术平均 | 简单基线 |
| `max` | 取最大预测概率 | 保守高召回 |
| `rank` | 倒数排名融合RRF | 消除尺度差异 |
| `geomean` | 几何平均 | 要求共识确认 |

**Tip:** 通过修改 `config.py` 中的 `FUSION_METHODS` 可调整启用的融合策略。

#### 3、融合组合配置
在 `config.py` 的 `FUSION_COMBOS` 中配置模型组合，支持跨类互补组合：

```python
FUSION_COMBOS = {
    'mlp+graphsage': ['mlp', 'graphsage'],
    'lightgbm+graphsage+evolvegcn': ['lightgbm', 'graphsage', 'evolvegcn'],
}
```

#### 4、结果汇总
所有模型与融合结果汇总至 `output/{dataset}/results/summary.csv`，包含AUC、AP、Recall@K、Precision@K等指标。

---

## 可解释性分析模块

### 使用步骤

#### 1、启动可解释性分析
运行 `--stages 4` 自动选择最佳GNN模型进行解释：

```bash
python main.py --data sample --stages 4 --features full
```

#### 2、GNNExplainer分析
系统自动加载测试集AP最高的GNN模型，使用GNNExplainer解释特定节点的预测结果，输出边重要性掩码。

**Note:** 解释结果保存于 `output/{dataset}/interpret/` 目录。

#### 3、子图可视化
系统抽取目标节点的K-hop子图，保存为可视化数据，便于后续分析节点周围的图结构。

---

## 特征重要性分析模块

### 使用步骤

#### 1、启动特征重要性分析
运行 `--stages 5` 执行特征重要性分析：

```bash
python main.py --data sample --stages 5 --features full
```

#### 2、综合评分计算
系统使用XGBoost和LightGBM的permutation重要性与内置重要性加权计算综合评分：

- permutation XGBoost: 30%
- permutation LightGBM: 30%
- XGBoost gain: 20%
- LightGBM gain: 20%

#### 3、特征筛选
运行 `select_important_features.py` 基于累积贡献度阈值筛选特征：

```bash
python select_important_features.py
```

**Important:** 筛选结果更新至 `config.py` 的 `IMPORTANT_FEATURE_DIMS`，需重新运行stage 1生成新特征集。

---

## 可视化模块

### 使用步骤

#### 1、ROC曲线生成
运行 `--stages 6` 生成多模型ROC对比图：

```bash
python main.py --data sample --stages 6 --features full
```

#### 2、风险分数分布
评估阶段自动生成每个模型的风险分数分布图，Normal（蓝色）与Fraud（红色）对比，保存于 `output/{dataset}/results/score_distributions/`。

#### 3、度分布对比
特征工程阶段生成正常/异常节点度分布对比图，包含线性尺度和log-log尺度，保存于 `output/{dataset}/results/data_degree_by_label.png`。

---

## 技术栈说明

| 技术/框架 | 用途 |
| --- | --- |
| Python | 核心编程语言 |
| PyTorch | 深度学习框架 |
| PyTorch Geometric | 图神经网络库 |
| scikit-learn | 传统机器学习算法与评估指标 |
| XGBoost | 梯度提升树模型 |
| LightGBM | 高效梯度提升树模型 |
| pandas | 数据处理与分析 |
| numpy | 数值计算 |
| matplotlib | 数据可视化 |
| seaborn | 统计图表绘制 |
| uv | Python包管理工具 |

---

## 开发指南

### 仓库克隆

```bash
git clone <repository-url>
cd 0717-proj
```

### 依赖安装

```bash
# 使用uv创建虚拟环境并安装依赖
uv venv
uv pip install -r requirements.txt
```

### 开发环境运行

```bash
# 激活虚拟环境（Windows）
.venv/Scripts/activate

# 运行完整流程（sample数据集）
python main.py --data sample --stages 1 2 3 4 5 6

# 运行特定阶段
python main.py --data sample --stages 2 3 --features full

# 指定模型
python main.py --data sample --stages 2 --models mlp graphsage
```

### 项目构建

项目无需编译，直接运行Python脚本即可。

### 测试执行

```bash
# 运行完整测试流程
python main.py --data sample --stages 2 3

# 验证特征工程
python main.py --data sample --stages 1
```

### 断点续跑

系统支持断点续跑，已训练/已评估的模型会自动跳过：

```bash
# 第一次运行（训练部分模型）
python main.py --data sample --stages 2 --features raw

# 第二次运行（自动跳过已训练模型，继续训练其他特征集）
python main.py --data sample --stages 2 --features structural full
```

---

## 项目结构

```
0717-proj/
├── config.py                 # 全局配置（超参数、路径、模型列表）
├── main.py                   # 主流程编排（6个阶段）
├── utils.py                  # 工具函数（指标计算、预测、IO）
├── requirements.txt          # 依赖列表
├── select_important_features.py  # 特征选择脚本
├── .gitignore                # Git忽略文件
├── README.md                 # 项目文档
│
├── models/                   # 模型定义
│   ├── __init__.py           # 模型构建入口
│   ├── mlp.py                # MLP模型
│   ├── lightgbm_model.py     # LightGBM模型
│   ├── xgboost_model.py      # XGBoost模型
│   ├── gcn.py                # GCN模型
│   ├── graphsage.py          # GraphSAGE模型
│   ├── gat.py                # GAT模型
│   ├── hetero_sage.py        # HeteroSAGE模型
│   ├── rgcn.py               # RGCN模型
│   ├── evolve_gcn.py         # EvolveGCN模型
│   └── tcn.py                # TCN模型
│
├── s1_data_features.py       # 阶段1：特征工程（数据理解、特征提取）
├── s2_train_eval.py          # 阶段2：模型训练（训练循环、早停）
├── s3_eval_fusion.py         # 阶段3：测试评估与模型融合
├── s4_interpret.py           # 阶段4：可解释性分析（GNNExplainer、子图）
├── s5_feature_importance.py  # 阶段5：特征重要性分析
├── s6_visualization.py       # 阶段6：ROC曲线可视化
│
├── data/                     # 数据集（.gitignore忽略）
│   ├── dgraphfin_sample/     # 采样数据集（5万节点）
│   └── DGraphFin/            # 完整数据集（370万节点）
│
├── output/                   # 输出目录（.gitignore忽略）
│   ├── sample/               # sample数据集输出
│   │   ├── features/         # 特征张量
│   │   ├── models/           # 训练模型
│   │   ├── results/          # 评估结果与融合结果
│   │   ├── interpret/        # 可解释性分析结果
│   │   └── feature_importance/  # 特征重要性分析
│   ├── full/                 # full数据集输出
│   └── main/                 # 运行日志
└── .workbuddy/               # 工作记忆（.gitignore忽略）
```

---

## 许可证

本项目采用 **MIT 许可证**。

MIT许可证允许自由使用、复制、修改和分发本项目的代码，无论是否用于商业目的。使用者只需保留原始版权声明和许可证声明即可。具体条款请参见项目根目录下的 LICENSE 文件。

---

**项目维护者:** Developer  
**联系方式:** developer@example.com  
**最后更新:** 2026年7月
