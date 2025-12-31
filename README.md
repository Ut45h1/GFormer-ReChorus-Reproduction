# GFormer-ReChorus Reproduction
**机器学习大作业 - Graph Transformer for Recommendation 模型复现与改进**

## 👥 小组成员
* **袁智豪** (23330158) - 负责核心模型构建与代码复现
* **郑瀚** (23330173) - 负责环境搭建、实验测试与报告撰写

## 📂 项目结构
本项目基于 ReChorus 框架复现了 GFormer 模型，并针对 Windows/WSL 环境进行了工程优化。

* `src/`: 核心代码目录
  * `models/general/GFormer.py`
  * `models/general/LightGCN.py`

* `data/`: 数据集目录
  * `LastFM/`: 包含处理好的训练与测试数据
  * `Grocery_and_Gourmet_Food/`:包含处理后的 Amazon Grocery 数据
* `log/`: 实验运行日志
* `requirements.txt`: 项目依赖库列表

---

## 🚀 快速开始 (Quick Start)

### 1. 环境安装
请确保安装了 Python 3.8+ 和 PyTorch。安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行实验

以下命令针对 Windows 环境作了优化（增加了 `--num_workers 0` 和 `--buffer 0` 以防止内存溢出和参数清零 Bug）。

我们实际实验时使用了WSL环境，命令由`--num_workers 0` 调整为 `--num_workers 4`

首先确保处于src文件夹中：
```bash
cd src
```
#### 复现 GFormer
*数据相对稀疏，主要验证模型有效性。*

```bash
python main.py --model_name GFormer --dataset LastFM --lr 1e-3 --l2 1e-4 --gpu 0 --path ../data/ --test_all 1 --lambda1 1.0 --lambda2 0.001 --n_layers 2 --num_workers 0
```


*数据极度稀疏 (5-core)，验证模型在工业级稀疏场景下的鲁棒性。*

```bash
python main.py --model_name GFormer --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-4 --gpu 0 --path ../data/ --test_all 1 --lambda1 1.0 --lambda2 0.001 --n_layers 2 --num_workers 0 --buffer 0
```

#### 基准模型对比 (LightGCN)

```bash
# LastFM
python main.py --model_name LightGCN --dataset LastFM --lr 1e-3 --l2 1e-4 --gpu 0 --path ../data/ --test_all 1 --num_workers 0

# Grocery
python main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-4 --gpu 0 --path ../data/ --test_all 1 --num_workers 0 --buffer 0
```

#### 基准模型对比 (DirectAU)

```bash
# LastFM
python main.py --model_name DirectAU --dataset LastFM --lr 1e-3 --l2 1e-4 --gpu 0 --path ../data/ --test_all 1 --num_workers 0

# Grocery
python main.py --model_name DirectAU --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-4 --gpu 0 --path ../data/ --test_all 1 --num_workers 0
```
---