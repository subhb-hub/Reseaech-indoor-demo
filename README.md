# Fashion-MNIST CS Research Demo

这是一个给大一学生使用的 CS 科研入门实操仓库。课堂目标不是追求最高精度，而是手把手走完一次最小科研闭环：

找论文 -> 找官方代码仓库 -> clone 官方仓库 -> 复现 baseline -> 自己实现 toy innovation -> 设计对照实验 -> 整理结果 -> 写 demo paper。

配套讲义见 [fashion_mnist_research_handbook.md](fashion_mnist_research_handbook.md)。

## 课前准备

推荐使用 conda：

```bash
conda create -n fashion-demo python=3.10 -y
conda activate fashion-demo
pip install -r requirements.txt
```

也可以用已有 Python 环境：

```bash
python -m pip install -r requirements.txt
```

检查环境：

```bash
python scripts/check_setup.py
```

## 课堂运行

### 1. 找到论文

课堂上先从搜索开始，而不是直接运行代码。

推荐搜索关键词：

```text
Fashion-MNIST paper
Fashion-MNIST arXiv
Fashion-MNIST a Novel Image Dataset for Benchmarking Machine Learning Algorithms
```

目标论文：

- arXiv: https://arxiv.org/abs/1708.07747
- 题目：Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms

### 2. 找到官方代码仓库

在论文页面、论文 PDF 或搜索结果中找到官方仓库：

```text
https://github.com/zalandoresearch/fashion-mnist
```

判断它是不是官方仓库时，看三个信号：

- 作者或组织是 `zalandoresearch`。
- README 中说明它是 Fashion-MNIST 数据集。
- 仓库包含 `data/`、`utils/`、`benchmark/` 等目录。

### 3. Clone 官方仓库并检查

课堂上执行：

```bash
bash scripts/clone_official_repo.sh
```

它会把官方仓库克隆到：

```text
external/fashion-mnist
```

检查脚本会用官方仓库的 `utils/mnist_reader.py` 读取数据，并打印训练集、测试集 shape。

### 4. 用官方数据链路做一次轻量复现

官方 README 提到可以运行 `benchmark/runner.py` 复现它的 benchmark，但完整 benchmark 覆盖很多 scikit-learn 分类器，课堂上可能偏慢。我们先做一个更适合笔记本的 smoke reproduction：

```bash
python scripts/reproduce_official_smoke.py
```

这一步会：

- 使用官方仓库的 `utils/mnist_reader.py` 读取 `external/fashion-mnist/data/fashion`。
- 取一个小训练子集和测试子集。
- 用一个轻量 scikit-learn baseline 训练并输出 accuracy。

做到这里，学生已经完成了“找到论文 -> 找到官方仓库 -> clone -> 用官方数据链路复现一个最小实验”。

### 5. 复现本课程 PyTorch baseline

课程里为了让 CPU 笔记本稳定完成 CNN 训练，我们再使用本仓库的 PyTorch baseline 进行复现。

先用 1 个 epoch 快速确认 baseline：

```bash
python train_fashion.py --variant baseline --epochs 1 --device cpu
```

如果时间和电脑性能允许，再改成 3 个 epoch：

```bash
python train_fashion.py --variant baseline --epochs 3 --device cpu
```

运行后会生成：

- `data/`: 自动下载的 Fashion-MNIST 数据。
- `outputs/`: 训练好的模型权重。
- `results_summary.csv`: 每次实验的汇总记录。

### 6. 课堂实现创新

`train_fashion.py` 里已经留下了两个课堂 TODO：

- `sobel_edge`: 学生现场实现 Sobel 边缘图。
- `ExtraChannelDataset`: 学生现场实现 `edge` 和 `noise` 两种额外通道。

完成 TODO 后，再运行：

```bash
python train_fashion.py --variant edge --epochs 1 --device cpu
python train_fashion.py --variant noise --epochs 1 --device cpu
```

### 7. 半自动课堂脚本

macOS / Linux 可以执行：

```bash
bash scripts/run_all_experiments.sh 1
```

这个脚本默认只跑 baseline，并提示创新 TODO 完成后的下一步命令。

## 项目结构

```text
.
├── data/                              # 数据集下载目录，不提交真实数据
├── demo_paper/                        # demo LaTeX paper 模板
├── outputs/                           # 模型输出目录，不提交模型权重
├── scripts/
│   ├── check_setup.py                 # 课前环境检查
│   ├── check_official_clone.py         # 检查官方仓库和数据读取
│   ├── clone_official_repo.sh          # clone 官方 Fashion-MNIST 仓库
│   ├── reproduce_official_smoke.py     # 用官方数据链路做轻量复现
│   └── run_all_experiments.sh          # 先跑 baseline，再提示创新实验
├── train_fashion.py                   # 主训练脚本，创新部分保留 TODO
├── requirements.txt
├── environment.yml
└── fashion_mnist_research_handbook.md
```

## 实验变体

| Variant | 输入 | 作用 |
|---|---|---|
| `baseline` | 原始灰度图 | 基础对照 |
| `edge` | 原始灰度图 + Sobel 边缘图 | 教学型创新 |
| `noise` | 原始灰度图 + 随机噪声图 | 负对照 |

注意：`edge` 和 `noise` 不是开箱即用答案，需要课堂上由学生补完 TODO。

## 写作模板

`demo_paper/main.tex` 是一份 2--4 页 demo paper 模板。学生跑完实验后，把 `results_summary.csv` 里的准确率填入表格即可。

## 说明

这个项目用于教学。Edge-Enhanced Fashion-CNN 是一个训练科研流程的 toy idea，不代表正式科研贡献，也不保证一定提升准确率。

课堂默认全部使用 CPU 运行，以减少不同学生电脑之间的环境差异。
