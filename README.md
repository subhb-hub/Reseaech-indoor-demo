# Fashion-MNIST CS Research Demo

这是一个给大一学生使用的 CS 科研入门实操仓库。课堂目标不是追求最高精度，而是手把手走完一次最小科研闭环：

读论文 -> 跑 baseline -> 做一个 toy innovation -> 设计对照实验 -> 整理结果 -> 写 demo paper。

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

先用 1 个 epoch 快速确认全流程：

```bash
python train_fashion.py --variant baseline --epochs 1 --device cpu
python train_fashion.py --variant edge --epochs 1 --device cpu
python train_fashion.py --variant noise --epochs 1 --device cpu
```

如果时间和电脑性能允许，再改成 3 个 epoch：

```bash
python train_fashion.py --variant baseline --epochs 3 --device cpu
python train_fashion.py --variant edge --epochs 3 --device cpu
python train_fashion.py --variant noise --epochs 3 --device cpu
```

运行后会生成：

- `data/`: 自动下载的 Fashion-MNIST 数据。
- `outputs/`: 训练好的模型权重。
- `results_summary.csv`: 每次实验的汇总记录。

## 一键运行三个实验

macOS / Linux 可以执行：

```bash
bash scripts/run_all_experiments.sh 1
```

其中 `1` 表示每个实验跑 1 个 epoch。想跑 3 个 epoch：

```bash
bash scripts/run_all_experiments.sh 3
```

## 项目结构

```text
.
├── data/                              # 数据集下载目录，不提交真实数据
├── demo_paper/                        # demo LaTeX paper 模板
├── outputs/                           # 模型输出目录，不提交模型权重
├── scripts/
│   ├── check_setup.py                 # 课前环境检查
│   └── run_all_experiments.sh          # 一键跑三个实验
├── train_fashion.py                   # 主训练脚本
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

## 写作模板

`demo_paper/main.tex` 是一份 2--4 页 demo paper 模板。学生跑完实验后，把 `results_summary.csv` 里的准确率填入表格即可。

## 说明

这个项目用于教学。Edge-Enhanced Fashion-CNN 是一个训练科研流程的 toy idea，不代表正式科研贡献，也不保证一定提升准确率。

课堂默认全部使用 CPU 运行，以减少不同学生电脑之间的环境差异。
