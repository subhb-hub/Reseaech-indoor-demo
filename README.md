# Fashion-MNIST Research Demo

这是一个给大一学生使用的 CS 科研入门课堂仓库。目标不是一键跑出最高准确率，而是让学生真实经历一次最小科研闭环：

```text
论文搜索 -> 官方仓库 -> 复制/迁移少量代码 -> 阅读代码 -> baseline 复现 -> toy innovation -> 消融实验 -> demo paper
```

核心设计原则：

- 官方 `zalandoresearch/fashion-mnist` 仓库用于理解论文、README、目录结构和原始 IDX 读取方式。
- 本课堂仓库中的训练统一使用 `torchvision.datasets.FashionMNIST` 自动下载和缓存数据。
- 不复制官方仓库的 `data/` 目录，避免课堂时间浪费在数据路径、压缩文件和 IDX 文件读取错误上。
- `edge` 和 `noise` 变体保留 TODO，让学生自己找到并实现创新插入位置。

## 课前准备

```bash
conda create -n fashion-demo python=3.10 -y
conda activate fashion-demo
pip install -r requirements.txt
python scripts/check_setup.py
```

## 课堂路线

### 1. 找论文和官方仓库

先搜索论文：

```text
Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms
```

再找到官方仓库：

```text
https://github.com/zalandoresearch/fashion-mnist
```

阅读任务见 [docs/01_find_paper_and_repo.md](docs/01_find_paper_and_repo.md)。

### 2. Clone 官方仓库并复制少量文件

```bash
bash scripts/clone_official_repo.sh
```

脚本会 clone 官方仓库到：

```text
official_fashion_mnist/
```

并复制这些文件到：

```text
third_party/fashion-mnist/
├── README.md
├── README.zh-CN.md   # 如果官方仓库中存在
├── LICENSE
└── mnist_reader.py
```

注意：不要复制官方 `data/` 目录。阅读任务见 [docs/02_copy_and_read_official_code.md](docs/02_copy_and_read_official_code.md)。

### 3. 跑课堂 baseline

第一次运行会由 `torchvision.datasets.FashionMNIST` 自动下载数据到 `data/`：

```bash
python src/train.py --variant baseline --epochs 2 --use-subset
```

兼容旧入口也可以运行：

```bash
python train_fashion.py --variant baseline --epochs 2 --use-subset
```

### 4. 找创新插入位置

阅读：

- [src/data.py](src/data.py)
- [src/transforms_custom.py](src/transforms_custom.py)
- [src/model_baseline.py](src/model_baseline.py)
- [src/train.py](src/train.py)

重点问题：

- transform 在哪里传入？
- 模型输入通道在哪里定义？
- 如果输入从 `[1, 28, 28]` 变成 `[2, 28, 28]`，哪些文件会被牵动？

任务见 [docs/03_find_innovation_position.md](docs/03_find_innovation_position.md)。

### 5. 实现 toy innovation 和消融

学生补完 [src/transforms_custom.py](src/transforms_custom.py) 中的 TODO 后运行：

```bash
python src/train.py --variant baseline --epochs 2 --use-subset
python src/train.py --variant edge --epochs 2 --use-subset
python src/train.py --variant noise --epochs 2 --use-subset
```

课后完整训练：

```bash
python src/train.py --variant baseline --epochs 10 --no-subset
python src/train.py --variant edge --epochs 10 --no-subset
python src/train.py --variant noise --epochs 10 --no-subset
```

## 项目结构

```text
.
├── README.md
├── third_party/
│   └── fashion-mnist/
│       └── SOURCE.md
├── src/
│   ├── data.py
│   ├── model_baseline.py
│   ├── train.py
│   ├── evaluate.py
│   └── transforms_custom.py
├── templates/
│   ├── innovation_template.py
│   ├── experiment_record.csv
│   └── demo_paper_template.tex
├── docs/
│   ├── 00_task_card.md
│   ├── 01_find_paper_and_repo.md
│   ├── 02_copy_and_read_official_code.md
│   ├── 03_find_innovation_position.md
│   └── 04_write_demo_paper.md
├── scripts/
│   ├── clone_official_repo.sh
│   ├── check_official_clone.py
│   ├── check_setup.py
│   └── run_all_experiments.sh
└── train_fashion.py
```

## 写作

跑完实验后，把 `results_summary.csv` 中的结果整理到：

- [templates/experiment_record.csv](templates/experiment_record.csv)
- [templates/demo_paper_template.tex](templates/demo_paper_template.tex)

更详细的写作任务见 [docs/04_write_demo_paper.md](docs/04_write_demo_paper.md)。
