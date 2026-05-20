# Fashion-MNIST 新生科研入门实操手册

> 主题论文：**Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms**  
> 教学目标：让新生在一次课内完成“读论文 → 找代码/数据 → 跑 baseline → 做一个 toy 创新 → 写 demo LaTeX paper”的完整科研闭环。  
> 本手册中的“创新点”是**教学型创新**，不保证真的有效，也不能直接当作正式论文贡献。它的价值在于训练科研流程、实验意识和论文表达。

---

## 0. 这次实操最终要交付什么？

完成本手册后，每位同学至少应该得到下面 4 个东西：

1. 一个能跑通的 Fashion-MNIST 分类 baseline；
2. 一个小型 toy innovation：**Edge-Enhanced Fashion-CNN**；
3. 一张实验对比表：Baseline vs. Edge-Enhanced vs. Noise-Control；
4. 一篇 2--4 页的 demo LaTeX paper 草稿。

你可以把它理解成一次“迷你科研训练营”：我们不追求 SOTA，而是追求第一次完整走通科研流程。

---

## 1. 论文与任务背景

### 1.1 这篇论文在做什么？

Fashion-MNIST 是一个类似 MNIST 的图像分类数据集。每张图像是 `28 × 28` 的灰度图，共有 10 个服饰类别，例如 T-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot。

它的基本设定是：

- 训练集：60,000 张图像；
- 测试集：10,000 张图像；
- 类别数：10 类；
- 图像尺寸：28 × 28；
- 图像类型：灰度图；
- 目标：输入一张服饰图片，预测其类别。

这篇论文的核心贡献不是提出一个复杂模型，而是提出一个更适合作为入门 benchmark 的数据集。它比传统 MNIST 更贴近真实视觉任务，同时又足够小，适合新手快速上手。

### 1.2 为什么适合新生科研入门？

因为它满足 4 个重要条件：

| 条件 | 为什么重要 | Fashion-MNIST 是否满足 |
|---|---|---|
| 数据集小 | 可以在普通电脑上跑 | 是 |
| 任务清楚 | 分类任务容易理解 | 是 |
| 代码生态成熟 | PyTorch / TensorFlow 都有现成接口 | 是 |
| 可做小创新 | CNN、数据增强、边缘特征、损失函数都能改 | 是 |

这意味着：新生不需要先学完所有深度学习理论，也能先完成一个最小可行科研项目。

---

## 2. 项目结构

本仓库已经提前准备好如下结构。学生课前拉取仓库后，不需要再手动创建这些文件，只需要进入仓库根目录并跟着课堂命令运行即可。

```bash
.
├── data/                              # 自动下载数据集
├── outputs/                           # 保存模型和实验结果
├── scripts/
│   ├── check_setup.py                 # 课前环境检查
│   └── run_all_experiments.sh          # 一键跑三个实验
├── train_fashion.py                   # 主训练脚本
├── results_summary.csv                # 自动生成的实验记录
└── demo_paper/                        # 后续写 LaTeX 小论文
    ├── main.tex
    └── refs.bib
```

进入项目文件夹：

```bash
cd fashion_mnist_demo
```

如果你的本地文件夹名称不是 `fashion_mnist_demo`，就进入你实际克隆下来的仓库目录。

---

## 3. 环境配置

### 3.1 使用 conda 创建环境

```bash
conda create -n fashion-demo python=3.10 -y
conda activate fashion-demo
```

### 3.2 安装依赖

课堂统一使用 CPU 版本，避免不同学生笔记本上的 CUDA / MPS 环境差异影响进度：

```bash
pip install torch torchvision numpy pandas matplotlib tqdm
```

检查是否安装成功：

```bash
python scripts/check_setup.py
```

如果能输出版本号，就说明基本环境可用。

---

## 4. 第一步：先跑通 baseline

### 4.1 什么是 baseline？

baseline 是你要比较的基础方法。它不一定很强，但必须：

- 能稳定运行；
- 实现简单；
- 结果可复现；
- 后续创新都能和它公平比较。

本手册使用一个简单 CNN 作为 baseline。

### 4.2 查看 `train_fashion.py`

本仓库已经提供了 `train_fashion.py`。课堂上可以先打开这个文件，带学生观察它由哪些部分组成：

- 参数解析；
- 数据集读取；
- baseline CNN；
- Sobel edge channel；
- noise control；
- 训练、测试和结果记录。

如果你想让学生练习“从空文件复制代码”的流程，也可以把下面完整代码作为讲义材料使用。

```python
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sobel_edge(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1, H, W], value range usually [0, 1]
    return: [1, H, W], normalized Sobel edge magnitude
    """
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)

    x4d = x.unsqueeze(0)  # [1, 1, H, W]
    gx = F.conv2d(x4d, kx, padding=1)
    gy = F.conv2d(x4d, ky, padding=1)
    edge = torch.sqrt(gx ** 2 + gy ** 2).squeeze(0)  # [1, H, W]
    edge = edge / (edge.max() + 1e-6)
    return edge


class ExtraChannelDataset(Dataset):
    """
    Wrap a FashionMNIST dataset and add an extra channel.

    mode='edge': concatenate original image and Sobel edge map.
    mode='noise': concatenate original image and random noise map.

    The noise mode is a negative control. If edge and random noise perform similarly,
    then our edge story is probably weak.
    """
    def __init__(self, base_dataset, mode: str):
        assert mode in ["edge", "noise"]
        self.base_dataset = base_dataset
        self.mode = mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]  # x: [1, 28, 28]
        if self.mode == "edge":
            extra = sobel_edge(x)
        else:
            extra = torch.rand_like(x)
        x = torch.cat([x, extra], dim=0)  # [2, 28, 28]
        return x, y


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="test", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def append_result(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="baseline",
                        choices=["baseline", "edge", "noise"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Variant: {args.variant}")
    print(f"Device: {device}")

    transform = transforms.ToTensor()

    train_base = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_base = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if args.variant in ["edge", "noise"]:
        train_dataset = ExtraChannelDataset(train_base, mode=args.variant)
        test_dataset = ExtraChannelDataset(test_base, mode=args.variant)
        in_channels = 2
    else:
        train_dataset = train_base
        test_dataset = test_base
        in_channels = 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SimpleCNN(in_channels=in_channels).to(device)
    print(f"Trainable parameters: {count_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        best_acc = max(best_acc, test_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.4f}"
        )

    model_path = os.path.join(args.output_dir, f"model_{args.variant}.pt")
    torch.save(model.state_dict(), model_path)

    result_row = {
        "variant": args.variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "params": count_parameters(model),
        "best_test_acc": round(best_acc, 6),
        "model_path": model_path,
    }
    append_result("results_summary.csv", result_row)
    print("Saved result to results_summary.csv")
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
```

---

## 5. 运行 baseline

执行：

```bash
python train_fashion.py --variant baseline --epochs 3 --device cpu
```

第一次运行会自动下载数据集。下载完成后，会看到类似输出：

```text
Variant: baseline
Device: cpu
Trainable parameters: 421642
Epoch 01/3 | train loss ... | train acc ... | test loss ... | test acc ...
Epoch 02/3 | train loss ... | train acc ... | test loss ... | test acc ...
Epoch 03/3 | train loss ... | train acc ... | test loss ... | test acc ...
Saved result to results_summary.csv
```

注意：

- 不同电脑、不同随机种子、不同 PyTorch 版本会导致结果略有变化；
- 课堂上不需要追求极限精度；
- 重点是你能解释清楚实验设置。

---

## 6. 第二步：设计一个教学型创新点

### 6.1 创新点名称

我们给这个 toy idea 起一个非常像论文的名字：

> **Edge-Enhanced Fashion-CNN: A Contour-Aware Baseline for Fashion-MNIST Classification**

或者更整活一点：

> **Edge Is Maybe All You Need**

注意：这个名字可以用来教学，但正式论文里不能这么随意。

### 6.2 动机怎么写？

Fashion-MNIST 的类别是衣服、鞋子、包等物品。很多类别之间的差异不仅体现在像素强度，还体现在轮廓上。例如：

- Sneaker 和 Sandal 的边缘结构不同；
- Bag 的整体轮廓与 T-shirt/top 不同；
- Coat、Shirt、Pullover 之间容易混淆，轮廓信息可能有帮助。

因此，我们提出一个非常简单的想法：

> 不只把原始灰度图输入 CNN，还额外计算一张 Sobel 边缘图，把它作为第二个输入通道。

### 6.3 方法怎么写？

原始 Fashion-MNIST 图像记为：

\[
X \in \mathbb{R}^{1 \times 28 \times 28}.
\]

使用 Sobel 算子计算水平和垂直方向的边缘响应：

\[
G_x = K_x * X, \quad G_y = K_y * X,
\]

其中 \(*\) 表示卷积操作。边缘强度图为：

\[
E = \sqrt{G_x^2 + G_y^2}.
\]

最后把原始图像和边缘图拼接起来：

\[
\tilde{X} = \mathrm{Concat}(X, E) \in \mathbb{R}^{2 \times 28 \times 28}.
\]

CNN 的第一层输入通道数从 1 改成 2，其余结构保持不变。

### 6.4 这个创新为什么适合教学？

因为它同时覆盖了科研训练中的几个关键概念：

| 科研概念 | 在这个 toy idea 中的体现 |
|---|---|
| 动机 | 服饰类别可能依赖轮廓信息 |
| 方法 | 添加 Sobel edge channel |
| 公平比较 | baseline 和 ours 使用同一个 CNN 主体 |
| 消融实验 | 对比原图、原图+边缘、原图+随机噪声 |
| 反证意识 | 如果随机噪声也提升，说明故事不可靠 |

---

## 7. 运行 toy innovation

### 7.1 运行 Edge-Enhanced 版本

```bash
python train_fashion.py --variant edge --epochs 3 --device cpu
```

这里的输入从 1 通道变成 2 通道：

```text
baseline input: [1, 28, 28]
edge input:    [2, 28, 28]
```

第 1 个通道是原图，第 2 个通道是边缘图。

### 7.2 运行 Noise-Control 版本

```bash
python train_fashion.py --variant noise --epochs 3 --device cpu
```

这个版本会把随机噪声作为第二个通道：

```text
noise input: [original image, random noise]
```

它的作用不是提升性能，而是作为负对照实验。

如果 Edge 版本比 Baseline 好一点，但 Noise 版本也差不多好，那么我们不能说“边缘信息真的有效”。

如果 Edge 版本比 Baseline 好，而 Noise 版本没有明显提升，那么这个 toy story 会更像一个合格的初步研究假设。

---

## 8. 整理实验结果

运行完三个版本后，打开：

```bash
results_summary.csv
```

你会看到类似结构：

| variant | epochs | params | best_test_acc |
|---|---:|---:|---:|
| baseline | 3 | ... | ... |
| edge | 3 | ... | ... |
| noise | 3 | ... | ... |

课堂报告时不要只说“我的方法更好”，而要按下面结构分析：

1. Baseline 达到了多少测试准确率？
2. Edge 版本相比 baseline 提升还是下降？
3. Noise control 的结果如何？
4. 如果 Edge 没提升，原因可能是什么？
5. 如果 Edge 提升了，它是否真的来自边缘信息，而不是参数量增加？

一个比较诚实的结论模板是：

> The edge-enhanced variant provides a simple way to inject contour information into the CNN input. Although the improvement is not guaranteed, this experiment demonstrates how to formulate a hypothesis, implement a minimal architectural change, and validate it with a baseline and a negative control.

翻译成中文就是：

> 边缘增强版本提供了一种简单的轮廓信息注入方式。虽然它不保证带来提升，但这个实验展示了如何提出假设、实现最小方法改动，并通过 baseline 和负对照进行验证。

---

## 9. 如何把它写成 demo LaTeX paper？

### 9.1 论文标题

可以使用：

```text
Edge-Enhanced Fashion-CNN: A Teaching-Oriented Baseline for Fashion-MNIST Classification
```

或者更轻松的课堂版本：

```text
Edge Is Maybe All You Need: A Toy Study on Fashion-MNIST Classification
```

### 9.2 Abstract 模板

```latex
\begin{abstract}
Fashion-MNIST is a widely used benchmark dataset for evaluating image classification algorithms on small-scale grayscale fashion images. In this demo paper, we reproduce a simple convolutional neural network baseline and introduce a teaching-oriented edge-enhanced variant. The proposed variant concatenates the original grayscale image with a Sobel edge map, allowing the model to access both appearance and contour information. We further include a random-noise control to examine whether the observed change is caused by meaningful edge information or merely by increasing the input dimensionality. This study is not intended to claim state-of-the-art performance; instead, it demonstrates a complete beginner-friendly research workflow, including baseline reproduction, minimal method modification, controlled comparison, and academic writing.
\end{abstract}
```

### 9.3 Introduction 应该写什么？

Introduction 可以分成 4 段：

1. 图像分类是机器学习入门任务；
2. Fashion-MNIST 是一个比 MNIST 更贴近真实服饰识别的 benchmark；
3. CNN baseline 能解决这个任务，但普通 CNN 只从原始像素学习特征；
4. 本文提出一个教学型边缘增强输入，用来展示如何从观察出发构造小创新。

可以直接参考下面这段：

```latex
Image classification is a fundamental task in machine learning and computer vision. Fashion-MNIST provides a compact yet more visually challenging alternative to the original MNIST dataset, as it contains grayscale images of fashion products from ten categories. Due to its small image size and standard train-test split, Fashion-MNIST is particularly suitable for beginner-level research training.

In this work, we use Fashion-MNIST as a case study to demonstrate how a beginner can move from baseline reproduction to a small method modification. We first implement a simple convolutional neural network as the baseline. Then, motivated by the observation that fashion categories often differ in object contours, we construct a Sobel edge map for each image and concatenate it with the original grayscale image as an additional input channel.

The goal of this work is not to propose a competitive model, but to provide a complete and reproducible research workflow. To avoid overclaiming, we additionally compare the edge-enhanced variant with a random-noise control, which helps examine whether the change comes from meaningful contour information or simply from increased input dimensionality.
```

### 9.4 Method 部分怎么写？

Method 只需要写清楚三件事：

1. Baseline CNN 是什么；
2. Edge map 怎么计算；
3. Edge map 怎么和原图拼接。

示例：

```latex
\section{Method}

\subsection{Baseline CNN}
We adopt a lightweight convolutional neural network as the baseline model. The network contains two convolutional blocks followed by a fully connected classifier. Each convolutional block consists of a convolutional layer, batch normalization, ReLU activation, and max pooling.

\subsection{Edge-Enhanced Input}
Given an input image $X \in \mathbb{R}^{1 \times 28 \times 28}$, we compute its Sobel edge responses along the horizontal and vertical directions:
\begin{equation}
G_x = K_x * X, \quad G_y = K_y * X,
\end{equation}
where $K_x$ and $K_y$ denote the Sobel kernels and $*$ denotes convolution. The edge magnitude map is computed as:
\begin{equation}
E = \sqrt{G_x^2 + G_y^2}.
\end{equation}
The final input is constructed by concatenating the original image and the edge map:
\begin{equation}
\tilde{X} = \mathrm{Concat}(X, E).
\end{equation}
This changes the input from one channel to two channels while keeping the main CNN architecture unchanged.
```

### 9.5 Experiment 部分怎么写？

你需要报告三个实验：

| Method | Input | Purpose |
|---|---|---|
| Baseline CNN | Original image | 基础对照 |
| Edge-Enhanced CNN | Original + Sobel edge | 教学型创新 |
| Noise-Control CNN | Original + random noise | 负对照 |

LaTeX 表格模板：

```latex
\begin{table}[t]
\centering
\caption{Comparison on Fashion-MNIST.}
\label{tab:results}
\begin{tabular}{lcc}
\hline
Method & Input Type & Test Accuracy \\
\hline
Baseline CNN & Original image & xx.xx \\
Noise-Control CNN & Original + random noise & xx.xx \\
Edge-Enhanced CNN & Original + Sobel edge & xx.xx \\
\hline
\end{tabular}
\end{table}
```

### 9.6 Discussion 应该怎么写？

不要写：

```text
Our method greatly improves the performance and proves that edge information is essential.
```

这太夸张了。

更好的写法是：

```latex
The edge-enhanced variant provides a simple and interpretable modification to the input representation. If it outperforms the baseline and the noise-control variant, the result may suggest that contour information is useful for Fashion-MNIST classification. However, if the improvement is marginal or unstable, the result should be interpreted as a teaching-oriented demonstration rather than strong evidence of methodological superiority.
```

这才是科研写作应该有的克制感。

---

## 10. 课堂现场演示建议

如果课堂时间有限，可以按下面节奏进行：

| 时间 | 内容 |
|---:|---|
| 10 min | 介绍 Fashion-MNIST 论文和数据集 |
| 10 min | 讲 baseline 是什么 |
| 15 min | 现场运行 baseline |
| 15 min | 解释 Sobel edge toy innovation |
| 15 min | 运行 edge 和 noise control |
| 20 min | 让学生填实验表格 |
| 20 min | 带学生写 Abstract / Method / Table |

如果电脑性能较弱，把 `--epochs 3` 改成 `--epochs 1`：

```bash
python train_fashion.py --variant baseline --epochs 1 --device cpu
python train_fashion.py --variant edge --epochs 1 --device cpu
python train_fashion.py --variant noise --epochs 1 --device cpu
```

课堂重点不是得到最高 accuracy，而是让学生第一次理解科研闭环。

---

## 11. 常见报错与解决方法

### 11.1 下载数据集失败

可能是网络问题。可以尝试：

```bash
python train_fashion.py --variant baseline --epochs 1 --device cpu
```

如果仍然失败，可以提前由老师下载好 `data/` 文件夹并发给学生。

### 11.2 `torch` 安装失败

先检查 Python 版本：

```bash
python --version
```

建议使用 Python 3.10 或 3.11。

### 11.3 训练太慢

减少 epoch：

```bash
python train_fashion.py --variant baseline --epochs 1 --device cpu
```

减少 batch size 不一定会更快，但可以避免内存不足：

```bash
python train_fashion.py --variant baseline --epochs 1 --batch-size 64 --device cpu
```

### 11.4 Mac 上不使用 `mps`

本仓库默认使用 CPU。即使你是 Apple Silicon Mac，也不需要使用 `mps`。如果想显式指定 CPU，可以写：

```bash
python train_fashion.py --variant baseline --epochs 1 --device cpu
```

---

## 12. 新生应该从这次实操学到什么？

这次实操的真正目标不是 Fashion-MNIST 本身，而是科研方法论：

1. 读论文时先找任务、数据、baseline、指标；
2. 找代码时优先找官方仓库或高 star 仓库；
3. 复现时不要一上来就改模型，先跑通 baseline；
4. 创新点不一定要复杂，先从一个可解释的小改动开始；
5. 实验不只是展示好结果，还要设计对照；
6. 写论文不能过度吹，要诚实解释结果。

一句话总结：

> 新生科研入门的第一步，不是提出惊天动地的创新，而是学会把一个小问题完整、规范、可复现地做完。

---

## 13. 课后可选扩展

如果学生已经跑通了这个 demo，可以继续尝试：

1. 加入数据增强：RandomCrop、RandomHorizontalFlip；
2. 使用更强模型：ResNet-like CNN；
3. 对比 MLP、CNN、Edge-CNN；
4. 画 confusion matrix，分析哪些类别容易混淆；
5. 把 edge channel 换成 Laplacian channel；
6. 写一篇完整 4 页 IEEE demo paper；
7. 把项目上传到 GitHub，并写 README。

---

## 14. 推荐 README 模板

项目上传 GitHub 时，可以写：

```markdown
# Edge-Enhanced Fashion-CNN

This is a beginner-friendly research demo based on Fashion-MNIST. We reproduce a simple CNN baseline and add a Sobel edge channel as a toy innovation. The project is designed for teaching how to move from paper reading to baseline reproduction, method modification, controlled experiments, and demo paper writing.

## Run

```bash
python train_fashion.py --variant baseline --epochs 3 --device cpu
python train_fashion.py --variant edge --epochs 3 --device cpu
python train_fashion.py --variant noise --epochs 3 --device cpu
```

## Variants

- `baseline`: original grayscale image.
- `edge`: original image + Sobel edge channel.
- `noise`: original image + random noise channel.

## Note

This project is for teaching purposes. The edge-enhanced variant is not claimed to be a state-of-the-art method.
```

---

## 15. 参考资料

1. Han Xiao, Kashif Rasul, Roland Vollgraf. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.  
   https://arxiv.org/abs/1708.07747

2. Official Fashion-MNIST GitHub Repository.  
   https://github.com/zalandoresearch/fashion-mnist

3. PyTorch Torchvision FashionMNIST Dataset Documentation.  
   https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
