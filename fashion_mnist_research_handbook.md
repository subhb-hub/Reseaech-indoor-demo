# Fashion-MNIST Research Demo Handbook

## 0. 本节课目标

你不是要直接跑出一个高准确率模型，而是要体验一次最小科研闭环：

```text
论文搜索 -> 官方仓库 -> 官方 convnet.py -> 架构阅读 -> PyTorch 迁移版 baseline -> toy innovation -> 消融实验 -> demo paper
```

最终定位：

```text
官方 Fashion-MNIST 仓库用于“论文、原始项目和官方 baseline 理解”；
本课堂仓库用于“PyTorch 工程化迁移、toy innovation 和消融实验”。
```

## 1. 找到论文

搜索关键词：

```text
Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms
Fashion-MNIST paper
Fashion-MNIST arXiv
```

记录：

- 论文标题
- 作者
- 数据集大小
- 图像尺寸
- 类别数量
- 官方 GitHub 仓库

## 2. Clone 官方仓库

```bash
git clone https://github.com/zalandoresearch/fashion-mnist.git official_fashion_mnist
```

观察：

- `README.md`
- `data/`
- `utils/`
- `benchmark/`
- `visualization/`

也可以使用课堂脚本：

```bash
bash scripts/clone_official_repo.sh
```

## 3. 复制必要文件到课堂仓库

不要复制 `data/`。

复制：

- `README.md`
- `README.zh-CN.md`，如果官方仓库中存在
- `LICENSE`
- `utils/mnist_reader.py`
- `benchmark/convnet.py`

脚本会复制到：

```text
third_party/fashion-mnist/
```

## 4. 阅读官方代码

先读：

- `third_party/fashion-mnist/README.md`
- `third_party/fashion-mnist/mnist_reader.py`
- `third_party/fashion-mnist/benchmark/convnet.py`

回答：

1. `README.md` 中写了 Fashion-MNIST 有多少训练样本？
2. 每张图像的大小是多少？
3. 一共有多少类？
4. `data/fashion` 目录原本是用来放什么的？
5. `mnist_reader.py` 是如何读取原始 IDX 数据的？
6. 官方 ConvNet 的第一层和第二层卷积分别是什么配置？
7. 为什么课堂训练不用官方 TensorFlow 旧代码，而使用 PyTorch 迁移版？

## 5. 阅读官方 ConvNet 并填写架构表

打开：

```text
third_party/fashion-mnist/benchmark/convnet.py
```

找到：

```python
def cnn_model_fn(features, labels, mode):
```

然后填写：

```text
templates/architecture_table.md
```

再对照本仓库中的 PyTorch 迁移版：

```text
src/model_official_like.py
```

核心结构：

```text
TensorFlow input [B, 28, 28, 1]
-> PyTorch input [B, 1, 28, 28]

Conv2D 32 filters, 5x5, same padding
-> nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)

Conv2D 64 filters, 5x5, same padding
-> nn.Conv2d(32, 64, kernel_size=5, padding=2)

Flatten 7*7*64
-> nn.Linear(7 * 7 * 64, 1024)
```

## 6. 运行官方风格 PyTorch baseline

```bash
python src/train.py --model official_like_cnn --variant baseline --epochs 2 --use-subset
```

默认使用 subset，保证课堂能跑完。完整训练可以在课后运行：

```bash
python src/train.py --model official_like_cnn --variant baseline --epochs 10 --no-subset
```

`src/model_baseline.py` 中的 `SimpleCNN` 可以作为 warm-up model，但本课程主 baseline 是 `OfficialLikeCNN`。

## 7. 找到创新插入位置

阅读：

- `src/data.py`
- `src/transforms_custom.py`
- `src/model_official_like.py`
- `src/train.py`

回答：

1. transform 在哪里传入？
2. `OfficialLikeCNN` 的第一层输入通道在哪里定义？
3. 如果增加 edge channel，为什么只需要把第一层 `in_channels` 从 1 改成 2？
4. 为什么 `fc1 = nn.Linear(7 * 7 * 64, 1024)` 不需要改？

关键形状：

```text
baseline 输入 shape: [1, 28, 28]
edge 输入 shape:    [2, 28, 28]
noise 输入 shape:   [2, 28, 28]
```

## 8. 实现 toy innovation

目标：

```text
original image -> [original image, edge map]
```

在 `src/transforms_custom.py` 中补完：

- `EdgeEnhancedTransform`
- `NoiseEnhancedTransform`

不要一开始就追求复杂方法。先实现一个能解释、能对比、能写进实验表的小改动。

## 9. 做消融实验

课堂默认命令：

```bash
python src/train.py --model official_like_cnn --variant baseline --epochs 2 --use-subset
python src/train.py --model official_like_cnn --variant edge --epochs 2 --use-subset
python src/train.py --model official_like_cnn --variant noise --epochs 2 --use-subset
```

记录：

- baseline 是否能稳定运行？
- edge 是否比 baseline 更好？
- noise 是否也变好？
- 如果 edge 和 noise 都变好，能不能证明 edge 有效？
- 如果 edge 没有变好，论文里应该如何诚实讨论？

## 10. 写 demo paper

使用：

```text
templates/demo_paper_template.tex
```

至少写清楚：

- Introduction: 为什么选择 Fashion-MNIST？
- Baseline: 官方 `benchmark/convnet.py` 和 `OfficialLikeCNN` 的关系是什么？
- Method: edge channel 是什么？
- Experiments: baseline / edge / noise 怎么对比？
- Results: 表格中的 accuracy 是多少？
- Discussion: 结果支持还是不支持你的想法？
