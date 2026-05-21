# Fashion-MNIST Research Demo Handbook

## 0. 本节课目标

你不是要直接跑出一个高准确率模型，而是要体验一次最小科研闭环：

```text
论文搜索 -> 官方仓库 -> 代码阅读 -> baseline 复现 -> toy innovation -> 消融实验 -> demo paper
```

最终定位：

```text
官方 Fashion-MNIST 仓库用于“论文与原始项目理解”；
本课堂仓库用于“工程化复现、toy innovation 和消融实验”。
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

脚本会复制到：

```text
third_party/fashion-mnist/
```

## 4. 阅读官方代码

回答：

1. `README.md` 中写了 Fashion-MNIST 有多少训练样本？
2. 每张图像的大小是多少？
3. 一共有多少类？
4. `data/fashion` 目录原本是用来放什么的？
5. `utils/mnist_reader.py` 是如何读取原始数据的？
6. 为什么我们最终不用 `mnist_reader.py`，而改用 `torchvision.datasets.FashionMNIST`？

注意：我们复制官方仓库中的 `mnist_reader.py`，不是为了直接使用它训练模型，而是为了学习官方代码如何组织数据读取逻辑。

在本课程仓库中，最终训练统一使用 `torchvision.datasets.FashionMNIST`，原因是：

1. 它是 PyTorch 生态中的标准数据接口；
2. 它可以自动下载和缓存数据；
3. 它能与 `transforms` 和 `DataLoader` 无缝连接；
4. 它更适合课堂快速复现实验。

## 5. 运行课堂 baseline

```bash
python src/train.py --variant baseline --epochs 2 --use-subset
```

默认使用 subset，保证课堂能跑完。完整训练可以在课后运行：

```bash
python src/train.py --variant baseline --epochs 10 --no-subset
```

## 6. 找到创新插入位置

阅读：

- `src/data.py`
- `src/transforms_custom.py`
- `src/model_baseline.py`
- `src/train.py`

回答：

1. transform 在哪里传入？
2. 模型输入通道在哪里定义？
3. 如果增加 edge channel，哪些文件要修改？

关键形状：

```text
原图输入 shape: [1, 28, 28]
edge 输入 shape: [2, 28, 28]
模型第一层 Conv2d 的 in_channels 也要从 1 改成 2
```

## 7. 实现 toy innovation

目标：

```text
original image -> [original image, edge map]
```

在 `src/transforms_custom.py` 中补完：

- `EdgeEnhancedTransform`
- `NoiseEnhancedTransform`

不要一开始就追求复杂方法。先实现一个能解释、能对比、能写进实验表的小改动。

## 8. 做消融实验

课堂默认命令：

```bash
python src/train.py --variant baseline --epochs 2 --use-subset
python src/train.py --variant edge --epochs 2 --use-subset
python src/train.py --variant noise --epochs 2 --use-subset
```

记录：

- baseline 是否能稳定运行？
- edge 是否比 baseline 更好？
- noise 是否也变好？
- 如果 edge 和 noise 都变好，能不能证明 edge 有效？
- 如果 edge 没有变好，论文里应该如何诚实讨论？

## 9. 写 demo paper

使用：

```text
templates/demo_paper_template.tex
```

至少写清楚：

- Introduction: 为什么选择 Fashion-MNIST？
- Method: baseline 是什么？edge channel 是什么？
- Experiments: baseline / edge / noise 怎么对比？
- Results: 表格中的 accuracy 是多少？
- Discussion: 结果支持还是不支持你的想法？

## 10. 这个设计的优点

这条路线把“复现”拆成几个真正有教育意义的动作：

```text
不是直接给数据，而是让学生知道数据从哪里来；
不是直接给最终代码，而是让学生知道代码从哪里来；
不是直接给创新，而是让学生找到创新应该插在哪里；
不是只跑一个结果，而是 baseline / innovation / negative control 三组对比；
不是只写感想，而是写成 demo paper。
```
