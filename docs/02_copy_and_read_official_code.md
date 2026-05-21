# Copy And Read Official Code

## 1. Clone 官方仓库

```bash
git clone https://github.com/zalandoresearch/fashion-mnist.git official_fashion_mnist
```

也可以运行：

```bash
bash scripts/clone_official_repo.sh
```

## 2. 复制少量文件

不要复制 `data/`。

```bash
mkdir -p third_party/fashion-mnist
cp official_fashion_mnist/README.md third_party/fashion-mnist/
cp official_fashion_mnist/README.zh-CN.md third_party/fashion-mnist/
cp official_fashion_mnist/LICENSE third_party/fashion-mnist/
cp official_fashion_mnist/utils/mnist_reader.py third_party/fashion-mnist/
```

如果官方仓库没有 `README.zh-CN.md`，可以跳过这一项。

## 3. 为什么复制 `mnist_reader.py`

我们复制官方仓库中的 `mnist_reader.py`，不是为了直接使用它训练模型，而是为了学习官方代码如何组织原始 IDX 数据读取逻辑。

在本课程仓库中，最终训练统一使用 `torchvision.datasets.FashionMNIST`，原因是：

1. 它是 PyTorch 生态中的标准数据接口；
2. 它可以自动下载和缓存数据；
3. 它能与 `transforms` 和 `DataLoader` 无缝连接；
4. 它更适合课堂快速复现实验。

## 4. 阅读问题

1. README.md 中写了 Fashion-MNIST 有多少训练样本？
2. 每张图像的大小是多少？
3. 一共有多少类？
4. `data/fashion` 目录原本是用来放什么的？
5. `mnist_reader.py` 的输入是什么？
6. `mnist_reader.py` 返回什么？
7. 它为什么需要 `kind="train"` 或 `kind="t10k"`？
8. 它和 `torchvision.datasets.FashionMNIST` 有什么区别？
