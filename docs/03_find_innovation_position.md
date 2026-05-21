# Find Innovation Position

## 1. 先运行 baseline

```bash
python src/train.py --variant baseline --epochs 2 --use-subset
```

## 2. 阅读四个文件

- `src/data.py`
- `src/transforms_custom.py`
- `src/model_baseline.py`
- `src/train.py`

## 3. 定位数据入口

在 `src/data.py` 中找到：

```python
datasets.FashionMNIST(...)
```

回答：

1. `root` 参数控制什么？
2. `train=True` 和 `train=False` 有什么区别？
3. `transform` 是在哪里传入的？
4. `download=True` 的作用是什么？

## 4. 定位创新位置

在 `src/transforms_custom.py` 中补完 TODO。

目标：

```text
baseline: [1, 28, 28]
edge:    [2, 28, 28]
noise:   [2, 28, 28]
```

## 5. 定位模型修改点

在 `src/model_baseline.py` 中找到：

```python
self.conv1 = nn.Conv2d(in_channels=in_channels, ...)
```

回答：

1. baseline 的 `in_channels` 应该是多少？
2. edge/noise 的 `in_channels` 应该是多少？
3. 为什么只改 transform 还不够？

## 6. 定位训练脚本连接点

在 `src/train.py` 中找到：

```python
if args.variant == "baseline":
    in_channels = 1
elif args.variant in ["edge", "noise"]:
    in_channels = 2
```

这说明创新会牵动数据输入、模型结构和训练脚本，而不是只改一个函数。
