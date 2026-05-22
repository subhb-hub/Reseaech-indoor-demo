# Innovation Guide: Edge-Enhanced Fashion-CNN

## 0. 这一步要解决什么问题？

前面几步我们已经完成了：

```text
找论文 -> 找官方仓库 -> 阅读官方 convnet.py -> 迁移得到 PyTorch baseline
```

现在要做的是科研流程里的下一步：

```text
从 baseline 出发，提出一个小而清楚的改动，并用对照实验验证它
```

本课堂的 toy innovation 是：

```text
Edge-Enhanced Fashion-CNN
```

一句话解释：

> 在原始灰度图之外，额外给模型一张边缘图，让 CNN 同时看到“像素外观”和“轮廓结构”。

注意：这是教学型创新，不是正式论文贡献。它的价值在于训练学生如何提出假设、修改代码、设计对照实验、解释结果。

---

## 1. 创新点到底是什么？

Fashion-MNIST 原始输入是一张灰度图：

```text
baseline input: [1, 28, 28]
```

其中：

- `1` 表示 1 个灰度通道；
- `28, 28` 表示图像高和宽。

我们的创新是：为每张图片再生成一张边缘图，然后和原图拼在一起：

```text
edge input: [2, 28, 28]
```

其中：

```text
channel 0: original grayscale image
channel 1: edge map
```

所以这个创新不是改整个网络，也不是换一个大模型，而是改输入表示：

```text
Original image -> Original image + Edge map
```

---

## 2. 为什么这个想法有道理？

Fashion-MNIST 的类别包括衣服、鞋、包等物体。很多类别之间的区别不只在像素亮度，也在轮廓形状。

例如：

| 类别对 | 可能依赖的视觉线索 |
|---|---|
| Sneaker vs. Sandal | 鞋底、开口、边缘结构 |
| Bag vs. T-shirt/top | 整体外形轮廓 |
| Coat vs. Shirt vs. Pullover | 袖子、领口、衣摆轮廓 |

因此我们提出一个假设：

> 如果模型能直接看到边缘信息，它可能更容易学习某些服饰类别的轮廓差异。

这个假设不一定成立。课堂实验的重点正是：不要只讲故事，要设计实验去检查它。

---

## 3. 边缘图的原理是什么？

边缘图的目标是突出图像中变化明显的位置。

直观理解：

- 如果相邻像素变化很小，这里可能是平坦区域；
- 如果相邻像素变化很大，这里可能是物体边界或纹理边缘。

常见做法是使用 Sobel filter。给定原图：

```text
X: [1, 28, 28]
```

可以计算水平方向和垂直方向的变化：

```text
Gx = horizontal edge response
Gy = vertical edge response
```

然后合成边缘强度：

```text
E = sqrt(Gx^2 + Gy^2)
```

最后把 `E` 缩放到稳定范围，例如 `[0, 1]`，再和原图拼接：

```text
X_edge = concat(X, E)
```

课堂实现时不需要追求最复杂的图像处理算法。只要得到一张和原图同尺寸的边缘提示图即可。

---

## 4. 它和官方 baseline 的关系是什么？

官方 Fashion-MNIST baseline 来源：

```text
third_party/fashion-mnist/benchmark/convnet.py
```

本仓库中的 PyTorch 迁移版：

```text
src/model_official_like.py
```

baseline 的输入是：

```text
[batch_size, 1, 28, 28]
```

edge 版本的输入是：

```text
[batch_size, 2, 28, 28]
```

因此模型里真正需要变化的是第一层卷积的输入通道数：

```python
self.conv1 = nn.Conv2d(in_channels=in_channels, ...)
```

但后面的结构不需要改：

```text
conv2 不变
pooling 不变
fc1 = nn.Linear(7 * 7 * 64, 1024) 不变
fc2 不变
```

原因是：第一层卷积之后，输出通道数仍然是 32；后续 feature map 的空间尺寸和通道数仍然按原来的网络流动。

---

## 5. 课堂实验中要改哪里？

### 5.1 数据入口

打开：

```text
src/data.py
```

找到：

```python
get_transform(variant)
```

它决定不同实验版本使用什么输入变换：

| variant | transform | 输出形状 |
|---|---|---|
| `baseline` | `transforms.ToTensor()` | `[1, 28, 28]` |
| `edge` | `EdgeEnhancedTransform()` | `[2, 28, 28]` |
| `noise` | `NoiseEnhancedTransform()` | `[2, 28, 28]` |

### 5.2 创新实现位置

打开：

```text
src/transforms_custom.py
```

需要完成两个类：

```python
EdgeEnhancedTransform
NoiseEnhancedTransform
```

不要改训练循环，不要先改模型主体。先把输入变换做对。

### 5.3 模型输入通道

打开：

```text
src/train.py
```

找到：

```python
if args.variant == "baseline":
    in_channels = 1
elif args.variant in ["edge", "noise"]:
    in_channels = 2
```

这说明：

- baseline 使用 1 通道输入；
- edge/noise 使用 2 通道输入；
- 模型类不需要写三份，只需要通过 `in_channels` 参数控制第一层。

---

## 6. EdgeEnhancedTransform 怎么做？

目标：

```text
PIL image -> [2, 28, 28] tensor
```

建议步骤：

1. 用 `transforms.ToTensor()` 把图片变成 `x`。
2. 从 `x` 计算一张边缘图 `edge`。
3. 确保 `edge` 的形状是 `[1, 28, 28]`。
4. 把 `x` 和 `edge` 在 channel 维度拼接。
5. 返回 `[2, 28, 28]`。

实现提示：

```text
x shape:    [1, 28, 28]
edge shape: [1, 28, 28]
output:     torch.cat([x, edge], dim=0)
```

边缘图可以用 Sobel-like 思路：

```text
1. 准备水平/垂直方向 filter
2. 对 x 做卷积
3. 合成 edge magnitude
4. 做归一化，避免数值过大
```

不要急着追求漂亮代码，先确保输出 shape 正确。

---

## 7. 为什么还要做 Noise-Control？

如果只比较：

```text
baseline vs. edge
```

即使 edge 变好了，也不能马上说“边缘信息有效”。

因为 edge 版本有两个变化：

1. 输入多了一个通道；
2. 这个通道是边缘信息。

我们需要区分：

```text
提升来自有意义的边缘？
还是只是因为输入通道变多？
```

所以加入 negative control：

```text
noise input: [original image, random noise]
```

它和 edge 一样也是 2 通道，但第二个通道没有语义信息。

这样实验更公平：

| 实验 | 输入 | 目的 |
|---|---|---|
| baseline | 原图 | 基础对照 |
| edge | 原图 + 边缘图 | 检查边缘信息是否有帮助 |
| noise | 原图 + 随机噪声 | 检查提升是否只是因为多了一个通道 |

---

## 8. NoiseEnhancedTransform 怎么做？

目标：

```text
PIL image -> [2, 28, 28] tensor
```

建议步骤：

1. 用 `transforms.ToTensor()` 得到 `x`。
2. 生成一张和 `x` 形状相同的随机噪声图。
3. 拼接 `x` 和 `noise`。
4. 返回 `[2, 28, 28]`。

实现提示：

```text
x shape:     [1, 28, 28]
noise shape: [1, 28, 28]
output:      [2, 28, 28]
```

注意：noise control 的目的不是提高准确率，而是帮助我们判断 edge 的解释是否可信。

---

## 9. 运行实验

先跑 baseline：

```bash
python src/train.py --model official_like_cnn --variant baseline --epochs 2 --use-subset
```

完成 `EdgeEnhancedTransform` 后跑：

```bash
python src/train.py --model official_like_cnn --variant edge --epochs 2 --use-subset
```

完成 `NoiseEnhancedTransform` 后跑：

```bash
python src/train.py --model official_like_cnn --variant noise --epochs 2 --use-subset
```

课堂上建议使用 `--use-subset`

---

## 11. 学生自查清单

实现完成后，检查这些问题：

1. `baseline` 的 transform 输出是不是 `[1, 28, 28]`？
2. `edge` 的 transform 输出是不是 `[2, 28, 28]`？
3. `noise` 的 transform 输出是不是 `[2, 28, 28]`？
4. `edge/noise` 是否触发 `in_channels=2`？
5. `fc1 = nn.Linear(7 * 7 * 64, 1024)` 为什么不需要改？
6. `results_summary.csv` 里是否有三行结果？
7. 你的结论有没有同时比较 baseline、edge 和 noise？

---

## 12. 课堂报告模板

可以按这个顺序汇报：

1. 我们复现了官方风格 PyTorch baseline。
2. 我们观察到 Fashion-MNIST 中部分类别可能依赖轮廓。
3. 因此我们提出 Edge-Enhanced input，把原图和边缘图拼成 2 通道输入。
4. 为了避免过度解释，我们加入 Noise-Control。
5. 三组实验结果分别是：baseline、edge、noise。
6. 根据结果，我们谨慎地得出结论：这个 toy idea 是否在当前实验中显示出初步价值。
