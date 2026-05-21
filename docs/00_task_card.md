# Task Card

## 课堂目标

完成一次最小科研闭环：

```text
论文搜索 -> 官方仓库 -> 复制/迁移代码 -> 读懂后改进 -> 消融实验 -> demo paper
```

## 必交内容

1. 论文和官方仓库信息记录。
2. `third_party/fashion-mnist/` 中复制的官方 README、LICENSE、`mnist_reader.py`。
3. baseline 运行结果。
4. edge / noise 两组消融结果。
5. 一份 demo paper 草稿。

## 核心原则

官方仓库用于阅读和理解；训练代码统一使用 `torchvision.datasets.FashionMNIST`。
