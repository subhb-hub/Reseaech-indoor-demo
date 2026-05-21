# Task Card

## 课堂目标

完成一次最小科研闭环：

```text
论文搜索 -> 官方仓库 -> 官方 convnet.py -> PyTorch 迁移版 baseline -> toy innovation -> 消融实验 -> demo paper
```

## 必交内容

1. 论文和官方仓库信息记录。
2. `third_party/fashion-mnist/` 中复制的官方 README、LICENSE、`mnist_reader.py`、`benchmark/convnet.py`。
3. `templates/architecture_table.md` 中填写的官方 ConvNet 架构表。
4. `official_like_cnn` baseline 运行结果。
5. edge / noise 两组消融结果。
6. 一份 demo paper 草稿。

## 核心原则

官方 `benchmark/convnet.py` 是 baseline 来源；本仓库提供 PyTorch 迁移版 `OfficialLikeCNN`；训练数据统一使用 `torchvision.datasets.FashionMNIST`。
