# Write Demo Paper

## 1. 整理结果

运行三组实验：

```bash
python src/train.py --model official_like_cnn --variant baseline --epochs 2 --use-subset
python src/train.py --model official_like_cnn --variant edge --epochs 2 --use-subset
python src/train.py --model official_like_cnn --variant noise --epochs 2 --use-subset
```

把 `results_summary.csv` 中的结果整理到：

```text
templates/experiment_record.csv
```

## 2. 写作模板

使用：

```text
templates/demo_paper_template.tex
```

## 3. 必写内容

- Introduction: 为什么选择 Fashion-MNIST？
- Baseline: 官方 `benchmark/convnet.py` 和 PyTorch 迁移版 `OfficialLikeCNN` 的关系是什么？
- Method: edge channel 如何改变输入？
- Experiments: baseline / edge / noise 怎么对比？
- Results: 表格中的 accuracy 是多少？
- Discussion: 结果是否支持你的想法？

## 4. 诚实讨论

如果 edge 没有提升，不要说“方法失败”。可以讨论：

- subset 太小；
- epoch 太少；
- edge map 设计太简单；
- CNN 可能已经能从原图中学习边缘；
- negative control 的结果说明额外通道本身也可能影响训练。
