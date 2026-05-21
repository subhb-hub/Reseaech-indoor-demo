# Find Innovation Position

## 1. Baseline Source

Our baseline comes from the official Fashion-MNIST ConvNet:

```text
third_party/fashion-mnist/benchmark/convnet.py
```

The runnable PyTorch version is:

```text
src/model_official_like.py
```

## 2. Run Official-Like Baseline

```bash
python src/train.py --model official_like_cnn --variant baseline --epochs 2 --use-subset
```

## 3. Locate Data Transform

Open:

```text
src/data.py
```

Find:

```python
get_transform(variant)
```

Answer:

1. What transform is used by `baseline`?
2. What transform is used by `edge`?
3. What transform is used by `noise`?

## 4. Locate Model Input Channel

Open:

```text
src/model_official_like.py
```

Find:

```python
self.conv1 = nn.Conv2d(in_channels=in_channels, ...)
```

Answer:

1. Why does baseline use `in_channels=1`?
2. Why do edge/noise use `in_channels=2`?
3. Why does `fc1 = nn.Linear(7 * 7 * 64, 1024)` not change?

## 5. Implement Toy Innovation

Open:

```text
src/transforms_custom.py
```

Complete:

- `EdgeEnhancedTransform`
- `NoiseEnhancedTransform`

## 6. Run Ablation

```bash
python src/train.py --model official_like_cnn --variant baseline --epochs 2 --use-subset
python src/train.py --model official_like_cnn --variant edge --epochs 2 --use-subset
python src/train.py --model official_like_cnn --variant noise --epochs 2 --use-subset
```
