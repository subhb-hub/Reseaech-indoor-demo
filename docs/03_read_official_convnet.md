# Read The Official ConvNet

## Goal

In this task, you will read the official Fashion-MNIST ConvNet implementation and understand how our PyTorch baseline is derived from it.

Official file copied to:

```text
third_party/fashion-mnist/benchmark/convnet.py
```

Our PyTorch version:

```text
src/model_official_like.py
```

## Step 1. Locate The Official Model Function

Open:

```text
third_party/fashion-mnist/benchmark/convnet.py
```

Find:

```python
def cnn_model_fn(features, labels, mode):
```

## Step 2. Find The Model Components

Find the following components:

1. input reshape
2. first convolution
3. first max pooling
4. second convolution
5. second max pooling
6. flatten
7. dense layer
8. dropout
9. logits layer
10. loss and optimizer

## Step 3. Fill The Architecture Table

Open:

```text
templates/architecture_table.md
```

Fill in the shapes and PyTorch equivalents.

## Step 4. Compare With PyTorch Version

Open:

```text
src/model_official_like.py
```

Answer:

1. Which line corresponds to the first official Conv2D layer?
2. Which line corresponds to the second official Conv2D layer?
3. Why is `padding=2` used for a 5x5 convolution?
4. Why is `fc1` input size `7 * 7 * 64`?
5. If we add an edge channel, which line must change?
