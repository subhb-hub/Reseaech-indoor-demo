# Official ConvNet Architecture Table

Read:

```text
third_party/fashion-mnist/benchmark/convnet.py
```

Then compare with:

```text
src/model_official_like.py
```

| Stage | Official TensorFlow code keyword | Input shape | Output shape | PyTorch equivalent |
| --- | --- | --- | --- | --- |
| Input reshape | `tf.reshape` | ? | ? | PyTorch tensor is `[B, C, H, W]` |
| Conv1 | `tf.layers.conv2d` | ? | ? | ? |
| Pool1 | `tf.layers.max_pooling2d` | ? | ? | ? |
| Conv2 | `tf.layers.conv2d` | ? | ? | ? |
| Pool2 | `tf.layers.max_pooling2d` | ? | ? | ? |
| Flatten | `tf.reshape` | ? | ? | ? |
| Dense | `tf.layers.dense` | ? | ? | ? |
| Dropout | `tf.layers.dropout` | ? | ? | ? |
| Logits | `tf.layers.dense` | ? | ? | ? |
