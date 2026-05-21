# Fashion-MNIST Official Files

This directory is reserved for a small copied subset of the official Fashion-MNIST repository:

```text
https://github.com/zalandoresearch/fashion-mnist
```

Copy only:

- `README.md`
- `README.zh-CN.md`, if available
- `LICENSE`
- `utils/mnist_reader.py`
- `benchmark/convnet.py`

Do not copy the official `data/` directory.

In this classroom repository:

- `benchmark/convnet.py` is used for reading the official ConvNet baseline.
- `mnist_reader.py` is used for understanding the original IDX data reader.
- Training uses `torchvision.datasets.FashionMNIST`.
- The runnable baseline is a PyTorch re-implementation in `src/model_official_like.py`.

Run:

```bash
bash scripts/clone_official_repo.sh
```

The official repository uses the MIT License, so keep the copied `LICENSE` file beside the copied source files.
