import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name != "cpu":
        raise ValueError("This classroom repo is configured to run on CPU only. Use --device cpu.")
    return torch.device("cpu")


def sobel_edge(x: torch.Tensor) -> torch.Tensor:
    """
    TODO for class:
    Implement Sobel edge extraction for a tensor with shape [1, H, W].

    Hints:
    1. Define horizontal and vertical Sobel kernels.
    2. Use F.conv2d with padding=1.
    3. Combine the two responses with sqrt(gx ** 2 + gy ** 2).
    4. Normalize the edge map before returning it.
    """
    raise NotImplementedError("课堂练习：请在这里实现 Sobel edge map。")


class ExtraChannelDataset(Dataset):
    """TODO for class: add an edge or random-noise channel to the base dataset."""

    def __init__(self, base_dataset: Dataset, mode: str):
        if mode not in {"edge", "noise"}:
            raise ValueError("mode must be 'edge' or 'noise'")
        self.base_dataset = base_dataset
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        if self.mode == "edge":
            extra = sobel_edge(x)
        else:
            # TODO for class: replace this line with a random tensor shaped like x.
            raise NotImplementedError("课堂练习：请在这里实现 random-noise control。")
        return torch.cat([x, extra], dim=0), y


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="test", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def append_result(csv_path: str, row: dict) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_datasets(args):
    transform = transforms.ToTensor()
    train_base = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_base = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_base = maybe_limit_dataset(train_base, args.limit_train)
    test_base = maybe_limit_dataset(test_base, args.limit_test)

    if args.variant in {"edge", "noise"}:
        print("This variant is intentionally left as a classroom TODO.")
        print("First reproduce the baseline, then implement ExtraChannelDataset during class.")
        return ExtraChannelDataset(train_base, args.variant), ExtraChannelDataset(test_base, args.variant), 2
    return train_base, test_base, 1


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST baseline and toy research variants.")
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "edge", "noise"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None, help="Use a small subset for smoke tests.")
    parser.add_argument("--limit-test", type=int, default=None, help="Use a small test subset for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Variant: {args.variant}")
    print(f"Device: {device}")

    train_dataset, test_dataset, in_channels = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SimpleCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Input channels: {in_channels}")
    print(f"Trainable parameters: {count_parameters(model)}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        best_acc = max(best_acc, test_acc)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} | test acc {test_acc:.4f}"
        )

    model_path = os.path.join(args.output_dir, f"model_{args.variant}.pt")
    torch.save(model.state_dict(), model_path)

    result_row = {
        "variant": args.variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "input_channels": in_channels,
        "params": count_parameters(model),
        "best_test_acc": round(best_acc, 6),
        "model_path": model_path,
    }
    append_result("results_summary.csv", result_row)
    print("Saved result to results_summary.csv")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
