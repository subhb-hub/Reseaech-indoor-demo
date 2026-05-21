import argparse
import csv
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_loaders
from src.evaluate import evaluate
from src.model_baseline import SimpleCNN, count_parameters
from src.model_official_like import OfficialLikeCNN


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name != "cpu":
        raise ValueError("This classroom repo is configured to run on CPU only. Use --device cpu.")
    return torch.device("cpu")


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


def build_model(model_name: str, in_channels: int):
    if model_name == "simple_cnn":
        return SimpleCNN(in_channels=in_channels, num_classes=10)

    if model_name == "official_like_cnn":
        return OfficialLikeCNN(in_channels=in_channels, num_classes=10)

    raise ValueError(f"Unknown model: {model_name}")


def append_result(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST baseline and toy research variants.")
    parser.add_argument(
        "--model",
        type=str,
        default="official_like_cnn",
        choices=["simple_cnn", "official_like_cnn"],
        help="Model architecture. official_like_cnn is the PyTorch version of the official convnet.py baseline.",
    )
    parser.add_argument("--variant", type=str, default="baseline", choices=["baseline", "edge", "noise"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--results-csv", type=str, default="./results_summary.csv")
    parser.add_argument("--train-subset-size", type=int, default=5000)
    parser.add_argument("--test-subset-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-subset", dest="use_subset", action="store_true", default=True)
    parser.add_argument("--no-subset", dest="use_subset", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "baseline":
        in_channels = 1
    elif args.variant in ["edge", "noise"]:
        in_channels = 2
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    train_loader, test_loader = get_loaders(
        variant=args.variant,
        data_root=args.data_root,
        batch_size=args.batch_size,
        use_subset=args.use_subset,
        train_subset_size=args.train_subset_size,
        test_subset_size=args.test_subset_size,
        download=args.download,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Model: {args.model}")
    print(f"Variant: {args.variant}")
    print(f"Device: {device}")
    print(f"Input channels: {in_channels}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
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

    model_path = output_dir / f"model_{args.model}_{args.variant}.pt"
    torch.save(model.state_dict(), model_path)

    result_row = {
        "model": args.model,
        "variant": args.variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "use_subset": args.use_subset,
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "input_channels": in_channels,
        "params": count_parameters(model),
        "best_test_acc": round(best_acc, 6),
        "model_path": str(model_path),
    }
    append_result(Path(args.results_csv), result_row)
    print(f"Saved result to {args.results_csv}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
