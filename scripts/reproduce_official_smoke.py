import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = REPO_ROOT / "external" / "fashion-mnist"


def parse_args():
    parser = argparse.ArgumentParser(description="Small CPU smoke reproduction using the official Fashion-MNIST repo.")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utils_dir = OFFICIAL_REPO / "utils"
    data_dir = OFFICIAL_REPO / "data" / "fashion"

    if not utils_dir.exists() or not data_dir.exists():
        print("Official Fashion-MNIST repo/data not found.")
        print("Run: bash scripts/clone_official_repo.sh")
        raise SystemExit(1)

    sys.path.insert(0, str(utils_dir))
    import mnist_reader

    x_train, y_train = mnist_reader.load_mnist(str(data_dir), kind="train")
    x_test, y_test = mnist_reader.load_mnist(str(data_dir), kind="t10k")

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(x_train), size=min(args.train_size, len(x_train)), replace=False)
    test_idx = rng.choice(len(x_test), size=min(args.test_size, len(x_test)), replace=False)

    x_train = x_train[train_idx].astype("float32") / 255.0
    y_train = y_train[train_idx]
    x_test = x_test[test_idx].astype("float32") / 255.0
    y_test = y_test[test_idx]

    clf = SGDClassifier(loss="log_loss", max_iter=20, random_state=args.seed, n_jobs=-1)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    print("Official-repo smoke reproduction finished.")
    print(f"Train subset: {len(x_train)}")
    print(f"Test subset: {len(x_test)}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print("This is only a fast reproduction checkpoint, not the final course baseline.")


if __name__ == "__main__":
    main()
