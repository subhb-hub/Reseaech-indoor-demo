import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = REPO_ROOT / "external" / "fashion-mnist"


def main() -> None:
    if not OFFICIAL_REPO.exists():
        print("Official repo not found.")
        print("Run: bash scripts/clone_official_repo.sh")
        raise SystemExit(1)

    expected = [
        OFFICIAL_REPO / "README.md",
        OFFICIAL_REPO / "utils" / "mnist_reader.py",
        OFFICIAL_REPO / "data" / "fashion",
        OFFICIAL_REPO / "benchmark",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        print("Official repo was cloned, but some expected paths are missing:")
        for path in missing:
            print(f"- {path.relative_to(REPO_ROOT)}")
        raise SystemExit(1)

    sys.path.insert(0, str(OFFICIAL_REPO / "utils"))
    import mnist_reader

    x_train, y_train = mnist_reader.load_mnist(str(OFFICIAL_REPO / "data" / "fashion"), kind="train")
    x_test, y_test = mnist_reader.load_mnist(str(OFFICIAL_REPO / "data" / "fashion"), kind="t10k")

    print("Official Fashion-MNIST clone looks good.")
    print(f"Train images: {x_train.shape}, train labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}, test labels: {y_test.shape}")
    print("Next classroom step: use this official data context to reproduce the PyTorch baseline.")


if __name__ == "__main__":
    main()
