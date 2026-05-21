from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_COPY = REPO_ROOT / "third_party" / "fashion-mnist"


def main() -> None:
    if not THIRD_PARTY_COPY.exists():
        print("Copied official files not found.")
        print("Run: bash scripts/clone_official_repo.sh")
        raise SystemExit(1)

    expected = [
        THIRD_PARTY_COPY / "README.md",
        THIRD_PARTY_COPY / "LICENSE",
        THIRD_PARTY_COPY / "mnist_reader.py",
        THIRD_PARTY_COPY / "benchmark" / "convnet.py",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        print("Some required copied files are missing:")
        for path in missing:
            print(f"- {path.relative_to(REPO_ROOT)}")
        raise SystemExit(1)

    optional = THIRD_PARTY_COPY / "README.zh-CN.md"
    print("Official Fashion-MNIST reading files are copied.")
    print("Students should read:")
    print("- third_party/fashion-mnist/README.md")
    print("- third_party/fashion-mnist/benchmark/convnet.py")
    print("- third_party/fashion-mnist/mnist_reader.py")
    print("Copied files:")
    for path in expected:
        print(f"- {path.relative_to(REPO_ROOT)}")
    if optional.exists():
        print(f"- {optional.relative_to(REPO_ROOT)}")
    print("Do not copy third_party/fashion-mnist/data/. Training uses torchvision.datasets.FashionMNIST.")


if __name__ == "__main__":
    main()
