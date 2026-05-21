import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    print(f"Python: {platform.python_version()}")

    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}")
        print("Run: pip install -r requirements.txt")
        raise SystemExit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print("Device policy: CPU only")

    from src.model_baseline import SimpleCNN, count_parameters
    from src.model_official_like import OfficialLikeCNN

    warmup = SimpleCNN(in_channels=1)
    official_like = OfficialLikeCNN(in_channels=1)
    print(f"Warm-up SimpleCNN parameters: {count_parameters(warmup)}")
    print(f"OfficialLikeCNN baseline parameters: {count_parameters(official_like)}")
    print("Setup check passed.")


if __name__ == "__main__":
    main()
