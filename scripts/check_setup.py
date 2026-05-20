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
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, "mps"):
        print(f"MPS available: {torch.backends.mps.is_available()}")

    from train_fashion import SimpleCNN, count_parameters

    model = SimpleCNN(in_channels=1)
    print(f"Baseline model parameters: {count_parameters(model)}")
    print("Setup check passed.")


if __name__ == "__main__":
    main()
