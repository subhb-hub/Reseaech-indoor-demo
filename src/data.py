from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch


def get_transform(variant: str):
    if variant == "baseline":
        return transforms.ToTensor()

    if variant == "edge":
        from src.transforms_custom import EdgeEnhancedTransform

        return EdgeEnhancedTransform()

    if variant == "noise":
        from src.transforms_custom import NoiseEnhancedTransform

        return NoiseEnhancedTransform()

    raise ValueError(f"Unknown variant: {variant}")


def get_loaders(
    variant="baseline",
    data_root="./data",
    batch_size=64,
    use_subset=True,
    train_subset_size=5000,
    test_subset_size=1000,
    download=True,
    num_workers=0,
):
    transform = get_transform(variant)

    train_set = datasets.FashionMNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=download,
    )

    test_set = datasets.FashionMNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=download,
    )

    if use_subset:
        generator = torch.Generator().manual_seed(42)

        train_indices = torch.randperm(len(train_set), generator=generator)[:train_subset_size]
        test_indices = torch.randperm(len(test_set), generator=generator)[:test_subset_size]

        train_set = Subset(train_set, train_indices)
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
