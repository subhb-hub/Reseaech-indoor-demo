import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialLikeCNN(nn.Module):
    """
    PyTorch re-implementation of the official Fashion-MNIST ConvNet baseline.

    Official source for reading:
        third_party/fashion-mnist/benchmark/convnet.py

    Correspondence:
        TensorFlow input [B, 28, 28, 1]
        -> PyTorch input [B, 1, 28, 28]

        Conv2D filters=32, kernel_size=5, padding="same"
        -> nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)

        Conv2D filters=64, kernel_size=5, padding="same"
        -> nn.Conv2d(32, 64, kernel_size=5, padding=2)

        Dense units=1024
        -> nn.Linear(7 * 7 * 64, 1024)

        Logits units=10
        -> nn.Linear(1024, 10)
    """

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=5,
            padding=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]
        x = torch.flatten(x, 1)  # [B, 7*7*64]
        x = F.relu(self.fc1(x))  # [B, 1024]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 10]
        return x
