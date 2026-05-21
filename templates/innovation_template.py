"""
Use this file to draft your toy innovation before editing src/transforms_custom.py.

Research question:
    Can an extra edge channel help a small CNN classify Fashion-MNIST images?

Expected input:
    PIL image from torchvision.datasets.FashionMNIST.

Expected output:
    Tensor with shape [2, 28, 28].
"""

import torch
from torchvision import transforms


class MyInnovationTransform:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        x = self.to_tensor(image)  # [1, 28, 28]

        # TODO:
        # 1. Create your extra channel.
        # 2. Make sure it has shape [1, 28, 28].
        # 3. Concatenate it with x.

        extra = torch.zeros_like(x)
        return torch.cat([x, extra], dim=0)
