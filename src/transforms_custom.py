import torch
from torchvision import transforms


class EdgeEnhancedTransform:
    """
    TODO:
    Implement an edge-enhanced input transform.

    Expected input:
        PIL image from Fashion-MNIST.

    Expected output:
        Tensor with shape [2, 28, 28].
        Channel 0: original grayscale image.
        Channel 1: edge map.
    """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        x = self.to_tensor(image)  # [1, 28, 28]

        # TODO 1:
        # Compute an edge map from x.
        # Hint: you may use a simple difference operator or Sobel-like filters.

        # TODO 2:
        # Concatenate original image and edge map.
        # Expected shape: [2, 28, 28]

        raise NotImplementedError("Please implement EdgeEnhancedTransform.")


class NoiseEnhancedTransform:
    """
    Negative-control variant.
    This is used to test whether adding an extra channel is always useful.
    """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        x = self.to_tensor(image)  # [1, 28, 28]

        # TODO:
        # Create random noise with the same shape as x.
        # Concatenate x and noise.
        # Expected shape: [2, 28, 28]

        raise NotImplementedError("Please implement NoiseEnhancedTransform.")
