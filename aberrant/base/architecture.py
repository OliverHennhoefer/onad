"""Neural network architecture base class for deep learning models."""

import abc
import random

import numpy as np
import torch
from torch import nn


class Architecture(abc.ABC, nn.Module):
    """
    Abstract base class for defining neural network architectures.

    This class ensures that any neural network architecture can be plugged into
    online anomaly detection models. It provides a consistent interface and
    device handling capabilities.

    Subclasses must implement the `forward` and `input_size` methods.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        """
        Initialize the architecture.

        Args:
            device: The device to run the model on. If None, uses CPU.
        """
        super().__init__()
        self.device = device or torch.device("cpu")
        self.to(self.device)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        """
        The expected input size for the network.

        Returns:
            Number of input features.
        """
        raise NotImplementedError

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set random seeds for reproducibility.

        Args:
            seed: Random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __repr__(self) -> str:
        """Return a string representation of the architecture."""
        return f"{self.__class__.__name__}(input_size={self.input_size}, device={self.device})"
