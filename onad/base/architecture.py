import abc
from typing import Dict, Any

import torch
from torch import nn


class Architecture(abc.ABC, nn.Module):
    """
    Abstract base class for defining neural network architectures.

    This class ensures that any architecture can be plugged into the online model.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Returns relevant parameters for the architecture.

        Ensures OnlineModel can access necessary attributes dynamically.
        """
        raise NotImplementedError
