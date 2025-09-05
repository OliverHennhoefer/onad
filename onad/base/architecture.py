import abc
import random

import numpy as np
import torch
from torch import nn


class Architecture(abc.ABC, nn.Module):
    """
    Abstract base class for defining neural network architectures.

    This class ensures that any deep can be plugged into the online model.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        raise NotImplementedError

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
