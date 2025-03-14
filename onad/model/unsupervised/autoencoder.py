import random
from typing import Dict

import numpy as np
import torch
from torch import nn, optim

from onad.base.architecture import Architecture
from onad.base.model import BaseModel


class Autoencoder(BaseModel):
    def __init__(
        self, model: Architecture, learning_rate: float = 0.001, seed: int = 1
    ):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=0
        )

        # Preallocate tensor size for input data based on expected structure
        self.x_tensor = torch.empty(
            1, self.model.input_size, dtype=torch.float32
        )

    def learn_one(self, x: Dict[str, float]) -> None:
        # Directly load data into the pre-allocated tensor
        self.x_tensor[0] = torch.tensor(list(x.values()), dtype=torch.float32)
        self.optimizer.zero_grad(set_to_none=True)  # In-place zeroing of gradients
        output = self.model(self.x_tensor)
        loss = self.criterion(output, self.x_tensor)
        loss.backward()
        self.optimizer.step()

    def score_one(self, x: Dict[str, float]) -> float:
        # Directly load data into the pre-allocated tensor
        self.x_tensor[0] = torch.tensor(list(x.values()), dtype=torch.float32)
        with torch.no_grad():
            output = self.model(self.x_tensor)
            loss = self.criterion(output, self.x_tensor)
        return loss.item()
