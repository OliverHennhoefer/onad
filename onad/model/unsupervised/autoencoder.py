import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional
from onad.base.model import BaseModel


class AutoencoderModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list, latent_size: int, dropout: float):
        super().__init__()

        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, latent_size))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, in_features),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features, input_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Autoencoder(BaseModel):
    def __init__(
            self,
            hidden_sizes: list,
            latent_size: int,
            learning_rate: float = 0.001,
            dropout: float = 0.1,
            warmup: int = 0,
            seed: Optional[int] = None,
    ):
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.warmup = warmup
        self.seed = seed
        self.feature_names = None
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        if self.seed is not None:
            self._set_seed(seed)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _initialize_model(self, x: Dict[str, float]):
        self.feature_names = sorted(x.keys())
        input_size = len(self.feature_names)
        self.model = AutoencoderModel(
            input_size,
            self.hidden_sizes,
            self.latent_size,
            self.dropout,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _convert_x_to_tensor(self, x: Dict[str, float]) -> torch.Tensor:
        if sorted(x.keys()) != self.feature_names:
            raise ValueError("Features do not match initial features")
        values = [x[key] for key in self.feature_names]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    def learn_one(self, x: Dict[str, float]) -> None:
        if self.feature_names is None:
            self._initialize_model(x)
        x_tensor = self._convert_x_to_tensor(x)
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(x_tensor)
        loss = self.criterion(output, x_tensor)
        loss.backward()
        self.optimizer.step()

    def score_one(self, x: Dict[str, float]) -> Optional[float]:
        if self.feature_names is None:
            self._initialize_model(x)

        x_tensor = self._convert_x_to_tensor(x)
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_tensor)
            loss = self.criterion(output, x_tensor)
        return loss.item()
