import torch
from torch import nn

from onad.base.architecture import Architecture


class VanillaAutoencoder(Architecture):
    def __init__(self, input_size: int, seed: int = 1):
        super().__init__()

        self._set_seed(seed) if seed is not None else None

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @property
    def input_size(self) -> int:
        return self.encoder[0].in_features


class VanillaLSTMAutoencoder(Architecture):
    def __init__(self, input_size: int, seed: int = 1):
        super().__init__()

        self._set_seed(seed) if seed is not None else None

        self.encoder = nn.LSTM(input_size, 64, batch_first=True)
        self.hidden_to_latent = nn.Linear(64, 16)

        self.latent_to_hidden = nn.Linear(16, 64)
        self.decoder = nn.LSTM(64, input_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        latent = self.hidden_to_latent(h_n.squeeze(0))
        hidden = self.latent_to_hidden(latent).unsqueeze(0)
        decoded, _ = self.decoder(hidden)
        return decoded

    @property
    def input_size(self) -> int:
        return self.encoder[0].in_features
