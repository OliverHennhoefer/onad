"""Neural network architectures for autoencoder models."""

import torch
from torch import nn

from onad.base.architecture import Architecture


class VanillaAutoencoder(Architecture):
    """
    Simple feedforward autoencoder with ReLU activations.

    Architecture: input -> 64 -> 32 -> 16 -> 32 -> 64 -> output

    Args:
        input_size: Number of input features.
        seed: Random seed for reproducibility (optional).
    """

    def __init__(self, input_size: int, seed: int | None = None) -> None:
        super().__init__()

        if seed is not None:
            self.set_seed(seed)

        self._input_size = input_size

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
        """Forward pass through encoder and decoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @property
    def input_size(self) -> int:
        """Number of input features."""
        return self._input_size


class VanillaLSTMAutoencoder(Architecture):
    """
    LSTM-based autoencoder for sequence data.

    Args:
        input_size: Number of input features.
        seed: Random seed for reproducibility (optional).
    """

    def __init__(self, input_size: int, seed: int | None = None) -> None:
        super().__init__()

        if seed is not None:
            self.set_seed(seed)

        self._input_size = input_size

        self.encoder = nn.LSTM(input_size, 64, batch_first=True)
        self.hidden_to_latent = nn.Linear(64, 16)

        self.latent_to_hidden = nn.Linear(16, 64)
        self.decoder = nn.LSTM(64, input_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM encoder and decoder."""
        _, (h_n, _) = self.encoder(x)
        latent = self.hidden_to_latent(h_n.squeeze(0))
        hidden = self.latent_to_hidden(latent).unsqueeze(0)
        decoded, _ = self.decoder(hidden)
        return decoded

    @property
    def input_size(self) -> int:
        """Number of input features."""
        return self._input_size
