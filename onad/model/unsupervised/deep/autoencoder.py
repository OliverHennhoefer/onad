from typing import Dict

import torch
from torch import optim

from onad.base.architecture import Architecture
from onad.base.model import BaseModel
from onad.utils.architecture.loss_fnc import LossFunction


class Autoencoder(BaseModel):
    """
    Autoencoder model for unsupervised anomaly detection.
    
    This class implements an autoencoder that learns to reconstruct input data
    and uses reconstruction error as an anomaly score. Higher reconstruction
    errors indicate greater anomaly likelihood.
    
    Attributes:
        model (Architecture): The neural network architecture.
        criterion (LossFunction): Loss function for training.
        optimizer (optim.Optimizer): Optimizer for training.
    """
    def __init__(
        self,
        model: Architecture,
        optimizer: optim.Optimizer,
        criterion: LossFunction
    ) -> None:
        """
        Initialize the Autoencoder model.
        
        Args:
            model (Architecture): The neural network architecture.
            optimizer (optim.Optimizer): Optimizer for training.
            criterion (LossFunction): Loss function for training.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # Preallocate tensor for input data to avoid repeated allocations
        self.x_tensor = torch.empty(1, self.model.input_size, dtype=torch.float32)

    def _dict_to_tensor(self, x: Dict[str, float]) -> None:
        """
        Convert dictionary to tensor, reusing pre-allocated tensor.
        
        Args:
            x (Dict[str, float]): Input dictionary with feature values.
            
        Raises:
            ValueError: If input size doesn't match model's expected input size.
        """
        values = list(x.values())
        if len(values) != self.model.input_size:
            raise ValueError(
                f"Input size {len(values)} doesn't match model input size {self.model.input_size}"
            )
        
        # Reuse pre-allocated tensor instead of creating new one
        self.x_tensor[0] = torch.tensor(values, dtype=torch.float32)

    def learn_one(self, x: Dict[str, float]) -> None:
        """
        Update the model with a single data point.
        
        Args:
            x (Dict[str, float]): Input dictionary with feature names as keys
                and feature values as values.
        """
        self._dict_to_tensor(x)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(self.x_tensor)
        loss = self.criterion(output, self.x_tensor)
        loss.backward()
        self.optimizer.step()

    def score_one(self, x: Dict[str, float]) -> float:
        """
        Compute the anomaly score for a single data point.
        
        Args:
            x (Dict[str, float]): Input dictionary with feature names as keys
                and feature values as values.
                
        Returns:
            float: Reconstruction error (anomaly score). Higher values indicate
                greater anomaly likelihood.
        """
        self._dict_to_tensor(x)
        with torch.no_grad():
            output = self.model(self.x_tensor)
            loss = self.criterion(output, self.x_tensor)
        return loss.item()
