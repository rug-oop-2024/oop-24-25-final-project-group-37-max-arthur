
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import torch
from copy import deepcopy
from typing import Literal

class Model(ABC):
    def __init__(self):
        self._parameters: dict = {}
        self._hyper_parameters: dict = {}
        self.type: Literal["regression", "classification"] = None  # ?do we need a setter for this here??

    @property
    def parameters(self):
        return deepcopy(self._parameters)
    
    @property
    def hyper_parameters(self):
        return deepcopy(self._hyper_parameters)
    
    @abstractmethod
    def to_artifact(self, name: str) -> 'Artifact':
        pass


    @abstractmethod
    def fit(
            self,
            observations: torch.Tensor,
            labels: torch.Tensor
    ) -> None:
        """
        Abstract method to fit the model to observations and labels.

        Args:
            observations (torch.Tensor): The training data for the model.
            labels (torch.Tensor): Target values corresponding to observations.
        """
        pass

    @abstractmethod
    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to predict from the trained model.

        Args:
            observations (torch.Tensor): The training data for the model.

        Returns:
            torch.Tensor: Tensor containing predictions.
        """
        pass