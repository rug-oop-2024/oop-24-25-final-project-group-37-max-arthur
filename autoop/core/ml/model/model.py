
from abc import ABC, abstractmethod
from typing import Literal
from copy import deepcopy
import pickle
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.trainer import Trainer
from torch import Tensor, from_numpy
import numpy as np



# could add a method here that checks that model has been
# trained before we call predict
# apparently, pipeline is passing np arrays here.

class Model(ABC):
    def __init__(self):
        self._model_parameters: dict = {}
        self._type: Literal["regression", "classification"] = None

    @property
    def model_parameters(self):
        return deepcopy(self._model_parameters)
    
    @property
    def type(self) -> str:
        return self._type
    
    @type.setter
    def type(self, value: Literal["regression", "classification"]) -> None:
        if value not in ["regression", "classification"]:
            raise ValueError(
                f"Type has to be 'regression', or 'classification'. Got {value} instead."
                )
        self._type = value

    @abstractmethod
    def fit(
            self,
            observations: Tensor,
            labels: Tensor
    ) -> None:
        """
        Abstract method to fit the model to observations and labels.

        Args:
            observations (torch.Tensor): The training data for the model.
            labels (torch.Tensor): Target values corresponding to observations.
        """
        pass

    @abstractmethod
    def predict(self, observations: Tensor) -> Tensor:
        """
        Abstract method to predict from the trained model.

        Args:
            observations (torch.Tensor): The training data for the model.

        Returns:
            torch.Tensor: Tensor containing predictions.
        """
        pass

    def to_artifact(self, name: str) -> 'Artifact':
        # ? need to see here later if this is the best way to store it
        # same for where we currently store the Model name
        artifact_data = {
            "parameters": self.model_parameters,
        }
        
        serialized_data = pickle.dumps(artifact_data)
        
        asset_path = f"models/{name}.pkl"
        metadata = {
            "model_type": self.type,
            "parameter_count": len(self.model_parameters),
        }
        
        artifact = Artifact(
            name=name,
            data=serialized_data,
            asset_path=asset_path,
            type=self.__class__.__name__,
            metadata=metadata
        )
        
        return artifact

class ClassificationMixin(Model):
    def __init__(
            self,
            num_epochs: int=10,
            lr: float=0.001
    ) -> None:
        super().__init__()
        self._trainer = None
        self._num_epochs = num_epochs
        self._lr = lr

    def _populate_model_parameters(self):
        # Populate _model_parameters with references to torch parameters
        # These update automatically. Also add hyperparams.
        self._model_parameters = {
            "num_epochs": self._num_epochs,
            "lr": self._lr,
            **{name: param for name, param in self.named_parameters()}
        }
    
    def _set_trainer(self):
        """Initialize the trainer with necessary parameters."""
        if not self._trainer:
            self._trainer = Trainer(
                self,
                num_epochs=self._num_epochs,
                lr=self._lr,
                loss_fn=self._loss_fn
            )

