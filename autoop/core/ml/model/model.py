
from abc import ABC, abstractmethod
from typing import Literal, Any, Union
from sklearn.base import BaseEstimator
from copy import deepcopy
import pickle
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.trainer import Trainer
import numpy as np
from torch import Tensor
from torch.nn import Linear, ModuleList


# could add a method here that checks that model has been
# trained before we call predict
# apparently, pipeline is passing np arrays here.

class Model(ABC):
    def __init__(self):
        self._parameters: dict = {}
        self._type: Literal["regression", "classification"] = None

    @property
    @abstractmethod
    def parameters(self) -> dict:
        pass

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
            observations: np.ndarray,
            labels: np.ndarray
    ) -> None:
        """
        Abstract method to fit the model to observations and labels.

        Args:
            observations (torch.Tensor): The training data for the model.
            labels (torch.Tensor): Target values corresponding to observations.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> Tensor:
        """
        Abstract method to predict from the trained model.

        Args:
            observations (torch.Tensor): The training data for the model.

        Returns:
            torch.Tensor: Tensor containing predictions.
        """
        pass

    def to_artifact(self, name: str) -> 'Artifact':
        serialized_data = pickle.dumps(self)

        asset_path = f"models/{name}.pkl"
        metadata = {
            "model_type": self.type,
            "parameter_count": len(self.parameters),
        }
        artifact = Artifact(
            name=name,
            data=serialized_data,
            asset_path=asset_path,
            type=self.__class__.__name__,
            metadata=metadata
        )
        return artifact


class RegressionModel(Model):
    def __init__(self) -> None:
        self.type = "regression"

    @property
    def model(self) -> Union[BaseEstimator, Any]:
        return deepcopy(self._model)

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        if labels.ndim == 2:
            if labels.shape[1] == 1:
                labels = labels.squeeze(1)
            else:
                raise ValueError(
                    "Expected labels of shape [B, 1] or [B]"
                    f"but got {labels.shape} instead."
                    )
        assert labels.shape[0] == observations.shape[0], (
            "Observations and labels must have the same number of samples. "
            f"Got {labels.shape[0]} and {observations.shape[0]} instead."
        )
        self._model.fit(observations, labels)

    def predict(self, observations: np.ndarray) -> Tensor:
        predictions = self._model.predict(observations)
        return Tensor(predictions)

class GradientModel(Model):
    def __init__(
            self,
            num_epochs: int = 10,
            lr: float = 0.001,
            optimizer: Literal["adam", "rmsprop", "SGD"] = "adam",
    ) -> None:
        super().__init__()
        self.type = "classification"
        self._num_epochs = num_epochs
        self._lr = lr
        self._optimizer = optimizer

    @property
    def parameters(self):
        return deepcopy({
            "num_epochs": self._num_epochs,
            "lr": self._lr,
            "optimizer": self._optimizer,
            **{name: param for name, param in self.named_parameters()}
        })

    def _create_layers(self) -> None:
        if self._num_layers == 1:
            self.layers = ModuleList([
                Linear(self._input_dim, self._output_dim)
            ])
            return
        layers = [Linear(self._input_dim, self._hidden_dim)]
        for _ in range(self._num_layers - 2):
            layers.append(Linear(self._hidden_dim, self._hidden_dim))
        layers.append(Linear(self._hidden_dim, self._output_dim))
        self.layers = ModuleList(layers)

    def _set_dims(
            self,
            observations: Tensor,
            labels: Tensor
    ) -> None:
        unique_labels = len(labels.unique())
        self._output_dim = unique_labels if unique_labels > 2 else 1
        self._input_dim = observations.size(1)
        if self._num_layers > 1:
            self._hidden_dim = (self._input_dim + self._output_dim) // 2

    def _create_trainer(self) -> Trainer:
        """Initialize the trainer with necessary parameters."""
        self._trainer = Trainer(
            self,
            num_epochs=self._num_epochs,
            lr=self._lr,
            loss_fn=self._loss_fn,
            optimizer=self._optimizer
        )
