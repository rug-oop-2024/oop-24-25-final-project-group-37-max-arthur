
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from torch import Tensor, from_numpy

from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """
    Abstract base class for machine learning models.

    Attributes:
        parameters (dict[str, Any]): A dictionary to store model parameters.
        fitted (bool): A flag indicating whether the model has been fitted.
    """

    def __init__(self) -> None:
        """
        Initialize the Model with default parameters and a fitted status.

        Returns:
            None
        """
        self._parameters: dict[str, Any] = {}
        self._fitted: bool = False

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """
        Abstractmethod for getting the model parameters.

        Returns:
            dict[str, Any]: A dictionary containing the model's parameters.
        """
        pass

    @property
    def type(self) -> Literal["regression", "classification"]:
        """
        Get the type attribute.

        Returns:
            Literal["regression", "classification"]: The type of model,
                either "regression" or "classification".
        """
        return self._type

    @type.setter
    def type(self, value: Literal["regression", "classification"]) -> None:
        """
        Set the type attribute.

        Args:
            value (Literal["regression", "classification"]):
                The proposed value for 'type'.

        Raises:
            ValueError: If the value for the model type is not one of
                "regression" or "classification".

        Returns:
            None
        """
        if value not in ["regression", "classification"]:
            raise ValueError(
                "Type has to be 'regression', or 'classification'."
                f" Got {value} instead."
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
            observations (np.ndarray): The training data for the model.
            labels (np.ndarray): Target values corresponding to observations.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> Tensor:
        """
        Abstract method to predict from the trained model.

        Args:
            observations (np.ndarray): The training data for the model.

        Returns:
            torch.Tensor: Tensor containing predictions.
        """
        pass

    def _check_fitted(self) -> None:
        """
        Check if the model is fitted.

        Raises:
            NotFittedError: If the model has not been fitted yet.

        Returns:
            None
        """
        if not self._fitted:
            raise NotFittedError(
                "This model is not fitted yet. Call 'fit' first."
            )

    def to_artifact(self, name: str) -> 'Artifact':
        """
        Serialize the model and create an Artifact object.

        Args:
            name (str): Name for the artifact. Also used
                for the asset_path.

        Returns:
            Artifact: Resulting artifact object.
        """
        serialized_data = pickle.dumps(self)

        asset_path = f"models/{name}.pkl"
        metadata = {
            "model_type": self.type,
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
    """
    Abstract subclass representing a general regression model.

    This class provides base functionality for regression models
    it is not intended to be instantiated on its own.

    Attributes:
        type (Literal["regression", "classification"]): Specifies the
            model type as 'regression'.
    """

    def __init__(self) -> None:
        """
        Initialize the RegressionModel with type 'regression'.

        Returns:
            None
        """
        super().__init__()
        self.type = "regression"

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the regression model to observations and labels.

        Args:
            observations (np.ndarray): Training data used to fit the model.
            labels (np.ndarray): Target values corresponding to observations.
                Expected shape is either [B, 1] or [B].

        Raises:
            ValueError: If labels have an unsupported shape.
            AssertionError: If the number of samples in observations and
                labels do not match.

        Returns:
            None
        """
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
        self._fitted = True

    def predict(self, observations: np.ndarray) -> Tensor:
        """
        Generate predictions using the trained model.

        Args:
            observations (np.ndarray): Data for which predictions are needed.

        Raises:
            NotFittedError: If the model has not been fitted.

        Returns:
            torch.Tensor: Predictions made by the model.
        """
        self._check_fitted()
        predictions = self._model.predict(observations)
        return Tensor(predictions)


class ClassificationFacadeModel(Model):
    """
    Abstract subclass representing a general regression model.

    This class provides base functionality for regression models
    it is not intended to be instantiated on its own.

    Attributes:
        type (Literal["regression", "classification"]): Specifies the
            model type as 'classification'.
        model (BaseEstimator): The wrapped model instance.
    """

    def __init__(self) -> None:
        """
        Initialize the ClassificationFacadeModel with type 'regression'.

        Returns:
            None
        """
        super().__init__()
        self.type = "classification"

    @property
    def model(self) -> BaseEstimator:
        """
        Get a deep copy of the underlying model instance.

        Returns:
            BaseEstimator: A copy of the wrapped model instance
                used in training and prediction.
        """
        return deepcopy(self._model)

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the model to observations and labels.

        Uses the wrapped models fit method.

        Args:
            observations (np.ndarray): Array of training data.
            labels (np.ndarray): Array of target labels.

        Raises:
            AssertionError: If the number of samples in observations
            and labels do not match or there are less than 2 classes
            to classify.

        Returns:
            None
        """
        assert labels.shape[0] == observations.shape[0], (
            "Observations and labels must have the same number of samples. "
            f"Got {labels.shape[0]} and {observations.shape[0]} instead."
        )
        labels = labels.argmax(1)
        unique_labels = len(np.unique(labels))
        assert unique_labels >= 2, (
            f"Expected at least 2 classes, got {unique_labels} instead."
        )
        self._model.fit(observations, labels)
        self._fitted = True

    def predict(self, observations: np.ndarray) -> Tensor:
        """
        Predict class labels for the given observations.

        Uses the wrapped models predict method.

        Args:
            observations (np.ndarray): Array of observations to predict.

        Raises:
            NotFittedError: If the model has not been fitted.

        Returns:
            Tensor: Predicted class labels as a tensor.
        """
        self._check_fitted()
        predictions = self._model.predict(observations)
        return from_numpy(predictions)
