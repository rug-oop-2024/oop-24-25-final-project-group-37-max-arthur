from copy import deepcopy
from typing import Any

import numpy as np
from torch import Tensor, from_numpy
from autoop.core.ml.model.model import Model


class MultipleLinearRegressionOld:
    """
    Implementation of a multiple Linear Regression model.

    Attributes:
        _parameters (dict): Dictionary storing the model parameters
        after fitting.

    Methods:
        fit(observations: np.ndarray, labels: np.ndarray) -> None:
            Fits the regression model to observations and labels.

        predict(observations: np.ndarray) -> np.ndarray:
            Makes predictions based on new observations using the fitted model.

        parameters() -> dict[str, np.ndarray]:
            Property that returns a deepcopy of the model parameters.
    """

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fits the linear regression model to the provided
        observations and labels.

        Args:
            observations (np.ndarray): 2D or 1D array of independent features.
            labels (np.ndarray): 1D array of dependent variables.

        Raises:
            ValueError: If matrix inversion fails.
        """
        ones = np.ones((observations.shape[0], 1), dtype=np.float64)
        adj_observations = np.concatenate((observations, ones), 1)
        square_mat = adj_observations.T @ adj_observations
        try:
            inv = np.linalg.inv(square_mat)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion error due to singular matrix.")
        parameters = (inv @ adj_observations.T) @ labels
        return parameters

    def predict(self, observations: np.ndarray):
        """
        Makes predictions using the fitted model for new observations.

        Args:
            observations (np.ndarray): 2D or 1D array of independent features.

        Returns:
            np.ndarray: Predicted labels for the input observations.

        Raises:
            ValueError: If called before fitting model.
        """
        if len(self.model_parameters) == 0:
            raise ValueError(
                "Can not predict predict before fitting model."
            )
        ones = np.ones((observations.shape[0], 1), dtype=np.float64)
        adj_observations = np.concatenate((observations, ones), 1)
        predictions = adj_observations @ self.model_parameters["parameters"]
        return predictions



class MultipleLinearRegression(Model, MultipleLinearRegressionOld):
    def __init__(self) -> None:
        super().__init__()
        self.type = "regression"

    def fit(self, observations: Tensor, labels: Tensor) -> None:
        params = MultipleLinearRegressionOld.fit(
            self, observations.numpy(), labels.numpy()
            )
        self._model_parameters["coefficients"] = from_numpy(params[:-1])
        self._model_parameters["intercept"] = from_numpy(params[-1:])

    def predict(self, observations: Tensor) -> Tensor:
        coefficients = self.model_parameters["coefficients"]
        intercept = self.model_parameters["intercept"]
        
        predictions = observations @ coefficients + intercept
        return predictions