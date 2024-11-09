from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np
from pydantic import (BaseModel, Field, PrivateAttr, field_validator,
                      model_validator)


class Model(BaseModel, ABC):
    """
    Abstract base class for implementing supervised machine learning models.

    Attributes:
        _parameters (dict): Private dict storing the parameters of the model.

    Methods:
        fit(observations: np.ndarray, labels: np.ndarray) -> None:
            Abstract method to fit the model to observations and labels.
        predict(observations: np.ndarray) -> np.ndarray:
            Abstract method to predict from the trained model.
    """

    _parameters: dict = PrivateAttr(default_factory=dict)

    @abstractmethod
    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        """
        Abstract method to fit the model to observations and labels.

        Args:
            observations (np.ndarray): The training data for the model.
            labels (np.ndarray): Target values corresponding to observations.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Abstract method to predict from the trained model.

        Args:
            observations (np.ndarray): The training data for the model.

        Returns:
            np.ndarray: Array containing predictions.
        """
        pass

class MultipleLinearRegressionOld(Model):
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

        _arg_validator(observations: np.ndarray,
        labels_or_params: np.ndarray, fit: bool) -> None:
            Validates the input arrays for fit or predict.
    """

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        """
        Fits the linear regression model to the provided
        observations and labels.

        Args:
            observations (np.ndarray): 2D or 1D array of independent features.
            labels (np.ndarray): 1D array of dependent variables.

        Raises:
            ValueError: If matrix inversion fails.
        """
        self._arg_validator(observations, labels, fit=True)

        ones = np.ones((observations.shape[0], 1), dtype=np.float64)
        adj_observations = np.concatenate((observations, ones), 1)
        square_mat = adj_observations.T @ adj_observations
        try:
            inv = np.linalg.inv(square_mat)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix inversion error due to singular matrix.")
        parameters = (inv @ adj_observations.T) @ labels
        self._parameters["parameters"] = parameters

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
        if len(self._parameters) == 0:
            raise ValueError(
                "Can not predict predict before fitting model."
            )
        self._arg_validator(
            observations, self._parameters["parameters"][:-1], fit=False
        )
        ones = np.ones((observations.shape[0], 1), dtype=np.float64)
        adj_observations = np.concatenate((observations, ones), 1)
        predictions = adj_observations @ self._parameters["parameters"]
        return predictions

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        """
        Getter for the private _parameters attribute.

        Returns:
            Deepcopy of the _parameters dictionary.
        """
        return deepcopy(self._parameters)

    def _arg_validator(
            self, observations: np.ndarray,
            labels_or_params: np.ndarray,
            fit: bool
    ) -> None:
        """
        Validates the dimensions and types of the input arrays.

        Runs Pydantic field_validators through by instantiating
        the Validator class.

        Args:
            observations (np.ndarray): 2D or 1D array of observations.
            labels_or_params (np.ndarray): 1D array of labels or parameters.
            fit (bool): Boolean indicating if the model is being fit or
            is predicting.
        """
        _ = Validator(
            observations=observations,
            labels_or_params=labels_or_params,
            fit=fit
        )


class Validator(BaseModel):
    """
    A class for validating input arrays for multiple linear regression.

    Attributes:
        fit (bool): True if model is fitting, False if model is predicting.
        labels_or_params (np.ndarray): Array of labels or parameters,
        depending on fit.
        observations (np.ndarray): Array of observations for the model.

    Methods:
        check_array_type(array: np.ndarray) -> np.ndarray:
            Ensures the input is a NumPy ndarray.

        validate_observations(observations_arr: np.ndarray) -> np.ndarray:
            Validates that the observations array has the correct type,
            dimensions, and is non-empty.

        validate_labels_or_params(lab_or_par_arr: np.ndarray) -> np.ndarray:
            Validates that the labels or parameters array has the correct type,
            dimensions, and is non-empty.

        compare_dimensions(observations_arr: np.ndarray, info: ValidationInfo)
        -> np.ndarray:
            Checks if the dimensions of observations match with
            labels_or_params.

        check_column_of_ones(observations_arr: np.ndarray) -> np.ndarray:
            Ensures no column in observations contains only ones.
    """
    labels_or_params: Any = Field(...)
    fit: bool
    observations: Any = Field(...)

    @field_validator("observations", "labels_or_params")
    @classmethod
    def check_array_type(cls, array):
        """
        Method for checking if the provided array is a NumPy ndarray.

        Args:
            array (np.ndarray): The array to check.

        Returns:
            np.ndarray: The validated array.

        Raises:
            TypeError: If the array is not a NumPy ndarray.
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"Expected a numpy.ndarray, but got {type(array).__name__}"
            )
        return array

    @field_validator("observations")
    @classmethod
    def validate_observations(cls, observations_arr: np.ndarray) -> np.ndarray:
        """
        Validates data type and dimensions for observations array.

        Args:
            observations_arr (np.ndarray): Array of observations.

        Returns:
            np.ndarray: The validated observations array.

        Raises:
            TypeError: If the observations array is not of floating-point type.
            ValueError: If the array is not 1 or 2 dimensional or is empty.
        """
        if not np.issubdtype(observations_arr.dtype, np.floating):
            raise TypeError(
                f"Expected observations to be a NumPy array of floats, "
                f"got {observations_arr.dtype} instead."
            )
        if not (1 <= observations_arr.ndim <= 2):
            raise ValueError(
                f"Expected observations to be 1 or 2 dimensional, "
                f"got {observations_arr.ndim} dimensions instead."
            )
        if observations_arr.size == 0:
            raise ValueError("Observations array cannot be empty.")
        return observations_arr

    @field_validator("labels_or_params")
    @classmethod
    def validate_labels_or_params(
        cls, lab_or_par_arr: np.ndarray
    ) -> np.ndarray:
        """
        Validates data type and dimensions for the lab_or_par_arr array.

        Args:
            lab_or_par_arr (np.ndarray): Array of labels or parameters.

        Returns:
            np.ndarray: The validated labels or parameters array.

        Raises:
            TypeError: If the labels or parameters array is not of
            type floating-point.
            ValueError: If the array is not 1 dimensional or is empty.
        """
        if not np.issubdtype(lab_or_par_arr.dtype, np.floating):
            raise TypeError(
                f"Expected labels to be a NumPy array of floats, "
                f"got {lab_or_par_arr.dtype} instead."
            )

        if lab_or_par_arr.ndim != 1:
            raise ValueError(
                f"Expected labels to be 1 dimensional, "
                f"got {lab_or_par_arr.ndim} dimensions instead."
            )
        if lab_or_par_arr.size == 0:
            raise ValueError("Labels array cannot be empty.")
        return lab_or_par_arr

    @model_validator(mode="after")
    def validate_dimensions(self) -> "Validator":
        """
        Compares dimensions of observations and labels or parameters.

        If the model is fitting, dimensions 0 must match.

        If the model is predicting dimension 1 of observations must
        match dimension 0 of parameters. If observations
        has only 1 feature (1D), parameters must have only one
        coefficient.

        Returns:
            Validator: the validated model.
            
        Raises:
            ValueError: If there is a dimension mismatch between observations
            and labels/parameters.
        """
        if self.fit:
            if self.observations.shape[0] != self.labels_or_params.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: Number of observations (rows) "
                    f"({self.observations.shape[0]}) must match the number "
                    f"of labels ({self.labels_or_params.shape[0]})."
                )
        else:
            num_features = (
                self.observations.shape[1] if self.observations.ndim > 1 else 1
            )
            if num_features != self.labels_or_params.shape[0]:
                raise ValueError(
                    f"Dimension mismatch: Number of features (columns) in "
                    f"observations ({self.observations.shape[-1]}) must match "
                    f"the number of parameters in the trained model "
                    f"({self.labels_or_params.shape[0]})."
                )
        return self

    @field_validator("observations")
    @classmethod
    def check_column_of_ones(cls, observations_arr: np.ndarray) -> np.ndarray:
        """
        Ensures that no column in the observations array contains only ones.

        Args:
            observations_arr (np.ndarray): The array of observations to check.

        Returns:
            np.ndarray: The validated observations array.

        Raises:
            ValueError: If a column in observations contains only ones.
        """
        for feature in range(observations_arr.shape[1]):
            if np.all(observations_arr[:, feature] == 1.0):
                raise ValueError(
                    f"Cannot fit model to observations with a column "
                    f"({feature}) containing only ones."
                )
        return observations_arr