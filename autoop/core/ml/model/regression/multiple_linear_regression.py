from copy import deepcopy

import numpy as np
from torch import Tensor
from autoop.core.ml.model.model import RegressionModel
from autoop.core.ml.model.regression.multiple_linear_regression_old import MultipleLinearRegressionOld as MLRold


class MultipleLinearRegression(RegressionModel):
    def __init__(self) -> None:
        super().__init__()
        self.type = "regression"

    def fit(self, observations: Tensor, labels: Tensor) -> None:
        params = MultipleLinearRegressionOld.fit(
            self, observations.numpy(), labels.numpy()
            )
        self._model_parameters["coefficients"] = from_numpy(params[:-1])
        self._model_parameters["intercept"] = from_numpy(params[-1:])

    def predict(self, observations: np.ndarray) -> Tensor:
        predictions = self._model.predict(observations)
        return Tensor(predictions)

    def predict(self, observations: Tensor) -> Tensor:
        coefficients = self.model_parameters["coefficients"]
        intercept = self.model_parameters["intercept"]
        predictions = observations.double() @ coefficients + intercept
        return predictions
