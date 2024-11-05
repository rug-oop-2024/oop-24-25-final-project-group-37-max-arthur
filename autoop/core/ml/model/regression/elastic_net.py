from autoop.core.ml.model.model import RegressionModel
from sklearn.linear_model import ElasticNet as ElasNet
from torch import Tensor
import numpy as np
from copy import deepcopy

class ElasticNet(RegressionModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._model = ElasNet(*args, **kwargs)
    
    @property
    def parameters(self) -> dict:
        params = {
            **self._model.get_params(),
            "fitted_parameters": self._model.coef_,
            "intercept": self._model.intercept_,
        }
        return params

    def predict(self, observations: np.ndarray) -> Tensor:
        predictions = self._model.predict(observations)
        return Tensor(predictions)