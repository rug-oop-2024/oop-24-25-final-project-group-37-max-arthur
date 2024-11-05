from sklearn.ensemble import RandomForestClassifier as RFC
from autoop.core.ml.model.model import Model
import numpy as np
from torch import Tensor, from_numpy
from copy import deepcopy


class RandomForestClassifier(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.type = "classification"
        self._model = RFC(*args, **kwargs)

    @property
    def parameters(self) -> dict:
        params = {
            **self._model.get_params(),
            "estimators_": self._model.estimators_,
            "n_features_in_": self._model.n_features_in_,
            "classes_": self._model.classes_,
            "n_outputs_": self._model.n_outputs_,
            "feature_importances_": self._model.feature_importances_,
        }
        return deepcopy(params)

    @property
    def model(self) -> RFC:
        return deepcopy(self._model)
    
    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        assert labels.shape[0] == observations.shape[0], (
            "Observations and labels must have the same number of samples. "
            f"Got {labels.shape[0]} and {observations.shape[0]} instead."
        )
        self._model.fit(observations, labels.argmax(1))
    
    def predict(self, observations: np.ndarray) -> Tensor:
        predictions = self._model.predict(observations)
        return from_numpy(predictions)
