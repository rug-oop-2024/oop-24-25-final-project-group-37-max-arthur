from copy import deepcopy
from typing import Any

from sklearn.ensemble import RandomForestClassifier as RFC

from autoop.core.ml.model.model import ClassificationFacadeModel


class RandomForestClassifier(ClassificationFacadeModel):
    """
    Random forest classifier model using facade pattern.

    Attributes:
        type (Literal["regression", "classification"]): Specifies the
            model type as 'classification'.
        model (BaseEstimator): The wrapped model instance.
        parameters (dict[str, Any]): Dictionary storing model parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the RandomForestClassifier with the wrapped sklearn model.

        Returns:
            None
        """
        super().__init__()
        self._model = RFC(*args, **kwargs)

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Dynamically return the model parameters.

        Returns the hyperparameters stored in the model
        as well as the fitted parameters from the wrapped model.

        Returns:
            dict[str, Any]: parameters dict including hyperparameters.
        """
        params = {
            **self._model.get_params(),
            "estimators_": getattr(self._model, "estimators_", None),
            "n_features_in_": getattr(self._model, "n_features_in_", None),
            "classes_": getattr(self._model, "classes_", None),
            "n_outputs_": getattr(self._model, "n_outputs_", None),
            "feature_importances_": getattr(
                self._model, "feature_importances_", None
            ),
        }
        return deepcopy(params)
