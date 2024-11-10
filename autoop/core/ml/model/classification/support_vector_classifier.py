from copy import deepcopy
from typing import Any

from sklearn.svm import SVC as SKLSVC

from autoop.core.ml.model.model import ClassificationFacadeModel


class SVC(ClassificationFacadeModel):
    """
    Support vector classifier model using facade pattern.

    Attributes:
        type (Literal["regression", "classification"]): Specifies the
            model type as 'classification'.
        model (BaseEstimator): The wrapped model instance.
        parameters (dict[str, Any]): Dictionary storing model parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the SVC model with the wrapped sklearn model.

        Returns:
            None
        """
        super().__init__()
        self._model = SKLSVC(*args, **kwargs)

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
            "support_": getattr(self._model, "support_", None),
            "support_vectors_": getattr(self._model, "support_vectors_", None),
            "n_features_in_": getattr(self._model, "n_features_in_", None),
            "classes_": getattr(self._model, "classes_", None),
            "probA_": getattr(self._model, "probA_", None),
            "probB_": getattr(self._model, "probB_", None),
        }
        return deepcopy(params)
