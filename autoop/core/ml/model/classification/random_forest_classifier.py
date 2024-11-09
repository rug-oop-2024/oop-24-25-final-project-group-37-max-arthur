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

    Methods:
        fit(observations: np.ndarray, labels: np.ndarray) -> None:
            Trains the model using provided observations and labels.

        predict(observations: np.ndarray) -> Tensor:
            Generates predictions from the trained model.

        to_artifact(name: str) -> Artifact:
            Serialize the model and create an Artifact object.
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
            "estimators_": self._model.estimators_,
            "n_features_in_": self._model.n_features_in_,
            "classes_": self._model.classes_,
            "n_outputs_": self._model.n_outputs_,
            "feature_importances_": self._model.feature_importances_,
        }
        return deepcopy(params)
