from copy import deepcopy

from sklearn.linear_model import Lasso as SklearnLasso

from autoop.core.ml.model.model import RegressionModel


class Lasso(RegressionModel):
    """
    Lasso model using facade pattern.

    Attributes:
        type (Literal["regression", "classification"]): Specifies the
            model type as 'regression'.
        model (SklearnLasso): The wrapped model instance.
        parameters (dict[str, Any]): Dictionary storing model parameters.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Lasso model with the wrapped sklearn model.

        Returns:
            None
        """
        super().__init__()
        self._model = SklearnLasso(*args, **kwargs)

    @property
    def model(self) -> SklearnLasso:
        """
        Get a deep copy of the underlying model instance.

        Returns:
            SklearnLasso: A copy of the wrapped model instance
                used in training and prediction.
        """
        return deepcopy(self._model)

    @property
    def parameters(self) -> dict:
        """
        Dynamically return the model parameters.

        Returns the hyperparameters stored in the model
        as well as the fitted parameters from the wrapped model.

        Returns:
            dict[str, Any]: parameters dict including hyperparameters.
        """
        params = {
            **self._model.get_params(),
            "fitted_parameters": getattr(self._model, "coef_", None),
            "intercept": getattr(self._model, "intercept_", None),
        }
        return params
