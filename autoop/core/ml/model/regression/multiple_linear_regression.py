from copy import deepcopy

from autoop.core.ml.model.model import RegressionModel
from autoop.core.ml.model.regression.multiple_linear_regression_old import \
    MultipleLinearRegressionOld as MLRold


class MultipleLinearRegression(RegressionModel):
    """
    Multiple linear regression model using facade pattern.

    Attributes:
        type (Literal["regression", "classification"]): Specifies the
            model type as 'regression'.
        model (ModelOld): The wrapped model instance.
        parameters (dict[str, Any]): Dictionary storing model parameters.
    """

    def __init__(self) -> None:
        """
        Initialize the MLR model with the wrapped model from assignment 1.

        Returns:
            None
        """
        super().__init__()
        self._model = MLRold()

    @property
    def model(self) -> MLRold:
        """
        Get a deep copy of the underlying model instance.

        Returns:
            MLRold: A copy of the wrapped model instance
                used in training and prediction.
        """
        return deepcopy(self._model)

    @property
    def parameters(self) -> dict:
        """
        Dynamically return the model parameters.

        Returns the parameters from the wrapped model instance.

        Returns:
            dict[str, Any]: parameters dict including hyperparameters.
        """
        return self._model.parameters
