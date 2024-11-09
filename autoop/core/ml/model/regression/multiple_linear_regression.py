from autoop.core.ml.model.model import RegressionModel
from autoop.core.ml.model.regression.multiple_linear_regression_old import \
    MultipleLinearRegressionOld as MLRold


class MultipleLinearRegression(RegressionModel):
    """
    Multiple linear regression model using facade pattern.

    Attributes:
        type (Literal["regression", "classification"]): Specifies the
            model type as 'regression'.
        model (BaseEstimator | Model): The wrapped model instance.
        parameters (dict[str, Any]): Dictionary storing model parameters.

    Methods:
        fit(observations: np.ndarray, labels: np.ndarray) -> None:
            Trains the model using provided observations and labels.

        predict(observations: np.ndarray) -> Tensor:
            Generates predictions from the trained model.

        to_artifact(name: str) -> Artifact:
            Serialize the model and create an Artifact object.
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
    def parameters(self) -> dict:
        """
        Dynamically return the model parameters.

        Returns the parameters from the wrapped model instance.

        Returns:
            dict[str, Any]: parameters dict including hyperparameters.
        """
        return self._model.parameters
