from sklearn.linear_model import ElasticNet as ElasNet

from autoop.core.ml.model.model import RegressionModel


class ElasticNet(RegressionModel):
    """
    ElasticNet model using facade pattern.

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

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the ElasticNet model with the wrapped sklearn model.

        Returns:
            None
        """
        super().__init__()
        self._model = ElasNet(*args, **kwargs)

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
            "fitted_parameters": self._model.coef_,
            "intercept": self._model.intercept_,
        }
        return params
