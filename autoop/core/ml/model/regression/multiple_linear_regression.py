from autoop.core.ml.model.model import RegressionModel
from autoop.core.ml.model.regression.multiple_linear_regression_old import MultipleLinearRegressionOld as MLRold


class MultipleLinearRegression(RegressionModel):
    def __init__(self) -> None:
        super().__init__()
        self._model = MLRold()

    @property
    def parameters(self) -> dict:
        return self._model.parameters
