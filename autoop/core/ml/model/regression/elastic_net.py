from autoop.core.ml.model.model import RegressionModel
from sklearn.linear_model import ElasticNet as ElasNet


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
