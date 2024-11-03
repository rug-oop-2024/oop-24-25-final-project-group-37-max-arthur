from autoop.core.ml.model.model import ClassificationMixin
from autoop.functional.activations import sigmoid
from autoop.core.ml.metric import get_metric
from torch import Tensor, no_grad
from torch.nn import Module, Linear


class LogisticRegression(ClassificationMixin, Module):
    def __init__(self, *args, **kwargs) -> None:
        ClassificationMixin.__init__(self, *args, **kwargs)
        Module.__init__(self)
        self.type = "classification"
        self._loss_fn = get_metric(
            "cross_entropy_loss", needs_activation=False
        )
        self.linear = None

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(self.linear(x))

    def fit(self, observations: Tensor, labels: Tensor) -> None:
        if self.linear is None or self.linear.in_features != observations.shape[1]:
            self.linear = Linear(observations.shape[1], 1)
            self._populate_model_parameters()
        self._set_trainer()
        self._trainer.train(observations, labels)

    def predict(self, observations: Tensor) -> Tensor:
        with no_grad():
            self.eval()
            outputs = self.forward(observations)
            return (outputs >= 0.5).int()