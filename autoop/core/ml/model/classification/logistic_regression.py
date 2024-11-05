from autoop.core.ml.model.model import GradientModel
from autoop.functional.activations import sigmoid
from autoop.core.ml.metric import get_metric
from autoop.functional.preprocessing import to_tensor
import numpy as np
from torch import Tensor, no_grad
from torch.nn import Module


class LogisticRegression(GradientModel, Module):
    def __init__(self, *args, **kwargs) -> None:
        GradientModel.__init__(self, *args, **kwargs)
        Module.__init__(self)
        self._num_layers = 1
        self._loss_fn = get_metric(
            "cross_entropy_loss", needs_activation=False
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = sigmoid(layer(x))
        return x

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        observations, labels = to_tensor(observations, labels)
        assert labels.max() == 1, "Labels must be binary (0 or 1)"
        assert labels.size(0) == observations.size(0), (
            "Observations and labels must have the same number of samples. "
            f"Got {labels.size(0)} and {observations.size(0)} instead."
        )
        labels = labels.argmax(1).long()
        self._set_dims(observations, labels)
        self._create_layers()
        self._trainer = self._create_trainer()
        self._trainer.train(observations, labels)

    def predict(self, observations: np.ndarray) -> Tensor:
        with no_grad():
            self.eval()
            observations = to_tensor(observations)
            outputs = self.forward(observations)
            return outputs
            return (outputs >= 0.5).int()
