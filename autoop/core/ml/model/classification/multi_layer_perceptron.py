from autoop.core.ml.model.model import GradientModel
from autoop.core.ml.metric import get_metric
from autoop.functional.preprocessing import to_tensor
from copy import deepcopy
import numpy as np
from torch import Tensor, no_grad
from torch.nn import Module
from torch.nn.functional import relu, cross_entropy

class MLP(Module, GradientModel):
    def __init__(
            self,
            num_layers: int = 3,
            *args,
            **kwargs
    ) -> None:
        GradientModel.__init__(self, *args, **kwargs)
        Module.__init__(self)
        self._num_layers = num_layers
        self._loss_fn = get_metric(
            "cross_entropy_loss", needs_activation=True
        )
        self._loss_fn = cross_entropy

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[: -1]:
            x = relu(layer(x))
        return self.layers[-1](x)

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        observations, labels = to_tensor(observations, labels)
        assert labels.size(0) == observations.size(0), (
            "Observations and labels must have the same number of samples. "
            f"Got {observations.size(0)} and {labels.size(0)} instead."
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
            if self._output_dim == 1:
                predictions = (outputs >= 0.5).int()
            else:
                predictions = outputs.argmax(1)
            return outputs

