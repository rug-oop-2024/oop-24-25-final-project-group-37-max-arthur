from autoop.core.ml.model.model import ClassificationModel
from autoop.functional.activations import sigmoid
from autoop.core.ml.metric import get_metric
from torch import Tensor, no_grad
from torch.nn import Module, Linear
from torch.optim import AdamW


class LogisticRegression(ClassificationModel, Module):
    def __init__(
            self,
            num_epochs: int=10,
            lr: float=0.001
    ) -> None:
        ClassificationModel.__init__(self)
        Module.__init__(self)
        self.type = "classification"
        self.linear = None
        self._populate_model_parameters()
        self._model_parameters["num_epochs"] = num_epochs
        self._model_parameters["lr"] = lr

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(self.linear(x))

    def fit(self, observations: Tensor, labels: Tensor) -> None:
        if self.linear is None or self.linear.in_features != observations.shape[1]:
            self.linear = Linear(observations.shape[1], 1)
        optimizer = AdamW(self.parameters(), lr=self.model_parameters["lr"])
        loss_fn = get_metric("cross_entropy_loss", needs_activation=False)
        self.train()
        for epoch in range(self.model_parameters["num_epochs"]):
            optimizer.zero_grad()
            y_pred = self.forward(observations)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

    def predict(self, observations: Tensor) -> Tensor:
        with no_grad():
            self.eval()
            outputs = self.forward(observations)
            return (outputs >= 0.5).int()