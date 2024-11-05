from typing import Callable, Literal
from torch.nn import Module
from torch import Tensor
from torch.optim import AdamW, Optimizer, RMSprop, SGD

class Trainer:
    def __init__(
            self,
            model: Module,
            loss_fn: Callable[[Tensor, Tensor], float | Tensor],
            num_epochs: int = 10,
            lr: float = 0.001,
            optimizer: Literal["adam", "rmsprop", "SGD"] = "adam"
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._num_epochs = num_epochs
        self._lr = lr
        self.optimizer = optimizer

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: str) -> None:
        options = ["adam", "rmsprop", "SGD"]
        assert value.lower() in options, (
            f"Optimizer has to be in {options}. Found {value} instead."
        )
        optimizer_dict = {
            "adam": AdamW,
            "rmsprop": RMSprop,
            "sgd": SGD
        }
        self._optimizer = optimizer_dict[value.lower()](
            [param for _, param in self._model.named_parameters()], self._lr
        )

    def train(self, observations: Tensor, labels: Tensor) -> None:
        self._model.train()
        for epoch in range(self._num_epochs):
            self._optimizer.zero_grad()
            predictions = self._model(observations)
            loss = self._loss_fn(predictions, labels)
            print(f"Loss: {loss}")
            loss.backward()
            self._optimizer.step()
