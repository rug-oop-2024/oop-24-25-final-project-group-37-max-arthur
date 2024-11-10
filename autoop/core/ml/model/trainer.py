from copy import deepcopy
from typing import Callable, Literal

from torch import Tensor
from torch.nn import Module
from torch.optim import SGD, AdamW, Optimizer, RMSprop


class Trainer:
    """
    Class representing the training process for gradient-based models.

    Attributes:
        model (Module): The model to be trained.
        optimizer (Optimizer): The optimizer used for training.
    """

    def __init__(
            self,
            model: Module,
            loss_fn: Callable[[Tensor, Tensor], Tensor],
            num_epochs: int = 10,
            lr: float = 0.001,
            optimizer: Literal["adam", "rmsprop", "SGD"] = "adam"
    ) -> None:
        """
        Initialize the Trainer class.

        Args:
            model (Module): The model to train.
            loss_fn (Callable[[Tensor, Tensor], Tensor]): Loss function used
                during training.
            num_epochs (int): Number of epochs for training. Default is 10.
            lr (float): Learning rate for the optimizer. Default is 0.001.
            optimizer (Literal["adam", "rmsprop", "SGD"]): Optimizer type for
                training. Default is 'adam'.

        Returns:
            None
        """
        self._model = model
        self._loss_fn = loss_fn
        self._num_epochs = num_epochs
        self._lr = lr
        self.optimizer = optimizer

    @property
    def model(self) -> Module:
        """
        Get a copy of the model being trained.

        Returns:
            Module: A copy of the model.
        """
        return deepcopy(self._model)

    @property
    def optimizer(self) -> Optimizer:
        """
        Get a copy of the optimizer.

        Returns:
            Optimizer: A copy of the optimizer used for training.
        """
        return deepcopy(self._optimizer)

    @optimizer.setter
    def optimizer(self, value: str) -> None:
        """
        Set the optimizer for training.

        Uses the input string to populate the optimizer attribute
        with an instance of the optimizer object.

        Args:
            value (str): The optimizer type ('adam', 'rmsprop', or 'SGD').

        Raises:
            AssertionError: If the specified optimizer type is not valid.

        Returns:
            None
        """
        options = ["adam", "rmsprop", "SGD"]
        assert value.lower() in options, (
            f"Optimizer must be one of {options}. Found '{value}' instead."
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
        """
        Run the training loop for the model.

        For each epoch, performs a forward pass, calculates loss, and
        backpropagates to update model parameters.

        Args:
            observations (Tensor): Training data for the model.
            labels (Tensor): Targets corresponding to the observations.

        Returns:
            None
        """
        self._model.train()
        for epoch in range(self._num_epochs):
            self._optimizer.zero_grad()
            predictions = self._model(observations)
            loss = self._loss_fn(predictions, labels)
            print(f"Loss: {loss}")
            loss.backward()
            self._optimizer.step()
