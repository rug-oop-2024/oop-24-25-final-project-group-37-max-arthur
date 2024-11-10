from copy import deepcopy
from typing import Any, Literal

import numpy as np
from torch import Tensor, no_grad
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import cross_entropy, relu

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.trainer import Trainer
from autoop.functional.preprocessing import to_tensor


class MLP(Model, Module):
    """
    Multi-layer perceptron model for classification.

    Initialized automatically according to the attribute
    num_layers and the dimensions of observations and
    labels passed during fit.

    Attributes:
        num_layers (int): Number of layers in the model, set to 1.
        parameters (dict[str, Any]): Dictionary storing model parameters.
        layers (ModuleList): Module list containing the layers of the model.
        trainer (Trainer): Contains the trainer object.
        type (Literal["classification, "regression"]): Specifies the model
            type as 'classification'.
    """

    def __init__(
            self,
            num_layers: int = 3,
            num_epochs: int = 10,
            lr: float = 0.001,
            optimizer: Literal["adam", "rmsprop", "SGD"] = "adam"
    ) -> None:
        """
        Initialize the multi-layer perceptron model.

        Args:
            num_layers (int): Number of layers for the model. Has to
                be a positive integer.
            num_epochs (int): Number of training epochs. Default is 10.
            lr (float): Learning rate for the optimizer. Default is 0.001.
            optimizer (Literal["adam", "rmsprop", "SGD"]): Optimizer type.
                Default is 'adam'.

        Returns:
            None
        """
        Model.__init__(self)
        Module.__init__(self)
        self.num_layers = num_layers
        self.type = "classification"
        self._loss_fn = cross_entropy
        self._num_epochs = num_epochs
        self._lr = lr
        self._optimizer = optimizer

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Dynamically return the model parameters.

        Returns the hyperparameters stored in the model
        as well as the fitted parameters from torch's nn.Module.

        Returns:
            dict[str, Any]: parameters dict including hyperparameters.
        """
        return deepcopy({
            "num_epochs": self._num_epochs,
            "lr": self._lr,
            "optimizer": self._optimizer,
            **{name: param for name, param in self.named_parameters()}
        })

    @property
    def layers(self) -> ModuleList:
        """
        Return a copy of the models layers if they have been created.

        Raises:
            AttributeError: If _layer have not been initialized yet.

        Returns:
            ModuleList: The models layers.
        """
        if not hasattr(self, '_layers'):
            raise AttributeError(
                "layers is not initialized. Call 'fit' first."
            )
        return deepcopy(self._layers)

    @property
    def trainer(self) -> Trainer:
        """
        Return a copy of the models trainer if it has been created.

        Raises:
            AttributeError: If the _trainer attribute does not exist.

        Returns:
            Trainer: A copy of the trainer instance of the model.
        """
        if not hasattr(self, '_trainer'):
            raise AttributeError(
                "trainer is not initialized. Call 'fit' first."
            )
        return deepcopy(self._trainer)

    @property
    def num_layers(self) -> int:
        """
        Get the value of num_layers.

        Returns:
            int: The number of layers.
        """
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        """
        Set the num_layers attribute.

        Args:
            value (int): The proposed value for 'num_layers'.

        Raises:
            ValueError: If the value for the num_layers is not a
                positive integers.

        Returns:
            None
        """
        if not isinstance(value, int):
            raise ValueError(f"Expected integer, got {type(value)} instead.")
        if value <= 0:
            raise ValueError(f"num_layers has to be > 0. Got {value}.")
        self._num_layers = value

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through all layers.

        Applies ReLu activation function after each layer apart from
        the final layer.

        Args:
            x (Tensor): Input tensor containing observations.

        Returns:
            Tensor: Output tensor from the logistic regression layer.
        """
        for layer in self._layers[: -1]:
            x = relu(layer(x))
        return self._layers[-1](x)

    def fit(self, observations: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the model to observations and labels.

        Calls helper functions to convert observations and labels to
        tensors, set input and output dimensions based on the data,
        create model layers, instantiate the Trainer, and initiate
        the training loop.

        Args:
            observations (np.ndarray): Array of training data.
            labels (np.ndarray): Array of target labels.

        Raises:
            AssertionError: If the number of samples in observations
            and labels do not match.

        Returns:
            None
        """
        observations, labels = to_tensor(observations, labels)
        assert labels.size(0) == observations.size(0), (
            "Observations and labels must have the same number of samples. "
            f"Got {observations.size(0)} and {labels.size(0)} instead."
        )
        labels = labels.argmax(1).long()
        self._set_dims(observations, labels)
        self._create_layers()
        self._create_trainer()
        self._trainer.train(observations, labels)
        self._fitted = True

    def predict(self, observations: np.ndarray) -> Tensor:
        """
        Predict class labels for the given observations.

        Converts observations to tensors, performs a forward pass without
        gradient computation, and returns predicted class labels.

        Args:
            observations (np.ndarray): Array of observations to predict.

        Raises:
            NotFittedError: If the model has not been fitted.

        Returns:
            Tensor: Predicted class labels as a tensor.
        """
        self._check_fitted()
        with no_grad():
            self.eval()
            observations = to_tensor(observations)
            outputs = self.forward(observations)
            return outputs.argmax(1)

    def _create_layers(self) -> None:
        """
        Initialize the model's layers based on specified dimensions.

        Create attribute self._layers containing the linear layers
        of the model. Layers are created based on the number of layers,
        and the calculated dimension.

        Returns:
            None
        """
        if self._num_layers == 1:
            self._layers = ModuleList([
                Linear(self._input_dim, self._output_dim)
            ])
            return
        layers = [Linear(self._input_dim, self._hidden_dim)]
        for _ in range(self._num_layers - 2):
            layers.append(Linear(self._hidden_dim, self._hidden_dim))
        layers.append(Linear(self._hidden_dim, self._output_dim))
        self._layers = ModuleList(layers)

    def _set_dims(
            self,
            observations: Tensor,
            labels: Tensor
    ) -> None:
        """
        Set the model's input and output dimensions.

        Uses the number of unique values in the labels tensor
        as a heuristic for the output dimension. Input dimension
        is set to the number of columns in observations. Hidden
        dimension is the integer mean of the sum of input dimension
        and output dimension.

        Args:
            observations (Tensor): Observations used for fitting.
            labels (Tensor): Labels used for fitting.

        Raises:
            AssertionError: If there are less then 2 classes.
        """
        unique_labels = len(labels.unique())
        assert unique_labels >= 2, (
            f"Expected at least 2 classes, got {unique_labels} instead."
        )
        self._output_dim = unique_labels
        self._input_dim = observations.size(1)
        if self._num_layers > 1:
            self._hidden_dim = (self._input_dim + self._output_dim) // 2

    def _create_trainer(self) -> None:
        """
        Initialize the trainer attribute with a Trainer object.

        Passes the model itself and the hyperparameters to initiate
        training setup.

        Returns:
            None
        """
        self._trainer = Trainer(
            self,
            num_epochs=self._num_epochs,
            lr=self._lr,
            loss_fn=self._loss_fn,
            optimizer=self._optimizer
        )
