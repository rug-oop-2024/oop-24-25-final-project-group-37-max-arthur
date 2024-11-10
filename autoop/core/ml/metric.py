from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from torch import Tensor, abs, argmax

from autoop.functional.activations import sigmoid, softmax
from autoop.functional.preprocessing import to_tensor

METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
    "precision",
    "accuracy",
    "recall",
    "f1_score"
]


def get_metric(name: str, needs_activation: bool = False) -> 'Metric':
    """
    Get a metric by name.

    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Metric: An instance of the corresponding Metric class.

    Raises:
        ValueError: If the name is not a valid metric.
    """
    metrics_dict = {
        "mean_squared_error": MeanSquaredError,
        "accuracy": Accuracy,
        "mean_absolute_error": MeanAbsoluteError,
        "precision": Precision,
        "recall": Recall,
        "f1_score": F1Score,
        "r_squared": RSquared,

    }
    if name.lower() not in METRICS:
        raise ValueError(
            f"Unknown metric: {name}, valid metrics are: {METRICS}"
        )
    return metrics_dict[name.lower()](needs_activation)


class Metric(ABC):
    """
    Abstract base class for evaluation metrics.

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __init__(self, needs_avtivation: bool = False) -> None:
        """
        Initialize the metric with an activation requirement.

        Args:
            needs_activation (bool): Indicates if activation is required.
                Default is False.
        """
        self._needs_activation = needs_avtivation

    @property
    def needs_activation(self) -> bool:
        """
        Indicates whether the metric requires activation.

        Returns:
            bool: True if activation is needed, otherwise False.
        """
        return self._needs_activation

    @abstractmethod
    def __call__(
            self,
            predictions: Tensor,
            labels: np.ndarray
    ) -> float:
        """
        Abstractmethod for calculating the method.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Computed metric.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Abstractmethod for returning the metric name.

        Returns:
            str: Name of the metric.
        """
        pass

    def _validate_inputs(self, predictions: Tensor, labels: Tensor) -> None:
        """
        Ensure predictions and labels have matching lengths.

        Args:
            predictions (Tensor): Predicted values.
            labels (Tensor): Actual values.

        Raises:
            AssertionError: If the lengths do not match.
        """
        assert predictions.size(0) == labels.size(0), (
            "Predictions and labels must have the same length."
        )

    def _select_activation(self, labels: Tensor) -> Callable[[Tensor], Tensor]:
        """
        Choose the appropriate activation function.

        Args:
            labels (Tensor): Label tensor for determining class type.

        Returns:
            Callable[[Tensor], Tensor]: Activation function.
        """
        is_binary_class = True if labels.unique().numel() == 2 else False
        return sigmoid if is_binary_class else softmax

    def evaluate(
            self,
            predictions: Tensor,
            labels: np.ndarray
    ) -> float:
        """
        Calculate the metric value as an alias for __call__.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: The computed metric value.
        """
        return self(predictions, labels)

    def _preprocess_classification(
            self,
            predictions: Tensor,
            labels: Tensor
    ) -> tuple[Tensor, Tensor, list[int]]:
        """
        Preprocess predictions and labels for classification.

        Args:
            predictions (Tensor): Predicted labels.
            labels (Tensor): Actual labels.

        Returns:
            tuple[Tensor, Tensor, list[int]]: Processed predictions, labels,
                and class list based on the number of classes.
        """
        labels = labels.squeeze()
        num_classes = list(labels.unique())
        if labels.ndim == 2:
            labels = labels.argmax(1).long()
        if predictions.dim() == 2 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)

        if predictions.ndim > 1:
            num_classes = list(range(predictions.size(1)))
            predictions = argmax(predictions, dim=1).long()
        return predictions, labels, num_classes


class MeanSquaredError(Metric):
    """
    Metric class for calculating mean squared error (MSE).

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        """
        Calculate the mean squared error.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Computed mean squared error.
        """
        labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        mse = ((labels - predictions) ** 2).mean().item()
        return mse

    def __str__(self) -> str:
        """
        Return the metric name.

        Returns:
            str: "mean_squared_error"
        """
        return "mean_squared_error"


class MeanAbsoluteError(Metric):
    """
    Metric class for calculating mean absolute error (MAE).

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        """
        Calculate the mean absolute error.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Computed mean absolute error.
        """
        labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        mae = abs(labels - predictions).mean().item()
        return mae

    def __str__(self) -> str:
        """
        Return the metric name.

        Returns:
            str: "mean_absolute_error"
        """
        return "mean_absolute_error"


class RSquared(Metric):
    """
    Metric class for calculating R-squared (coefficient of determination).

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        """
        Calculate the R-squared score.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Computed R-squared score.
        """
        labels = to_tensor(labels)
        residual_sum_se = ((labels - predictions) ** 2).sum().item()
        total_sum_se = ((labels - labels.mean()) ** 2).sum().item()
        if total_sum_se == 0:
            rs = 1.0 if residual_sum_se == 0 else 0.0
        else:
            rs = 1 - residual_sum_se / total_sum_se
        return rs

    def __str__(self) -> str:
        """
        Return the metric name.

        Returns:
            str: "r_squared"
        """
        return "r_squared"


class Accuracy(Metric):
    """
    Metric class for calculating classification accuracy.

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        """
        Calculate accuracy.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Computed accuracy.
        """
        labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        if self.needs_activation:
            predictions = self._select_activation(labels)(predictions)
        predictions, labels, _ = self._preprocess_classification(
            predictions, labels
        )
        num_correct = (predictions == labels).sum()
        accuracy = num_correct / labels.size(0)
        return accuracy.item()

    def __str__(self) -> str:
        """
        Return the metric name.

        Returns:
            str: "accuracy"
        """
        return "accuracy"


class Precision(Metric):
    """
    Metric class for calculating precision in classification tasks.

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __call__(self, predictions: Tensor, labels: np.ndarray) -> float:
        """
        Calculate the precision metric.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Average precision across classes.
        """
        labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        if self.needs_activation:
            predictions = self._select_activation(labels)(predictions)
        predictions, labels, classes = self._preprocess_classification(
            predictions, labels
        )

        precision_list = []
        for cls in classes:
            tp = ((predictions == cls) & (labels == cls)).sum().item()
            fp = ((predictions == cls) & (labels != cls)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_list.append(precision)

        return Tensor(precision_list).mean().item()

    def __str__(self) -> str:
        """
        Return the metric name.

        Returns:
            str: "precision"
        """
        return "precision"


class Recall(Metric):
    """
    Metric class for calculating recall in classification tasks.

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __call__(self, predictions: Tensor, labels: np.ndarray) -> float:
        """
        Calculate the recall metric.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Average recall across classes.
        """
        labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        if self.needs_activation:
            predictions = self._select_activation(labels)(predictions)
        predictions, labels, classes = self._preprocess_classification(
            predictions, labels
        )

        recall_list = []
        for cls in classes:
            tp = ((predictions == cls) & (labels == cls)).sum().item()
            fn = ((predictions != cls) & (labels == cls)).sum().item()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_list.append(recall)

        return Tensor(recall_list).mean().item()

    def __str__(self) -> str:
        """
        Return the metric name.

        Returns:
            str: "recall"
        """
        return "recall"


class F1Score(Metric):
    """
    Metric class for calculating the F1 score in classification tasks.

    Attributes:
        needs_activation (bool): Indicates if the metric requires an
            activation function.
    """

    def __init__(self, needs_activation: bool = True) -> None:
        """
        Initialize the F1 score metric, with precision and recall as instances.

        Args:
            needs_activation (bool): Indicates if activation is required.

        Returns:
            None
        """
        self.precision = Precision(needs_activation)
        self.recall = Recall(needs_activation)
        super().__init__(needs_activation)

    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        """
        Calculate the F1 score.

        Args:
            predictions (Tensor): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            float: Computed F1 score.
        """
        precision = self.precision(predictions, labels)
        recall = self.recall(predictions, labels)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1

    def __str__(self) -> str:
        """
        Return the metric name.

        Returns:
            str: "f1_score"
        """
        return "f1_score"
