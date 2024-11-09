from typing import Callable
from abc import ABC, abstractmethod
import numpy as np
from torch import argmax, Tensor, log, abs, where, long
from autoop.functional.activations import softmax, sigmoid
from autoop.functional.preprocessing import to_tensor

METRICS = [
    "mean_squared_error",
    "cross_entropy_loss",
    "mean_absolute_error",
    "r_squared",
    "precision",
    "accuracy",
    "recall",
    "f1_score"
]


#  could introduce functions upstream, that pass is_binary e.g.
def get_metric(name: str, needs_activation: bool=False) -> 'Metric':
    """
    Factory function to get a metric by name.
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
        "cross_entropy_loss": CrossEntropyLoss,
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


#  we currently only return Tensor in CrossEntropyLoss
#  Have to do this to align with torch, but this does not align
#  with our base class.


class Metric(ABC):
    def __init__(self, needs_avtivation: bool=False):
        self._needs_activation = needs_avtivation

    @property
    def needs_activation(self) -> bool:
        return self._needs_activation

    @abstractmethod
    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def _validate_inputs(self, predictions: Tensor, labels: Tensor) -> None:
        assert predictions.size(0) == labels.size(0), (
            "Predictions and labels must have the same length."
        )

    def _select_activation(self, labels: Tensor) -> Callable[[Tensor], Tensor]:
        is_binary_class = True if labels.unique().numel() == 2 else False
        return sigmoid if is_binary_class else softmax

    def evaluate(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        return self(predictions, labels)

    def _preprocess_tensors(
            self,
            predictions: Tensor,
            labels: np.ndarray
    ) -> tuple[Tensor, Tensor, list[int]]:
        labels = labels.squeeze()
        if labels.ndim == 2:
            labels = labels.argmax(1).long()
        is_binary = labels.unique().numel() <= 2
        if predictions.dim() == 2 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)

        if is_binary and predictions.ndim == 1:
            predictions = where(predictions >= 0.5, 1, 0)
            num_classes = [1]
        elif predictions.ndim > 1:
            num_classes = list(range(predictions.size(1)))
            predictions = argmax(predictions, dim=1).long()
        else:
            num_classes = [1]
        return predictions, labels, num_classes


class MeanSquaredError(Metric):
    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        if not isinstance(labels, Tensor):
            labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        mse = ((labels - predictions) ** 2).mean().item()
        return mse

    def __str__(self) -> str:
        return "mean_squared_error"


class MeanAbsoluteError(Metric):
    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        if not isinstance(labels, Tensor):
            labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        mae = abs(labels - predictions).mean().item()
        return mae

    def __str__(self) -> str:
        return "mean_absolute_error"


class RSquared(Metric):
    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        if not isinstance(labels, Tensor):
            labels = to_tensor(labels)
        residual_sum_se = ((labels - predictions) ** 2).sum().item()
        total_sum_se = ((labels - labels.mean()) ** 2).sum().item()
        if total_sum_se == 0:
            rs = 1.0 if residual_sum_se == 0 else 0.0
        else:
            rs = 1 - residual_sum_se / total_sum_se
        return rs

    def __str__(self) -> str:
        return "r_squared"


class Accuracy(Metric):
    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        if not isinstance(labels, Tensor):
            labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        if self.needs_activation:
            predictions = self._select_activation(labels)(predictions)
        predictions, labels, _ = self._preprocess_tensors(predictions, labels)
        num_correct = (predictions == labels).sum()
        accuracy = num_correct / labels.size(0)
        return accuracy.item()

    def __str__(self) -> str:
        return "accuracy"


class Precision(Metric):
    def __call__(self, predictions: Tensor, labels: np.ndarray) -> float:
        if not isinstance(labels, Tensor):
            labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        if self.needs_activation:
            predictions = self._select_activation(labels)(predictions)
        predictions, labels, classes = self._preprocess_tensors(predictions, labels)

        precision_list = []
        for cls in classes:
            tp = ((predictions == cls) & (labels == cls)).sum().item()
            fp = ((predictions == cls) & (labels != cls)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_list.append(precision)

        return Tensor(precision_list).mean().item()

    def __str__(self) -> str:
        return "precision"

class Recall(Metric):
    def __call__(self, predictions: Tensor, labels: np.ndarray) -> float:
        if not isinstance(labels, Tensor):
            labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        if self.needs_activation:
            predictions = self._select_activation(labels)(predictions)
        predictions, labels, classes = self._preprocess_tensors(predictions, labels)

        recall_list = []
        for cls in classes:
            tp = ((predictions == cls) & (labels == cls)).sum().item()
            fn = ((predictions != cls) & (labels == cls)).sum().item()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_list.append(recall)

        return Tensor(recall_list).mean().item()

    def __str__(self) -> str:
        return "recall"


class F1Score(Metric):
    def __init__(self, needs_activation: bool=True) -> None:
        self.precision = Precision(needs_activation)
        self.recall = Recall(needs_activation)
        super().__init__(needs_activation)

    def __call__(
            self, predictions: Tensor, labels: np.ndarray
    ) -> float:
        precision = self.precision(predictions, labels)
        recall = self.recall(predictions, labels)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1

    def __str__(self) -> str:
        return "f1_score"


class CrossEntropyLoss(Metric):
    def __call__(
            self,
            predictions: Tensor,
            labels: np.ndarray
    ) -> float:
        if not isinstance(labels, Tensor):
            labels = to_tensor(labels)
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        if labels.dtype != long:
            labels = labels.long()
        if self.needs_activation:
            predictions = self._select_activation(labels)(predictions)
        predictions = predictions.clamp(min=1e-7, max=1 - 1e-7)
        # Binary case: handle both [B, 1] and [B] shapes
        if predictions.dim() == 2 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1) # Convert [B, 1] to [B]
        if predictions.dim() == 1:
            loss_tensor = -(
                labels * log(predictions) + (1 - labels) * log(1 - predictions)
            )
        else:
            loss_tensor = -log(predictions[range(predictions.size(0)), labels])
        return loss_tensor.mean()

    def __str__(self) -> str:
        return "cross_entropy_loss"
