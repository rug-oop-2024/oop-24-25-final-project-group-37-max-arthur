from abc import ABC, abstractmethod
from typing import Any  # !why is this here?
import torch
from torch.nn import functional as F

METRICS = [
    "mean_squared_error",
    "accuracy",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str) -> 'Metric':
    """
    Factory function to get a metric by name.
    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Metric: An instance of the corresponding Metric class.

    Raises:
        ValueError: If the name is not a valid metric.
    """
    if name.lower() in ["mean_squared_error", "mse"]:
        return MeanSquaredError()
    elif name.lower() == "accuracy":
        return Accuracy()
    else:
        raise ValueError(f"Unknown metric: {name}")


class Metric(ABC):
    """Base class for all metrics.
    """
    @abstractmethod
    def __call__(
            self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> float:
        pass


class MeanSquaredError(Metric):
    def __call__(
            self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> float:
        assert len(predictions) == len(labels)
        return F.mse_loss(predictions, labels).item()


class Accuracy(Metric):
    def __call__(
            self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> float:
        assert len(predictions) == len(labels)
        if predictions.ndim > 1:
            predictions = torch.argmax(predictions, dim=1)
        num_correct = (predictions == labels).sum().item()
        return num_correct / len(labels)
