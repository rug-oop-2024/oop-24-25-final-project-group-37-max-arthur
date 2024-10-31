from abc import ABC, abstractmethod
from torch import argmax, Tensor, exp, log, abs, where

METRICS = [
    "mean_squared_error",
    "accuracy",
    "cross_entropy_loss",
    "mean_absolute_error",
    "precision",
    "recall",
    "f1_score",
    "r_squared"
]


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
    return metrics_dict[name.lower()]()


#  !Somewhere we should check for feature type maybe?
#  we currently return floats instead of tensor[float] everywhere

class Metric(ABC):
    def _validate_inputs(self, predictions: Tensor, labels: Tensor) -> None:
        assert predictions.size(0) == labels.size(0), (
            "Predictions and labels must have the same length."
        )

    def _select_activation(self, labels: Tensor) -> Tensor:
        is_binary_class = True if labels.max().item() < 2 else False
        return SigmoidActivation() if is_binary_class else SoftmaxActivation()

    @abstractmethod
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        pass


class MeanSquaredError(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        mse = ((labels - predictions) ** 2).mean().item()
        return mse


class MeanAbsoluteError(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        mae = abs(labels - predictions).mean().item()
        return mae


class RSquared(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        residual_sum_se = ((labels - predictions) ** 2).sum().item()
        total_sum_se = ((labels - labels.mean()) ** 2).sum().item()
        rs = 1 - residual_sum_se / total_sum_se if total_sum_se != 0 else 0.0
        return rs


class Accuracy(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        predictions = self._select_activation(labels).apply(predictions)
        if predictions.ndim == 1:
            predictions = where(predictions >= 0.5, 1, 0)
        if predictions.ndim > 1:
            predictions = argmax(predictions, dim=1)
        num_correct = (predictions == labels).sum()
        accuracy = num_correct / labels.size(0)
        return accuracy.item()


class Precision(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        predictions = self._select_activation(labels).apply(predictions)

        # binary case
        if predictions.ndim == 1:
            predictions = where(predictions >= 0.5, 1, 0)
            classes = [1]

        # multi-class case
        else:
            classes = range(predictions.size(1))
            predictions = argmax(predictions, dim=1)
        precision_list = []
        for cls in classes:
            tp = ((predictions == cls) & (labels == cls)).sum().item()
            fp = ((predictions == cls) & (labels != cls)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_list.append(precision)

        precision = Tensor(precision_list).mean().item()
        return precision


class Recall(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        predictions = self._select_activation(labels).apply(predictions)

        # binary case
        if predictions.ndim == 1:
            predictions = where(predictions >= 0.5, 1, 0)
            classes = [1]
        # multi-class case
        else:
            classes = range(predictions.size(1))
            predictions = argmax(predictions, dim=1)

        recall_list = []
        for cls in classes:
            tp = ((predictions == cls) & (labels == cls)).sum().item()
            fn = ((predictions != cls) & (labels == cls)).sum().item()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_list.append(recall)

        recall = Tensor(recall_list).mean().item()
        return recall


class F1Score(Metric):
    def __init__(self) -> None:
        self.precision = Precision()
        self.recall = Recall()
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        precision = self.precision(predictions, labels)
        recall = self.recall(predictions, labels)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1


class CrossEntropyLoss(Metric):
    def __call__(
            self,
            predictions: Tensor,
            labels: Tensor
    ) -> float:
        """_summary_

        Args:
            predictions (Tensor): _description_
            labels (Tensor): _description_

        Returns:
            float: _description_
        """
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        predictions = self._select_activation(labels).apply(predictions)
        predictions = predictions.clamp(min=1e-7, max=1 - 1e-7)
        if predictions.ndim == 1:
            loss_tensor = -(
                labels * log(predictions) + (1 - labels) * log(1 - predictions)
            )
        else:
            loss_tensor = -log(predictions[range(predictions.size(0)), labels])
        return loss_tensor.mean().item()


class ActivationStrategy(ABC):
    @abstractmethod
    def apply(self, predictions: Tensor) -> Tensor:
        pass


class SoftmaxActivation(ActivationStrategy):
    def apply(self, predictions: Tensor) -> Tensor:
        exp_predictions = exp(predictions)
        return exp_predictions / exp_predictions.sum(dim=1, keepdim=True)


class SigmoidActivation(ActivationStrategy):
    def apply(self, predictions: Tensor) -> Tensor:
        return 1 / (1 + exp(-predictions))
