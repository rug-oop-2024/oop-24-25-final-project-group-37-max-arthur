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


def get_metric(name: str) -> 'Metric':  # ?be careful of leakage from the returns
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
        residual_sum_se = ((labels - predictions) ** 2).sum()
        total_sum_se = ((labels - labels.mean()) ** 2).sum()
        rs = 1 - residual_sum_se / total_sum_se
        return rs.item()


class Accuracy(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
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
        num_classes = predictions.size(1) if predictions.ndim > 1 else 1
        predictions = self._select_activation(labels).apply(predictions)
        if predictions.ndim > 1:  # multi-class
            predictions = argmax(predictions, dim=1)
        else:  # binary
            predictions = where(predictions >= 0.5, 1, 0)
        
        precision_list = []
        for class_idx in range(num_classes):
            tp = ((predictions == class_idx) & (
                labels == class_idx)).sum().item()
            fp = ((predictions == class_idx) & (
                labels != class_idx)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_list.append(precision)

        precision = sum(precision_list) / num_classes
        return precision


class Recall(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        self._validate_inputs(predictions, labels)
        labels = labels.squeeze()
        num_classes = predictions.size(1) if predictions.ndim > 1 else 1
        predictions = self._select_activation(labels).apply(predictions)
        if predictions.ndim > 1:  # multi-class
            predictions = argmax(predictions, dim=1)
        else:  # binary
            predictions = where(predictions >= 0.5, 1, 0)

        recall_list = []
        for class_idx in range(num_classes):
            tp = ((predictions == class_idx) & (
                labels == class_idx)).sum().item()
            fn = ((predictions != class_idx) & (
                labels == class_idx)).sum().item()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_list.append(recall)

        recall = sum(recall_list) / num_classes
        return recall


class F1Score(Metric):
    def __call__(
            self, predictions: Tensor, labels: Tensor
    ) -> float:
        recall = Recall()
        precision = Precision()
        r = recall(predictions, labels)
        p = precision(predictions, labels)
        f1 = 2 * (r * p) / (r + p)
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
        loss_tensor = -log(predictions[range(predictions.shape[0]), labels])  # this line doesnt work for binary i think
        return loss_tensor.mean()


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
