import unittest

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)

from autoop.core.ml.metric import get_metric


class TestMetrics(unittest.TestCase):
    """Unit tests for metrics."""

    def test_mean_squared_error(self) -> None:
        """Test the mean squared error metric."""
        metric = get_metric("mean_squared_error")
        predictions = torch.tensor([1.0, 2.0, 3.0])
        labels = np.array([1.0, 2.0, 3.0])
        expected = mean_squared_error(labels, predictions.numpy())
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)
        predictions = torch.tensor([2.0, 2.0, 2.0])
        labels = np.array([1.0, 2.0, 3.0])
        expected = mean_squared_error(labels, predictions.numpy())
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)

    def test_mean_absolute_error(self) -> None:
        """Test the mean absolute error metric."""
        metric = get_metric("mean_absolute_error")
        predictions = torch.tensor([1.0, 2.0, 3.0])
        labels = np.array([1.0, 2.0, 3.0])
        expected = mean_absolute_error(labels, predictions.numpy())
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)
        predictions = torch.tensor([2.0, 2.0, 2.0])
        labels = np.array([1.0, 2.0, 3.0])
        expected = mean_absolute_error(labels, predictions.numpy())
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)

    def test_r_squared(self) -> None:
        """Test the R-squared metric."""
        metric = get_metric("r_squared")
        predictions = torch.tensor([3.0, -0.5, 2.0, 7.0])
        labels = np.array([2.5, 0.0, 2.0, 8.0])
        expected = r2_score(labels, predictions.numpy())
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)
        predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
        labels = np.array([2.5, 0.0, 2.0, 8.0])
        expected = r2_score(labels, predictions.numpy())
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)

    def test_accuracy(self) -> None:
        """Test the accuracy metric."""
        metric = get_metric("accuracy", needs_activation=True)
        predictions = torch.tensor([[2.0, 1.0, 0.0],
                                    [0.0, 3.0, 1.0],
                                    [1.0, 0.0, 2.0],
                                    [1.0, 2.0, 0.0]])
        labels = np.array([2, 1, 0, 1])
        preds_np = torch.argmax(predictions, dim=1).numpy()
        expected = accuracy_score(labels, preds_np)
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)
        predictions = torch.tensor([[0, 1, 2],
                                    [0, 2, 1],
                                    [2, 0, 1],
                                    [1, 2, 0]])
        labels = np.array([0, 2, 1, 1])
        preds_np = torch.argmax(predictions, dim=1).numpy()
        expected = accuracy_score(labels, preds_np)
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)

    def test_precision(self) -> None:
        """Test the precision metric."""
        metric = get_metric("precision", needs_activation=True)
        predictions = torch.tensor([[0.1, 0.7, 0.2],
                                    [0.3, 0.4, 0.3],
                                    [0.2, 0.2, 0.6],
                                    [0.5, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 1])
        preds_np = torch.argmax(predictions, dim=1).numpy()
        expected = precision_score(
            labels, preds_np, average='macro', zero_division=0
        )
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)

    def test_recall(self) -> None:
        """Test the recall metric."""
        metric = get_metric("recall", needs_activation=True)
        predictions = torch.tensor([[0.1, 0.7, 0.2],
                                    [0.3, 0.4, 0.3],
                                    [0.2, 0.2, 0.6],
                                    [0.5, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 1])
        preds_np = torch.argmax(predictions, dim=1).numpy()
        expected = recall_score(
            labels, preds_np, average='macro', zero_division=0
        )
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)

    def test_f1_score(self) -> None:
        """Test the F1 score metric."""
        metric = get_metric("f1_score", needs_activation=True)
        predictions = torch.tensor([[0.1, 0.7, 0.2],
                                    [0.3, 0.4, 0.3],
                                    [0.2, 0.2, 0.6],
                                    [0.5, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 1])
        preds_np = torch.argmax(predictions, dim=1).numpy()
        expected = f1_score(labels, preds_np, average='macro', zero_division=0)
        self.assertAlmostEqual(metric(predictions, labels), expected, places=6)
