import unittest
import torch
from autoop.core.ml.metric import get_metric
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score



class TestMetrics(unittest.TestCase):
    def test_mean_squared_error(self):
        predictions = torch.tensor([3.0, -0.5, 2.0, 7.0])
        labels = torch.tensor([2.5, 0.0, 2.0, 8.0])
        custom_mse = get_metric("mean_squared_error", needs_activation=False)(predictions, labels)
        sklearn_mse = mean_squared_error(labels.numpy(), predictions.numpy())
        self.assertAlmostEqual(custom_mse, sklearn_mse, places=5)

    def test_mean_absolute_error(self):
        predictions = torch.tensor([3.0, -0.5, 2.0, 7.0])
        labels = torch.tensor([2.5, 0.0, 2.0, 8.0])
        custom_mae = get_metric("mean_absolute_error", needs_activation=False)(predictions, labels)
        sklearn_mae = mean_absolute_error(labels.numpy(), predictions.numpy())
        self.assertAlmostEqual(custom_mae, sklearn_mae, places=5)

    def test_r_squared(self):
        predictions = torch.tensor([3.0, -0.5, 2.0, 7.0])
        labels = torch.tensor([2.5, 0.0, 2.0, 8.0])
        custom_r2 = get_metric("r_squared", needs_activation=False)(predictions, labels)
        sklearn_r2 = r2_score(labels.numpy(), predictions.numpy())
        self.assertAlmostEqual(custom_r2, sklearn_r2, places=5)

    def test_accuracy(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.4])
        labels = torch.tensor([1, 0, 1, 0])
        custom_accuracy = get_metric("accuracy", needs_activation=False)(predictions, labels)
        sklearn_accuracy = accuracy_score(labels.numpy(), (predictions.numpy() >= 0.5).astype(int))
        self.assertAlmostEqual(custom_accuracy, sklearn_accuracy, places=5)

    def test_precision(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.4])
        labels = torch.tensor([1, 0, 1, 0])
        custom_precision = get_metric("precision", needs_activation=False)(predictions, labels)
        sklearn_precision = precision_score(labels.numpy(), (predictions.numpy() >= 0.5).astype(int), average='macro')
        self.assertAlmostEqual(custom_precision, sklearn_precision, places=5)

    def test_recall(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.4])
        labels = torch.tensor([1, 0, 1, 0])
        custom_recall = get_metric("recall", needs_activation=False)(predictions, labels)
        sklearn_recall = recall_score(labels.numpy(), (predictions.numpy() >= 0.5).astype(int), average='macro')
        self.assertAlmostEqual(custom_recall, sklearn_recall, places=5)

    def test_f1_score(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.4])
        labels = torch.tensor([1, 0, 1, 0])
        custom_f1 = get_metric("f1_score", needs_activation=False)(predictions, labels)
        sklearn_f1 = f1_score(labels.numpy(), (predictions.numpy() >= 0.5).astype(int), average='macro')
        self.assertAlmostEqual(custom_f1, sklearn_f1, places=5)

    def test_cross_entropy_loss(self):
        predictions = torch.tensor([0.9, 0.2, 0.8, 0.4])
        labels = torch.tensor([1, 0, 1, 0], dtype=torch.float)
        custom_ce = get_metric("cross_entropy_loss", needs_activation=False)(predictions, labels)
        loss_torch = torch.nn.BCELoss()
        torch_cle = loss_torch(predictions, labels).item()
        self.assertAlmostEqual(custom_ce, torch_cle, places=5)

    def test_multiclass_cross_entropy_loss(self):
        predictions = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [0.9, 0.1, 2.2]])
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        custom_ce = get_metric("cross_entropy_loss")(predictions, labels)
        loss_torch = torch.nn.CrossEntropyLoss()
        torch_cle = loss_torch(predictions, labels)
        self.assertAlmostEqual(custom_ce, torch_cle.item(), places=5)
