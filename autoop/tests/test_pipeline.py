import unittest

import pandas as pd
from sklearn.datasets import fetch_openml

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import MeanSquaredError
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types


class TestPipeline(unittest.TestCase):
    """Unit tests for the Pipeline class."""

    def setUp(self) -> None:
        """Set up for the test."""
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(),
            input_features=list(
                filter(lambda x: x.name != "age", self.features)
            ),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self) -> None:
        """Test that the Pipeline instance is initialized correctly."""
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self) -> None:
        """Test that features are preprocessed and artifacts are created."""
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self) -> None:
        """Test that data is split correctly into training and testing sets."""
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(
            self.pipeline._train_X[0].shape[0], int(0.8 * self.ds_size)
        )
        self.assertEqual(
            self.pipeline._test_X[0].shape[0], self.ds_size - int(
                0.8 * self.ds_size
            )
        )

    def test_train(self) -> None:
        """Test that the model is trained and parameters are updated."""
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self) -> None:
        """Test that the model evaluation functions as expected."""
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        self.assertEqual(len(self.pipeline._metrics_results), 1)
