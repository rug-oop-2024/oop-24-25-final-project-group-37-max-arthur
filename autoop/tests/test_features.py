import unittest

import pandas as pd
from sklearn.datasets import fetch_openml, load_iris

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


class TestFeatures(unittest.TestCase):
    """Unit tests for feature detection."""

    def setUp(self) -> None:
        """Set up the test."""
        pass

    def test_detect_features_continuous(self) -> None:
        """Test that continuous features are correctly detected."""
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "numerical")

    def test_detect_features_with_categories(self) -> None:
        """Test that categorical features are correctly detected."""
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        numerical_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(
            lambda x: x.name in numerical_columns, features
        ):
            self.assertEqual(detected_feature.type, "numerical")
        for detected_feature in filter(
            lambda x: x.name in categorical_columns, features
        ):
            self.assertEqual(detected_feature.type, "categorical")
