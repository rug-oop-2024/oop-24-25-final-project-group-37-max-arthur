import pickle
from typing import List

import numpy as np
from torch import Tensor

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


class Pipeline():
    """
    A class representing a machine learning pipeline.

    Combines data preprocessing, model training, and evaluation.

    Attributes:
        metrics (List[Metric]): List of evaluation metrics.
        dataset (Dataset): The dataset to be used in the pipeline.
        model (Model): The model to be trained and evaluated.
        input_features (List[Feature]): Features used as model inputs.
        target_feature (Feature): The target feature to be predicted.
        split (float): Fraction of data used for training. Default is 0.8.

    Methods:
        artifacts() -> List[Artifact]:
            Retrieve artifacts generated during pipeline execution.
        execute() -> dict[str, Tensor]:
            Execute the pipeline, returning evaluation metrics and predictions.
        to_artifact(name: str) -> Artifact:
            Serialize the pipeline instance to an artifact.
    """

    def __init__(
            self,
            metrics: List[Metric],
            dataset: Dataset,
            model: Model,
            input_features: List[Feature],
            target_feature: Feature,
            split: float = 0.8,
    ) -> None:
        """
        Initialize the Pipeline.

        Args:
            metrics (List[Metric]): Evaluation metrics for the model.
            dataset (Dataset): Dataset for model training and testing.
            model (Model): Model to be trained and evaluated.
            input_features (List[Feature]): List of features used as model
                inputs.
            target_feature (Feature): Feature to be predicted.
            split (float): Ratio of data to use for training. Default is 0.8.

        Raises:
            ValueError: If model type does not match target feature type.

        Returns:
            None
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and \
           model.type != "classification":
            raise ValueError(
                "Model type must be classification for categorical"
                " target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """
        Return a string representation of the pipeline's configuration.

        Returns:
            str: A formatted string detailing the pipeline's model,
                features, and metrics.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Access the pipeline's model.

        Returns:
            Model: The model instance used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieve artifacts generated during pipeline execution.

        Returns:
            List[Artifact]: Artifacts like encoders and scalers
                saved during execution.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(
            self,
            name: str,
            artifact: dict[str, str | dict]
    ) -> None:
        """
        Register an artifact for later retrieval.

        Args:
            name (str): Name of the artifact.
            artifact (dict[str, str | dict]): The artifact data and its type.

        Returns:
            None
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocess input and target features.

        Returns:
            None
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by
        # feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """
        Split data into training and testing sets based on split ratio.

        Returns:
            None
        """
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenate multiple arrays along the second axis.

        Args:
            vectors (List[np.array]): List of arrays to be concatenated.

        Returns:
            np.array: The concatenated array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Train the model using the training data.

        Returns:
            None
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluate the fitted model on specified metrics.

        Returns:
            None
        """
        X_test = self._compact_vectors(self._test_X)
        X_train = self._compact_vectors(self._train_X)
        Y_test = self._test_y
        Y_train = self._train_y
        self._metrics_results = {}
        predictions_test = self._model.predict(X_test)
        predictions_train = self._model.predict(X_train)
        for metric in self._metrics:
            metric_name = metric.__class__.__name__
            self._metrics_results[metric_name] = {}
            self._metrics_results[metric_name]["test"] = metric.evaluate(
                predictions_test, Y_test
            )
            self._metrics_results[metric_name]["train"] = metric.evaluate(
                predictions_train, Y_train
            )

        self._predictions = predictions_test

    def execute(self) -> dict[str, Tensor]:
        """
        Execute the pipeline.

        Performs preprocessing, splitting, training, and evaluation.

        Returns:
            dict[str, Tensor]: Dictionary with evaluation metrics
                and model predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }

    def to_artifact(self, name: str) -> 'Artifact':
        """
        Serialize the pipeline instance into an artifact.

        Args:
            name (str): The name of the artifact.

        Returns:
            Artifact: Serialized pipeline artifact.
        """
        data = pickle.dumps(self)
        return Artifact(
            name=name,
            data=data,
            asset_path=f"pipeline/pipeline_of_{self.model.type}_{name}",
            type="pipeline"
        )
