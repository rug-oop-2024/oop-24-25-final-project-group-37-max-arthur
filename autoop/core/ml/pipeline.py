from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8,
                 ):
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError("Model type must be classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous target feature")

    def __str__(self):
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
    def model(self):
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Used to get the artifacts generated during the pipeline execution to be saved
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
        artifacts.append(Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact):
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        (target_feature_name, target_data, artifact) = preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in input_results]

    def _split_data(self):
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in self._input_vectors]
        self._train_y = self._output_vector[:int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        return np.concatenate(vectors, axis=1)

    def _train(self):
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self):
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

    def execute(self):
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }

    def to_artifact(self, name: str) -> 'Artifact':
        model_artifact = next(artifact for artifact in self.artifacts if artifact.name.startswith("pipeline_model"))
        data = {
            "model": model_artifact.id,
            "dataset": self._dataset.id,
            "metrics": [str(metric) for metric in self._metrics],
            "target_feature": {"name": self._target_feature.name, "type": self._target_feature.type},
            "input_features": [{"name": feature.name, "type": feature.type} for feature in self._input_features],
            "split": self._split
        }
        return Artifact(name=name, data=pickle.dumps(data), asset_path=f"pipeline/pipeline_of_{model_artifact.type}", type="pipeline")

    @staticmethod
    def from_artifact(metrics: List[str],
                 dataset_id: str,
                 model_id: str,
                 input_features: List[dict],
                 target_feature: dict,
                 split:int,
                 automl):
        dataset = automl.registry.get(dataset_id)
        
