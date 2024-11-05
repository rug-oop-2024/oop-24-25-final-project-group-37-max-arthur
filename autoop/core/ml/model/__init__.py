
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression, Lasso, ElasticNet
from autoop.core.ml.model.classification import LogisticRegression, RandomForestClassifier, MLP

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "ElasticNet"
]

CLASSIFICATION_MODELS = [
    "LogisticRegression",
    "MultiLayerPerceptron",
    "RandomForestClassifier"
]

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    model_dict = {
        "logisticregression": LogisticRegression,
        "multilayerperceptron": MLP,
        "randomforestclassifier": RandomForestClassifier,
        "multiplelinearregression": MultipleLinearRegression,
        "lasso": Lasso,
        "elasticnet": ElasticNet


    }
    if model_name.lower() not in model_dict:
        raise ValueError(
            f"Unknown model: {model_name}, valid models are: {model_dict.values()}"
        )
    return model_dict[model_name.lower()]
