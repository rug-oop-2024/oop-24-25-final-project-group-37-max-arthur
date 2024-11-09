from autoop.core.ml.model.classification import (MLP, SVC,
                                                 RandomForestClassifier)
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (ElasticNet, Lasso,
                                             MultipleLinearRegression)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "ElasticNet"
]

CLASSIFICATION_MODELS = [
    "SupportVectorClassifier",
    "MultiLayerPerceptron",
    "RandomForestClassifier"
]


def get_model(model_name: str) -> Model:
    """
    Return a model based on a given string.

    Args:
        model_name (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        Model: _description_
    """
    model_dict = {
        "supportvectorclassifier": SVC,
        "multilayerperceptron": MLP,
        "randomforestclassifier": RandomForestClassifier,
        "multiplelinearregression": MultipleLinearRegression,
        "lasso": Lasso,
        "elasticnet": ElasticNet


    }
    if model_name.lower() not in model_dict:
        raise ValueError(
            f"Unknown model: {model_name}, valid models are: "
            f"{model_dict.values()}"
        )
    return model_dict[model_name.lower()]
