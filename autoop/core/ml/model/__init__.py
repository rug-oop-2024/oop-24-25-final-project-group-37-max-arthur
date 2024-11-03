
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.classification import LogisticRegression

REGRESSION_MODELS = [
    "MultipleLinearRegression"
]

CLASSIFICATION_MODELS = [
    "LogisticRegression"
]

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression()
    
    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "LogisticRegression":
            return LogisticRegression()
    
    else:
        raise ValueError(f"Model '{model_name}' not found in available models.")
    