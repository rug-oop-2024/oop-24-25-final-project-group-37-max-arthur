from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """

    df = dataset.read()
    features = []
    for column in df.columns:
        column_type = "numerical" if all(isinstance(element, float) or isinstance(element, int) for element in df[column]) else "categorical"
        features.append(Feature(name=column, type=column_type))
    return features
