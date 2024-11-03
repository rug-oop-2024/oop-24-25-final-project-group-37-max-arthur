from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

UNIQUE_THRESHOLD = 0.2

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detect the feature type for each feature in Dataset.

    For floats and integers, features are detected as categorical
    if 20% or less of the elements are unique values.

    Args:
        dataset: Dataset object.
    Returns:
        List[Feature]: List of features with their types.
    """
    df = dataset.read()
    features = []
    for column in df.columns:
        unique_ratio = df[column].nunique() / len(df)
        if all(isinstance(element, (float, int)) for element in df[column]):
            column_type = "categorical" if unique_ratio < UNIQUE_THRESHOLD else "numerical"
        else:
            column_type = "categorical"
        features.append(Feature(name=column, type=column_type))
    return features
