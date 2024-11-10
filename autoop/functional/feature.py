from typing import List

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(
        dataset: Dataset,
        label_as_category: str | None = None
) -> List[Feature]:
    """
    Detect the feature type for each feature in Dataset.

    Args:
        dataset: Dataset object.
        label_as_category: The column name of the label
            to be treated as a categorical feature.

    Returns:
        List[Feature]: List of features with their types.
    """
    df = dataset.read()
    features = []
    for column in df.columns:
        if label_as_category == column and all(
            isinstance(element, int) for element in df[column]
        ):
            column_type = "categorical"
        elif all(isinstance(element, (float, int)) for element in df[column]):
            column_type = "numerical"
        else:
            column_type = "categorical"
        features.append(Feature(name=column, type=column_type))
    return features
