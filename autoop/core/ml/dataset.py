import io

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """
    A class representing a dataset artifact.

    Attributes:
        name (str): The name of the artifact.
        data (bytes): The binary data of the artifact.
        asset_path (str): The file path or identifier for the artifact.
        type (str): The type or category of the artifact.
        version (str): Version identifier of the artifact. Default is '1.0.0'.
        tags (list[str]): List of tags associated with the artifact.
        metadata (dict): Additional metadata for the artifact.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Dataset artifact, setting its type to "dataset".

        Args:
            name (str): The name of the artifact.
            data (bytes): The binary data associated with the artifact.
            asset_path (str, optional): The path or identifier of the asset.
                Default is None.
            type (str, optional): The type or category of the artifact.
                Default is an empty string.
            version (str, optional): The version of the artifact.
                Default is '1.0.0'.
            tags (list[str], optional): Tags for categorizing the artifact.
                Default is an empty list.
            metadata (dict, optional): Additional metadata for the artifact.
                Default is an empty dictionary.

        Returns:
            None
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ) -> 'Dataset':
        """
        Create a Dataset artifact from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be saved as a
                Dataset artifact.
            name (str): The name of the dataset artifact.
            asset_path (str): The path or identifier for the dataset artifact.
            version (str, optional): Version identifier for the dataset.
                Default is "1.0.0".

        Returns:
            Dataset: An instance of Dataset artifact based on the dataframe.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Return the dataframe from the Dataset artifact.

        Returns:
            pd.DataFrame: The Dataset's data as a pandas DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save a new pandas DataFrame to the dataset artifact.

        Encodes the DataFrame to binary CSV format and updates
        the artifact's data.

        Args:
            data (pd.DataFrame): The DataFrame to be saved to the artifact.

        Returns:
            bytes: The updated binary data of the dataset.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
