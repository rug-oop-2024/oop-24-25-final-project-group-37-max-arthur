from typing import Literal


class Feature():
    """
    A class representing a feature from a dataset.

    Attributes:
        name (str): The name of the feature.
        type (Literal["categorical", "numerical"]): The type of the feature,
            indicating whether it is categorical or numerical.
    """

    def __init__(
            self,
            name: str,
            type: Literal["categorical", "numerical"]
    ) -> None:
        """
        Initialize a Feature with a name and type.

        Args:
            name (str): The name of the feature.
            type (Literal["categorical", "numerical"]): The type of the
                feature, either 'categorical' or 'numerical'.

        Returns:
            None
        """
        self.name = name
        self.type = type

    def __str__(self) -> str:
        """
        Return a string representation of the feature, including name and type.

        Returns:
            str: A formatted string displaying the feature's name and type.
        """
        return f"Name: {self.name} | Type: {self.type}"
