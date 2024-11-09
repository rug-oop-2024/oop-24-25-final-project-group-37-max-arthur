import base64


class Artifact:
    """
    A class representing a data artifact with metadata and version control.

    Attributes:
        name (str): The name of the artifact.
        data (bytes): The binary data of the artifact.
        asset_path (str): The file path or identifier for the artifact.
        type (str): The type or category of the artifact.
        version (str): Version identifier of the artifact. Default is '1.0.0'.
        tags (list[str]): List of tags associated with the artifact.
        metadata (dict): Additional metadata for the artifact.

    Methods:
        read() -> bytes: Read the artifact data.
        save(data: bytes) -> bytes: Save new data to the artifact.
    """

    def __init__(
            self,
            name: str,
            data: bytes,
            asset_path: str = None,
            type: str = "",
            version: str = "1.0.0",
            tags: list[str] = [],
            metadata: dict = {}
    ) -> None:
        """
        Initialize the Artifact with its attributes.

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
        self._id = None
        self.type = type
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.tags = tags
        self.metadata = metadata

    @property
    def id(self) -> str:
        """
        Dynamically generate a unique identifier for the artifact based.

        Returns:
            str: A base64 encoded identifier of the format
                'encoded_path:version'.
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """
        Read the artifact data.

        Returns:
            bytes: The binary data of the artifact.
        """
        return self.data

    def save(self, data: bytes) -> bytes:
        """
        Save new data to the artifact.

        Args:
            data (bytes): The new data to store in the artifact.

        Returns:
            bytes: The updated binary data of the artifact.
        """
        self.data = data
        return data
