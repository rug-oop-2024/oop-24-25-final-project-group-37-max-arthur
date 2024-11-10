import os
from abc import ABC, abstractmethod
from glob import glob
from typing import List


class NotFoundError(Exception):
    """Exception raised when a specified path is not found."""

    def __init__(self, path: str) -> None:
        """
        Initialize the exception.

        Args:
            path (str): The path that was not found.

        Returns:
            None
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class for storage operations."""

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path to save data.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): Path to load data.

        Returns:
            bytes: Loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path.

        Args:
            path (str): Path to delete data.

        Returns:
            None
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path.

        Args:
            path (str): Path to list.

        Returns:
            list: List of paths.
        """
        pass


class LocalStorage(Storage):
    """Local storage class for saving, loading, deleting, and listing data."""

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize LocalStorage with a specified base path.

        Args:
            base_path (str): The base directory for storage.
                Defaults to "./assets".

        Returns:
            None
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a local file.

        Args:
            data (bytes): Data to be saved.
            key (str): Relative path key for where to save the data.

        Returns:
            None
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a local file.

        Args:
            key (str): Relative path key for where to load the data from.

        Returns:
            bytes: The loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete a local file.

        Args:
            key (str): Relative path key for the file to delete.
                Defaults to root ("/").

        Returns:
            None
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all files under a given prefix path.

        Args:
            prefix (str): Prefix path to filter the listed files.

        Returns:
            List[str]: List of file paths matching the prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """
        Verify that a path exists.

        Args:
            path (str): Path to check.

        Raises:
            NotFoundError: If the path does not exist.

        Returns:
            None
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Construct the full path by joining the base path and a relative path.

        Args:
            path (str): Relative path to join with the base path.

        Returns:
            str: The full path.
        """
        return os.path.join(self._base_path, path)
