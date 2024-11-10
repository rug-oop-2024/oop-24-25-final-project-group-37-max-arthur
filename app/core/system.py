from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    def __init__(self,
                 database: Database,
                 storage: Storage):
        """
        Initializes the System class with a database and storage.

        Args:
            database (Database): The database instance to be used by the
            system.
            storage (Storage): The storage instance to be used by the system.
        """

        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact by saving its data to storage and its metadata
        to the database.

        Args:
            artifact (Artifact): The artifact to be registered.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists artifacts from the database, optionally filtered by type.

        Args:
            type (str, optional): The type of artifacts to filter by.
            If None, all artifacts are listed. Defaults to None.
        Returns:
            List[Artifact]: A list of Artifact objects.
        """

        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve an artifact from the database and storage.

        Args:
            artifact_id (str): The unique identifier of the artifact to
            retrieve.
        Returns:
            Artifact: An instance of the Artifact class populated with data
            from the database and storage.
        """

        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact from the database and its associated asset from
        storage.

        Args:
            artifact_id (str): The unique identifier of the artifact to be
            deleted.
        Returns:
            None
        """

        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes the System class with the provided storage and database.

        Args:
            storage (LocalStorage): An instance of LocalStorage to handle file
            storage operations.
            database (Database): An instance of Database to handle database
            operations.
        """

        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """
        Retrieves the singleton instance of the AutoMLSystem class. If the
        instance does not exist, it initializes it with LocalStorage and
        Database objects.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem class.
        """

        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        return self._registry
