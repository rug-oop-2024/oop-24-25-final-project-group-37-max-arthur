import base64  # i think this should be used?
import uuid

class Artifact:
    def __init__(
            self,
            name: str,
            data: bytes,
            type: str = "",
            asset_path: str = None,
            version: str = "1.0.0",
            tags: list[str] = None,
            metadata: dict = None,
            id: str = None
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.type = type
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.tags = tags or []
        self.metadata = metadata or {}

    def read(self) -> bytes:
        return self.data

    def save(self, data: bytes) -> bytes:  # ?????
        self.data = data
        return self.data
