import base64

class Artifact:
    def __init__(
            self,
            name: str,
            data: bytes,
            asset_path: str,
            type: str = "",
            version: str = "1.0.0",
            tags: list[str] = None,
            metadata: dict = None
    ) -> None:
        self._id = None
        self.type = type
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.tags = tags or []
        self.metadata = metadata or {}

    @property
    def id(self):
        return {base64(self.asset_path)}:{self.version}

    def read(self) -> bytes:
        return self.data

    def save(self, data: bytes) -> bytes:  # ????? Can remove if not used
        self.data = data
        return self.data
