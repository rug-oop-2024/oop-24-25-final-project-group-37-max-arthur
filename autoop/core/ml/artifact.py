import base64


class Artifact:
    def __init__(
            self,
            name: str,
            data: bytes,
            asset_path: str,
            type: str = "",
            version: str = "1.0.0",
            tags: list[str] = [],
            metadata: dict = {}
    ) -> None:
        self._id = None
        self.type = type
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version
        self.tags = tags
        self.metadata = metadata

    @property
    def id(self):
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        return self.data

    def save(self, data: bytes) -> bytes:  # ????? Can remove if not used
        self.data = data                   # i guess we are changing the data
        return data
