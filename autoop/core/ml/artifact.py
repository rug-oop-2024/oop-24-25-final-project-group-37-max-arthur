import base64  # not used, why is this here?


class Artifact:
    def __init__(
            self,
            type: str,
            name: str,
            asset_path: str,
            data: bytes,
            version: str
    ) -> None:
        self.type = type
        self.name = name
        self.asset_path = asset_path
        self.data = data
        self.version = version

    def read(self) -> bytes:
        return self.data

    def save(self, data: bytes) -> bytes:
        raise NotImplementedError("To be implemented.")
