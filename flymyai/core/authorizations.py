import dataclasses

import httpx


class ClientInfo:
    @property
    def authorization_headers(self):
        raise NotImplemented()

    @property
    def _project_path(self):
        raise NotImplemented

    @property
    def prediction_path(self):
        raise NotImplemented

    @property
    def openapi_schema_path(self):
        raise NotImplemented


@dataclasses.dataclass
class APIKeyClientInfo(ClientInfo):
    apikey: str
    username: str
    project_name: str

    @property
    def authorization_headers(self):
        return {"X-API-KEY": self.apikey}

    @property
    def _project_path(self):
        return httpx.URL(f"/api/v1/{self.username}/{self.project_name}/")

    @property
    def prediction_path(self):
        return self._project_path.join(httpx.URL("predict"))

    @property
    def openapi_schema_path(self):
        return self._project_path.join(httpx.URL("openapi.json"))


class ClientInfoFactory:
    _raw_auth: dict

    def __init__(self, raw_auth: dict):
        self._raw_auth = raw_auth

    def _build_auth(self) -> ClientInfo:
        if "apikey" in self._raw_auth:
            return APIKeyClientInfo(**self._raw_auth)
        else:
            raise NotImplemented("This type of authorization is not implemented yet!")

    def build_auth(self):
        return self._build_auth()
