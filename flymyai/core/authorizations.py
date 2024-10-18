import copy
import dataclasses
from typing import Optional

import httpx

from flymyai.core.exceptions import ImproperlyConfiguredClientException


class ClientInfo:
    """
    Base class for all ClientInfo objects
    """

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
    """
    Encapsulates information about a project.
    Uses X-API-KEY header to perform an auth
    """

    apikey: str
    username: Optional[str] = None
    project_name: Optional[str] = None

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
    def prediction_async_path(self):
        return self._project_path.join(httpx.URL("predict/async/"))

    @property
    def prediction_result_path(self):
        return self._project_path.join(httpx.URL("predict/async/result/"))

    @property
    def prediction_cancel_path(self):
        return self._project_path.join(httpx.URL("predict/cancel/"))

    @property
    def prediction_stream_path(self):
        return self._project_path.join(httpx.URL("predict/stream/"))

    @property
    def openapi_schema_path(self):
        return self._project_path.join(httpx.URL("openapi.json"))

    def copy_for_model(self, model: str):
        copied = copy.deepcopy(self)
        if not model:
            raise ImproperlyConfiguredClientException(
                "model should be provided as <owner username>/<model>"
            )
        split_info = model.split("/")
        if len(split_info) != 2:
            raise ImproperlyConfiguredClientException(
                "model should be provided as <owner username>/<model>"
            )
        copied.username = split_info[0]
        copied.project_name = split_info[1]
        return copied
