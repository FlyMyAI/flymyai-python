import os
from typing import Callable, Awaitable, Generic, TypeVar, Union, overload

import httpx

from flymyai.multipart.payload import MultipartPayload
from flymyai.core.authorizations import APIKeyClientInfo, ClientInfoFactory
from flymyai.core.exceptions import FlyMyAIPredictException, FlyMyAIExceptionGroup
from flymyai.core.models import PredictionResponse, OpenAPISchemaResponse
from flymyai.utils.utils import retryable_callback, aretryable_callback

DEFAULT_RETRY_COUNT = os.getenv("FLYMYAI_MAX_RETRIES", 2)

_PossibleClients = TypeVar(
    "_PossibleClients", bound=Union[httpx.Client, httpx.AsyncClient]
)


_predict_timeout = httpx.Timeout(None, connect=10)


class BaseClient(Generic[_PossibleClients]):
    _client: _PossibleClients
    max_retries: int
    auth: APIKeyClientInfo

    def __init__(self, auth: APIKeyClientInfo | dict, max_retries=DEFAULT_RETRY_COUNT):
        if isinstance(auth, dict):
            self.auth = ClientInfoFactory(auth).build_auth()
        elif isinstance(auth, APIKeyClientInfo):
            self.auth = auth
        else:
            raise TypeError("Invalid credentials. dict required!")
        self._client = self._construct_client()
        self.max_retries = max_retries

    @overload
    async def predict(self, input_data: dict, max_retries=None) -> PredictionResponse:
        ...

    @overload
    def predict(self, input_data: dict, max_retries=None) -> PredictionResponse:
        ...

    def predict(self, input_data: dict, max_retries=None) -> PredictionResponse:
        ...

    @overload
    async def openapi_schema(self, max_retries=None) -> OpenAPISchemaResponse:
        ...

    @overload
    def openapi_schema(self, max_retries=None) -> OpenAPISchemaResponse:
        ...

    def openapi_schema(self, max_retries=None) -> OpenAPISchemaResponse:
        ...

    @staticmethod
    def _wrap_request(request_callback: Callable):
        response = request_callback()
        try:
            return response.raise_for_status()
        except httpx.HTTPError:
            raise FlyMyAIPredictException.from_response(response)

    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        """Close the underlying HTTPX client.

        The client will *not* be usable after this.
        """
        # If an error is thrown while constructing a client, self._client
        # may not be present
        if hasattr(self, "_client"):
            self._client.close()

    def _construct_client(self):
        raise NotImplemented


class BaseSyncClient(BaseClient[httpx.Client]):
    def _construct_client(self):
        return httpx.Client(
            headers=self.auth.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://flymy.ai/"),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()

    def _predict(self, payload: MultipartPayload):
        return self._wrap_request(
            lambda: self._client.post(
                self.auth.prediction_path,
                **payload.serialize(),
                timeout=_predict_timeout,
                headers=self.auth.authorization_headers,
            )
        )

    def predict(self, payload: dict, max_retries=None):
        payload = MultipartPayload(payload)
        history, response = retryable_callback(
            lambda: self._predict(payload),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse(history, response.json())

    def _openapi_schema(self):
        return self._wrap_request(
            lambda: self._client.get(
                self.auth.openapi_schema_path,
                headers=self.auth.authorization_headers,
            )
        )

    def openapi_schema(self, max_retries=None):
        history, response = retryable_callback(
            lambda: self._openapi_schema(),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse(history, response.json())

    @classmethod
    def run_predict(cls, auth: dict, payload: dict):
        auth = ClientInfoFactory(raw_auth=auth).build_auth()
        with cls(auth) as client:
            return client.predict(payload)


class BaseAsyncClient(BaseClient[httpx.AsyncClient]):
    def _construct_client(self):
        return httpx.AsyncClient(
            headers=self.auth.authorization_headers,
            base_url=os.getenv("FLYMYAI_DSN", "https://flymy.ai/"),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_client"):
            await self._client.aclose()

    async def openapi_schema(self, max_retries=None):
        history, response = await aretryable_callback(
            lambda: self._openapi_schema(),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return OpenAPISchemaResponse(history, response.json())

    def _openapi_schema(self):
        return self._wrap_request(
            lambda: self._client.get(
                self.auth.openapi_schema_path,
                headers=self.auth.authorization_headers,
            )
        )

    def _predict(self, payload: MultipartPayload):
        return self._wrap_request(
            lambda: self._client.post(
                self.auth.prediction_path,
                timeout=_predict_timeout,
                **payload.serialize(),
                headers=self.auth.authorization_headers,
            )
        )

    async def predict(self, payload: dict, max_retries=None):
        payload = MultipartPayload(input_data=payload)
        history, response = await aretryable_callback(
            lambda: self._predict(payload),
            max_retries or self.max_retries,
            FlyMyAIPredictException,
            FlyMyAIExceptionGroup,
        )
        return PredictionResponse(history, response.json())

    @staticmethod
    async def _wrap_request(request_callback: Callable[..., Awaitable[httpx.Response]]):
        response = await request_callback()
        try:
            return response.raise_for_status()
        except httpx.HTTPStatusError:
            raise FlyMyAIPredictException.from_response(response)

    async def close(self):
        await self._client.aclose()

    @classmethod
    async def arun_predict(cls, auth: dict, payload: dict):
        auth = ClientInfoFactory(raw_auth=auth).build_auth()
        async with cls(auth) as client:
            return await client.predict(payload)
