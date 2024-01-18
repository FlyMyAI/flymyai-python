import httpx

from flymyai.core._response import FlyMyAIResponse
from flymyai.core._streaming import ServerSentEvent
from flymyai.core.exceptions import BaseFlyMyAIException


class ResponseFactoryException(Exception):
    ...


class ResponseFactory(object):
    """
    Factory for FlyMyAIResponse objects
    """

    def __init__(
        self,
        httpx_response: httpx.Response = None,
        httpx_request: httpx.Request = None,
        sse: ServerSentEvent = None,
    ):
        if not httpx_response and not sse:
            raise ResponseFactoryException("httpx_response and sse params required")
        if httpx_response and not httpx_request:
            self.httpx_request = httpx_response.request
        else:
            self.httpx_request = httpx_request
        self.sse = sse
        self.httpx_response = httpx_response

    def get_sse_status_code(self):
        return self.sse.json().get("status_code", 200)

    def _base_construct_from_sse(self):
        sse_status = self.get_sse_status_code()
        if sse_status < 400:
            return FlyMyAIResponse(
                status_code=sse_status,
                content=self.sse.data or self.sse.event,
                request=self.httpx_request,
                headers=self.httpx_response.headers or self.sse.headers,
            )
        else:
            raise BaseFlyMyAIException.from_response(
                FlyMyAIResponse(
                    status_code=sse_status,
                    content=self.sse.data or self.sse.event,
                    request=self.httpx_request,
                    headers=self.httpx_response.headers or self.sse.headers,
                )
            )

    def _base_construct_from_httpx_response(self):
        if self.httpx_response.status_code < 400:
            return FlyMyAIResponse.from_httpx(self.httpx_response)
        else:
            raise BaseFlyMyAIException.from_response(
                FlyMyAIResponse.from_httpx(self.httpx_response)
            )

    def construct(self):
        if self.sse:
            return self._base_construct_from_sse()
        elif self.httpx_response:
            return self._base_construct_from_httpx_response()
