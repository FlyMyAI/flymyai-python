import httpx

from flymyai.core._response import FlyMyAIResponse
from flymyai.core._streaming import ServerSentEvent
from flymyai.core.exceptions import BaseFlyMyAIException
from flymyai.core.response_factory.base_response_factory import (
    ResponseFactory,
    ResponseFactoryException,
)


class SSEInferenceResponseFactory(ResponseFactory):
    def __init__(
        self,
        httpx_response: httpx.Response = None,
        httpx_request: httpx.Request = None,
        sse: ServerSentEvent = None,
    ):
        if not httpx_response and not sse:
            raise ResponseFactoryException("httpx_response and sse params required")
        super().__init__(httpx_response, httpx_request)
        self.sse = sse

    def get_sse_status_code(self):
        return self.sse.json().get(
            "status", self.httpx_response.status_code if self.httpx_response else 200
        )

    def _base_construct_from_sse(self):
        sse_status = self.get_sse_status_code()
        is_details = self.sse.json().get("details") is not None
        if is_details and sse_status == 200:
            sse_status = 599
        if sse_status < 400 and not is_details:
            response = FlyMyAIResponse(
                status_code=sse_status,
                content=self.sse.data or self.sse.event,
                request=self.httpx_request,
                headers=self.httpx_response.headers or self.sse.headers,
            )
            response.is_event = self.sse.event is not None
            return response
        else:
            raise BaseFlyMyAIException.from_response(
                FlyMyAIResponse(
                    status_code=sse_status,
                    content=self.sse.data or self.sse.event,
                    request=self.httpx_request,
                    headers=self.httpx_response.headers
                    or getattr(self.sse, "headers", {}),
                )
            )

    def construct(self):
        if self.sse:
            return self._base_construct_from_sse()
        elif self.httpx_response:
            return self._base_construct_from_httpx_response()
