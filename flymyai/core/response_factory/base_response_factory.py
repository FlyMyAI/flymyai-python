from abc import abstractmethod

import httpx

from flymyai.core._response import FlyMyAIResponse
from flymyai.core._streaming import ServerSentEvent
from flymyai.core.exceptions import BaseFlyMyAIException


class ResponseFactoryException(Exception): ...


class ResponseFactory(object):
    """
    Factory for FlyMyAIResponse objects
    """

    def __init__(
        self,
        httpx_response: httpx.Response = None,
        httpx_request: httpx.Request = None,
        *_,
        **__
    ):
        if httpx_response and not httpx_request:
            self.httpx_request = httpx_response.request
        else:
            self.httpx_request = httpx_request
        self.httpx_response = httpx_response

    def _base_construct_from_httpx_response(self):
        if self.httpx_response.status_code < 400:
            return FlyMyAIResponse.from_httpx(self.httpx_response)
        else:
            raise BaseFlyMyAIException.from_response(
                FlyMyAIResponse.from_httpx(self.httpx_response)
            )

    @abstractmethod
    def construct(self):
        raise NotImplementedError
