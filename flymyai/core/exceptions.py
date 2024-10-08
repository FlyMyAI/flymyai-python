import datetime
from typing import List, Type

from ._response import FlyMyAIResponse
from .models.error_responses import (
    FlyMyAI401Response,
    FlyMyAI422Response,
    Base4xxResponse,
    FlyMyAI400Response,
    FlyMyAI421Response,
)


class ImproperlyConfiguredClientException(Exception): ...


class BaseFlyMyAIException(Exception):
    msg: str
    requires_retry: bool
    _response: FlyMyAIResponse

    def __init__(self, msg, requires_retry=False, response=None):
        self.msg = msg
        self.requires_retry = requires_retry
        self._response = response

    @property
    def response(self):
        return self._response

    @classmethod
    def from_5xx(cls, response: FlyMyAIResponse):
        msg = f"""
                INTERNAL SERVER ERROR ({response.status_code}):
                REQUEST URL: {response.url};
                Content [0:250]: {response.content.decode()[0:250]}
                Timestamp [UTC]: {datetime.datetime.utcnow()}
        """
        internal_error_mapping = {
            500: lambda: cls(msg, False, response=response),
            502: lambda: cls(msg, True, response=response),
            503: lambda: cls(msg, False, response=response),
            504: lambda: cls(msg, True, response=response),
            524: lambda: cls(msg, True, response=response),
            # unknown issue, probably detected on the client side
            599: lambda: cls(msg, False, response=response),
            # broker issues, they are not billed at all
            5000: lambda: cls(msg, False, response=response),
            5320: lambda: cls(msg, True, response=response),
        }
        return internal_error_mapping.get(
            response.status_code, lambda: cls(msg, False)
        )()

    @classmethod
    def from_4xx(cls, response: FlyMyAIResponse):
        response_validation_templates = {
            400: FlyMyAI400Response,
            401: FlyMyAI401Response,
            421: FlyMyAI421Response,
            422: FlyMyAI422Response,
        }
        response_4xx = response_validation_templates.get(
            response.status_code, Base4xxResponse
        ).from_response(response)
        return cls(
            msg=response_4xx.to_msg(),
            requires_retry=response_4xx.requires_retry,
            response=response,
        )

    @classmethod
    def from_response(cls, response: FlyMyAIResponse):
        if 400 <= response.status_code < 500:
            return cls.from_4xx(response)
        if response.status_code >= 500:
            return cls.from_5xx(response)

    def __str__(self):
        return self.msg


class FlyMyAIPredictException(BaseFlyMyAIException):
    @classmethod
    def from_base_exception(cls, exception: BaseFlyMyAIException):
        return cls(exception.msg, exception.requires_retry, exception.response)


class FlyMyAIOpenAPIException(BaseFlyMyAIException): ...


class FlyMyAIExceptionGroup(Exception):
    def __init__(self, errors: List[BaseFlyMyAIException], **kwargs):
        self.errors = errors
        exceptions_message = ";".join([str(err) for err in errors])
        self.message = f"FlyMyAI exception history: {exceptions_message}"
        super().__init__(self.message)

    def __str__(self):
        return self.message
