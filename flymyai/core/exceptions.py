from ._response import FlyMyAIResponse
from .models import (
    FlyMyAI401Response,
    FlyMyAI422Response,
    Base4xxResponse,
    FlyMyAI400Response,
)


class BaseFlyMyAIException(Exception):
    msg: str
    requires_retry: bool

    def __init__(self, msg, requires_retry=False):
        self.msg = msg
        self.requires_retry = requires_retry

    @classmethod
    def from_5xx(cls, response: FlyMyAIResponse):
        msg = f"""
                INTERNAL SERVER ERROR ({response.status_code}):
                REQUEST URL: {response.url};
            """
        internal_error_mapping = {
            500: lambda: cls(msg, False),
            502: lambda: cls(msg, True),
            503: lambda: cls(msg, False),
            504: lambda: cls(msg, True),
            524: lambda: cls(msg, True),
        }
        return internal_error_mapping.get(
            response.status_code, lambda: cls(msg, False)
        )()

    @classmethod
    def from_4xx(cls, response: FlyMyAIResponse):
        response_validation_templates = {
            400: FlyMyAI400Response,
            401: FlyMyAI401Response,
            422: FlyMyAI422Response,
        }
        response_4xx = response_validation_templates.get(
            response.status_code, Base4xxResponse
        ).from_response(response)
        return cls(
            msg=response_4xx.to_msg(), requires_retry=response_4xx.requires_retry
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
    ...


class FlyMyAIExceptionGroup(Exception):
    def __init__(self, errors: list[BaseFlyMyAIException], **kwargs):
        self.errors = errors
        exceptions_message = ";".join([str(err) for err in errors])
        self.message = f"FlyMyAI exception history: {exceptions_message}"
        super().__init__(self.message)

    def __str__(self):
        return self.message
