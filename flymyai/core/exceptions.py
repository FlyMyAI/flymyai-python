import datetime
from typing import List, Union

from ._response import FlyMyAIResponse
from .models.base import ResponseLike
from .models.error_responses import (
    FlyMyAI401Response,
    FlyMyAI422Response,
    Base4xxResponse,
    FlyMyAI400Response,
    FlyMyAI421Response,
    FlyMyAI425Response,
    FlyMyAIRetryable4xxResponse,
)


class RetryTimeoutExceededException(TimeoutError): ...


class ImproperlyConfiguredClientException(Exception): ...


class BaseFlyMyAIException(Exception):
    msg: str
    requires_retry: bool
    _response: Union[FlyMyAIResponse, ResponseLike]

    def __init__(self, msg, requires_retry=False, response=None):
        self.msg = msg
        self.requires_retry = requires_retry
        self._response = response

    @property
    def response(self):
        return self._response

    @classmethod
    def internal_error_mapping(cls):
        # Every HTTP 5xx is sent again with backoff, as the official OpenAI and
        # Anthropic SDKs do; the server can still say no with ``x-should-retry: false``
        # (flymyai.core._retry). Codes above 599 come from the old pod broker.
        return {
            500: lambda msg, response: cls(msg, True, response=response),
            502: lambda msg, response: cls(msg, True, response=response),
            503: lambda msg, response: cls(msg, True, response=response),
            504: lambda msg, response: cls(msg, True, response=response),
            524: lambda msg, response: cls(msg, True, response=response),
            # unknown issue, probably detected on the client side
            599: lambda msg, response: cls(msg, False, response=response),
            # broker issues, they are not billed at all
            5000: lambda msg, response: cls(msg, False, response=response),
            5320: lambda msg, response: cls(msg, True, response=response),
        }

    @classmethod
    def from_5xx(cls, response: Union[FlyMyAIResponse, ResponseLike]):
        msg = f"""
                INTERNAL SERVER ERROR ({response.status_code}):
                REQUEST URL: {response.url};
                Content [0:250]: {response.content.decode()[0:250]}
                Timestamp [UTC]: {datetime.datetime.utcnow()}
        """
        retryable = 500 <= response.status_code < 599
        return cls.internal_error_mapping().get(
            response.status_code,
            lambda m, r: cls(m, retryable, response=r),
        )(msg, response)

    @classmethod
    def client_error_mapping(cls):
        return {
            400: FlyMyAI400Response,
            401: FlyMyAI401Response,
            # request timeout, conflict, rate limited: sent again with backoff
            408: FlyMyAIRetryable4xxResponse,
            409: FlyMyAIRetryable4xxResponse,
            421: FlyMyAI421Response,
            422: FlyMyAI422Response,
            # requested too early
            425: FlyMyAI425Response,
            429: FlyMyAIRetryable4xxResponse,
        }

    @classmethod
    def from_4xx(cls, response: Union[FlyMyAIResponse, ResponseLike]):
        response_validation_templates = cls.client_error_mapping()
        response_4xx = response_validation_templates.get(
            response.status_code, Base4xxResponse
        ).from_response(response)
        return cls(
            msg=response_4xx.to_msg(),
            requires_retry=response_4xx.requires_retry,
            response=response,
        )

    @classmethod
    def from_response(cls, response: Union[FlyMyAIResponse, ResponseLike]):
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


class FlyMyAIAsyncTaskException(FlyMyAIPredictException): ...


class FlyMyAIStreamError(FlyMyAIPredictException):
    """
    The server reported an error inside a stream that had already started (an SSE
    event with an error status after HTTP 200). Not sent again: the request reached
    the model. ``prediction_id`` (when the stream announced one) and
    ``events_received`` say how far it got.
    """

    def __init__(
        self,
        msg,
        requires_retry=False,
        response=None,
        *,
        prediction_id=None,
        events_received=0,
    ):
        super().__init__(msg, requires_retry, response)
        self.prediction_id = prediction_id
        self.events_received = events_received

    @classmethod
    def from_base_exception(
        cls, exception: BaseFlyMyAIException, prediction_id=None, events_received=0
    ):
        return cls(
            exception.msg,
            False,
            exception.response,
            prediction_id=prediction_id,
            events_received=events_received,
        )


class FlyMyAIStreamInterrupted(FlyMyAIPredictException):
    """
    The connection broke after the stream had delivered at least one event. The SDK
    never sends such a stream again by itself (the model may still be running, and a
    second run would repeat the output and its cost), as the official OpenAI and
    Anthropic SDKs do not. ``prediction_id`` (when the stream announced one),
    ``events_received`` and the original error (``__cause__``) are kept; retry the
    call yourself if a second run is acceptable.
    """

    def __init__(self, msg, *, prediction_id=None, events_received=0):
        super().__init__(msg, False, None)
        self.prediction_id = prediction_id
        self.events_received = events_received


class FlyMyAIOpenAPIException(BaseFlyMyAIException): ...


class FlyMyAIExceptionGroup(Exception):
    def __init__(self, errors: List[Exception], **kwargs):
        self.errors = errors
        exceptions_message = ";".join([str(err) for err in errors])
        self.message = f"FlyMyAI exception history: {exceptions_message}"
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def fma_errors(self):
        return list(
            filter(lambda err: isinstance(err, BaseFlyMyAIException), self.errors)
        )

    def non_fma_errors(self):
        return list(
            filter(lambda err: not isinstance(err, BaseFlyMyAIException), self.errors)
        )
