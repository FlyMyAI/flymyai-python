from typing import Optional, Iterator, TypeVar, Callable

from flymyai.core._response import FlyMyAIResponse
from flymyai.core.authorizations import APIKeyClientInfo
from flymyai.core.clients.base_client import BaseClient
from flymyai.core.exceptions import BaseFlyMyAIException
from flymyai.core.models.successful_responses import (
    StreamDetails,
    PredictionPartial,
    PredictionEvent,
)
from flymyai.core.stream_iterators.exceptions import StreamCancellationException
from flymyai.core.types.event_types import EventType

_SyncEventCallbackType = TypeVar(
    "_SyncEventCallbackType", bound=Callable[[PredictionEvent], None]
)


class PredictionStream:
    stream_details: StreamDetails
    event_callback: _SyncEventCallbackType = None
    prediction_id: str
    follow_cancelling: bool = True

    _client: BaseClient
    _client_info: APIKeyClientInfo

    def __init__(
        self,
        response_iterator: Iterator,
        client: BaseClient,
        client_info: APIKeyClientInfo,
    ):
        self.response_iterator = response_iterator
        self._client = client
        self._client_info = client_info

    def cancel(self):
        if not hasattr(self, "prediction_id"):
            raise StreamCancellationException("No prediction_id obtained!")
        return self._client.cancel_prediction(
            self.prediction_id, client_info=self._client_info
        )

    def set_on_event(self, callback: _SyncEventCallbackType):
        self.event_callback = callback

    def __iter__(self):
        return self

    def loop_iter(self):
        response_end = None
        while not response_end:
            next_resp: FlyMyAIResponse = self.response_iterator.__next__()
            if not next_resp.is_event:
                response_end = next_resp
                return response_end
            else:
                evt = PredictionEvent.from_response(next_resp)
                if evt.event_type == EventType.STREAM_ID:
                    self.prediction_id = evt.prediction_id
                if self.event_callback:
                    self.event_callback(evt)
                if self.follow_cancelling and evt.event_type == EventType.CANCELLING:
                    raise StopIteration

    def __next__(self):
        response_end = None
        try:
            response_end = self.loop_iter()
            return PredictionPartial.from_response(response_end)
        except BaseFlyMyAIException as e:
            response_end = e.response
            raise e
        except Exception as e:
            raise e
        finally:
            if not response_end:
                raise StopIteration()
            stream_details_marshalled = response_end.json().get("stream_details")
            if stream_details_marshalled:
                self.stream_details = StreamDetails.model_validate(
                    stream_details_marshalled
                )
