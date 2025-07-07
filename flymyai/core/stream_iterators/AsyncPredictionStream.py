import asyncio
from typing import AsyncIterator, TypeVar, Callable, Union, Awaitable

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

_AsyncEventCallbackType = TypeVar(
    "_AsyncEventCallbackType",
    bound=Union[
        Callable[[PredictionEvent], None], Callable[[PredictionEvent], Awaitable[None]]
    ],
)


class AsyncPredictionStream:
    stream_details: StreamDetails

    event_callback: _AsyncEventCallbackType = None

    prediction_id: str

    follow_cancelling: bool = True

    _client: BaseClient
    _client_info: APIKeyClientInfo

    def __init__(
        self,
        response_iterator: AsyncIterator,
        client: BaseClient,
        client_info: APIKeyClientInfo,
    ):
        self.response_iterator = response_iterator
        self._client = client
        self._client_info = client_info

    async def cancel(self):
        if not hasattr(self, "prediction_id"):
            raise StreamCancellationException("No prediction_id obtained!")
        return await self._client.cancel_prediction(
            self.prediction_id, client_info=self._client_info
        )

    def __aiter__(self):
        return self

    def set_on_event(self, callback_or_coro: _AsyncEventCallbackType):
        self.event_callback = callback_or_coro

    async def loop_iter(self):
        response_end = None
        while not response_end:
            next_resp: FlyMyAIResponse = await self.response_iterator.__anext__()
            if not next_resp.is_event:
                response_end = next_resp
                return response_end
            else:
                evt = PredictionEvent.from_response(next_resp)
                if not self.event_callback:
                    pass
                else:
                    coro_or_res = self.event_callback(evt)
                    if asyncio.iscoroutine(coro_or_res):
                        asyncio.run_coroutine_threadsafe(
                            coro_or_res, asyncio.get_event_loop()
                        )
                if self.follow_cancelling and evt.event_type == EventType.CANCELLING:
                    raise StopAsyncIteration
                if evt.event_type == EventType.STREAM_ID:
                    self.prediction_id = evt.prediction_id

    async def __anext__(self):
        response_end = None
        try:
            response_end = await self.loop_iter()
            return PredictionPartial.from_response(response_end)
        except BaseFlyMyAIException as e:
            response_end = e.response
            raise e
        except Exception as e:
            raise e
        finally:
            if not response_end:
                raise StopAsyncIteration()
            stream_details_marshalled = response_end.json().get("stream_details")
            if stream_details_marshalled:
                self.stream_details = StreamDetails.model_validate(
                    stream_details_marshalled
                )
