import asyncio
from typing import AsyncIterator, TypeVar, Callable, Union, Awaitable

from flymyai.core._response import FlyMyAIResponse
from flymyai.core.exceptions import BaseFlyMyAIException
from flymyai.core.models import StreamDetails, PredictionPartial, PredictionEvent
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

    follow_cancelling: bool = True

    def __init__(self, response_iterator: AsyncIterator):
        self.response_iterator = response_iterator

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
                    if (
                        self.follow_cancelling
                        and evt.event_type == EventType.CANCELLING
                    ):
                        raise StopAsyncIteration

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
