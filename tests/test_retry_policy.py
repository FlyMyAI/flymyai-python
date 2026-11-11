"""Retries as the official OpenAI and Anthropic SDKs do them (flymyai.core._retry).

No network: every client talks to an httpx.MockTransport and every wait is recorded
instead of slept.
"""

import asyncio
import email.utils
import json
import time

import httpx
import pytest

from flymyai import (
    FlyMyAIExceptionGroup,
    FlyMyAIPredictException,
    FlyMyAIStreamError,
    FlyMyAIStreamInterrupted,
)
from flymyai.core import _retry
from flymyai.core.clients.AsyncClient import BaseAsyncClient
from flymyai.core.clients.SyncClient import BaseSyncClient
from flymyai.core.clients import base_client
from flymyai.core.exceptions import FlyMyAIAsyncTaskException

ID_EVENT = b'event: {"prediction_id": "pred-1", "event_type": "id"}\n\n'
CHUNK_1 = b'data: {"status": 200, "output_data": {"output": ["Hel"]}}\n\n'
CHUNK_2 = (
    b'data: {"status": 200, "output_data": {"output": ["lo"]}, "stream_details":'
    b' {"input_tokens": 3, "output_tokens": 2}}\n\n'
)
ERROR_EVENT = b'data: {"status": 500, "details": "provider failed"}\n\n'
PREDICT_OK = b'data: {"status": 200, "output_data": {"output": ["done"]}}\n\n'
ASYNC_OK = b'{"prediction_id": "pred-async-1"}'


class _Chunks(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body that yields its chunks, then breaks like a dropped connection."""

    def __init__(self, chunks, error=None):
        self.chunks = list(chunks)
        self.error = error

    def __iter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


def _sse(chunks, error=None, status=200, headers=None):
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream", **(headers or {})},
        stream=_Chunks(chunks, error),
    )


class _Server:
    """Answers each request with the next planned response and keeps the requests."""

    def __init__(self, *plan):
        self.plan = list(plan)
        self.requests = []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        step = self.plan.pop(0) if len(self.plan) > 1 else self.plan[0]
        if isinstance(step, Exception):
            raise step
        return step() if callable(step) else step


def _form(request: httpx.Request) -> bytes:
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    return request.read().replace(boundary, b"BOUNDARY")


def _sync_client(server, max_retries=2):
    class Client(BaseSyncClient):
        def _construct_client(self):
            return httpx.Client(
                transport=httpx.MockTransport(server),
                base_url="https://api.test/",
                headers=self.client_info.authorization_headers,
            )

    return Client("fly-test", "owner/project", max_retries=max_retries)


def _async_client(server, max_retries=2):
    class Client(BaseAsyncClient):
        def _construct_client(self):
            return httpx.AsyncClient(
                transport=httpx.MockTransport(server),
                base_url="https://api.test/",
                headers=self.client_info.authorization_headers,
            )

    return Client("fly-test", "owner/project", max_retries=max_retries)


@pytest.fixture
def waits(monkeypatch):
    """Every backoff the clients take, in seconds; nothing really sleeps."""
    recorded = []

    def fake_sleep(seconds):
        recorded.append(seconds)

    async def fake_async_sleep(seconds, *args, **kwargs):
        recorded.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(asyncio, "sleep", fake_async_sleep)
    return recorded


# --- the policy itself -------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [(408, True), (409, True), (429, True), (500, True), (503, True), (599, True)]
    + [(400, False), (401, False), (404, False), (422, False)],
)
def test_status_policy_like_openai(status, expected):
    assert _retry.status_should_retry(status) is expected


def test_should_retry_header_wins_over_status():
    assert _retry.status_should_retry(400, {"x-should-retry": "true"}) is True
    assert _retry.status_should_retry(503, {"x-should-retry": "false"}) is False
    assert _retry.status_should_retry(503, {"x-should-retry": "maybe"}) is True


def test_gateway_bodies_that_forbid_another_attempt():
    reused = json.dumps({"code": "idempotency_key_reused", "detail": "x"})
    unknown = json.dumps({"code": "tool_outcome_unknown", "retry_safe": False})
    in_flight = json.dumps({"code": "idempotency_key_in_flight", "detail": "x"})
    assert _retry.status_should_retry(409, None, reused) is False
    assert _retry.status_should_retry(409, None, unknown) is False
    assert _retry.status_should_retry(409, None, in_flight) is True


def test_retry_after_headers():
    assert _retry.parse_retry_after({"retry-after-ms": "1500"}) == 1.5
    assert _retry.parse_retry_after({"retry-after": "3"}) == 3.0
    date = email.utils.formatdate(time.time() + 30, usegmt=True)
    assert 25 < _retry.parse_retry_after({"retry-after": date}) <= 31
    assert _retry.parse_retry_after({}) is None


def test_backoff_bounds():
    for attempt, ceiling in [(0, 0.5), (1, 1.0), (2, 2.0), (3, 4.0), (9, 8.0)]:
        delay = _retry.retry_delay(attempt)
        assert ceiling * 0.75 <= delay <= ceiling
    assert _retry.retry_delay(0, {"retry-after": "7"}) == 7.0
    # Longer than two minutes: back to the exponential backoff.
    assert _retry.retry_delay(0, {"retry-after": "600"}) <= 0.5


def test_max_retries_env_is_a_number():
    assert isinstance(base_client.DEFAULT_RETRY_COUNT, int)


# --- streams (sync) -------------------------------------------------------------


def _drain(stream):
    return [partial.output_data["output"][0] for partial in stream]


def test_stream_cut_before_first_event_is_sent_again(waits):
    server = _Server(
        lambda: _sse([], httpx.ReadError("cut")),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    stream = _sync_client(server).stream({"prompt": "hi"})
    assert _drain(stream) == ["Hel", "lo"]
    assert len(server.requests) == 2
    assert len(waits) == 1 and 0.375 <= waits[0] <= 0.5
    assert stream.stream_details.output_tokens == 2


def test_stream_cut_after_first_event_is_never_sent_again(waits):
    server = _Server(lambda: _sse([ID_EVENT, CHUNK_1], httpx.ReadError("cut")))
    stream = _sync_client(server).stream({"prompt": "hi"})
    received = []
    with pytest.raises(FlyMyAIStreamInterrupted) as caught:
        for partial in stream:
            received.append(partial.output_data["output"][0])
    assert received == ["Hel"]
    assert len(server.requests) == 1
    assert caught.value.prediction_id == "pred-1"
    assert caught.value.events_received == 2
    assert isinstance(caught.value.__cause__, httpx.ReadError)
    assert isinstance(caught.value, FlyMyAIPredictException)


def test_stream_503_with_should_retry_true_is_sent_again(waits):
    server = _Server(
        lambda: _sse(
            [b'{"detail": "busy"}'], status=503, headers={"x-should-retry": "true"}
        ),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    assert _drain(_sync_client(server).stream({"prompt": "hi"})) == ["Hel", "lo"]
    assert len(server.requests) == 2


def test_stream_500_with_should_retry_false_is_not(waits):
    server = _Server(
        lambda: _sse(
            [b'{"detail": "no"}'], status=500, headers={"x-should-retry": "false"}
        ),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    with pytest.raises(FlyMyAIPredictException) as caught:
        _drain(_sync_client(server).stream({"prompt": "hi"}))
    assert caught.value.response.status_code == 500
    assert len(server.requests) == 1
    assert waits == []


def test_stream_retry_after_is_honoured(waits):
    server = _Server(
        lambda: _sse([b"{}"], status=429, headers={"retry-after": "3"}),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    assert _drain(_sync_client(server).stream({"prompt": "hi"})) == ["Hel", "lo"]
    assert waits == [3.0]


def test_stream_error_event_raises_a_typed_error(waits):
    server = _Server(lambda: _sse([ID_EVENT, CHUNK_1, ERROR_EVENT]))
    stream = _sync_client(server).stream({"prompt": "hi"})
    with pytest.raises(FlyMyAIStreamError) as caught:
        _drain(stream)
    assert caught.value.prediction_id == "pred-1"
    assert caught.value.events_received == 2
    assert caught.value.response.status_code == 500
    assert len(server.requests) == 1


def test_stream_that_never_starts_raises_the_connection_error(waits):
    server = _Server(lambda: _sse([], httpx.ReadError("cut")))
    with pytest.raises(httpx.ReadError):
        _drain(_sync_client(server).stream({"prompt": "hi"}))
    # One attempt and FMA_RECONNECT_RETRIES (3) more, none of them started.
    assert len(server.requests) == 1 + base_client._RECONNECT_RETRIES
    assert len(waits) == base_client._RECONNECT_RETRIES


# --- predict and async submit (sync) -------------------------------------------


def test_predict_503_should_retry_true_then_ok(waits):
    server = _Server(
        lambda: _sse(
            [b'{"detail": "busy"}'], status=503, headers={"x-should-retry": "true"}
        ),
        lambda: _sse([PREDICT_OK]),
    )
    result = _sync_client(server).predict({"prompt": "hi"})
    assert result.output_data["output"] == ["done"]
    assert len(server.requests) == 2


def test_predict_500_should_retry_false_is_not_sent_again(waits):
    server = _Server(
        lambda: _sse(
            [b'{"detail": "no"}'], status=500, headers={"x-should-retry": "false"}
        ),
        lambda: _sse([PREDICT_OK]),
    )
    with pytest.raises(FlyMyAIExceptionGroup) as caught:
        _sync_client(server).predict({"prompt": "hi"})
    assert len(caught.value.errors) == 1
    assert len(server.requests) == 1


def test_predict_cut_before_its_event_is_sent_again(waits):
    server = _Server(
        lambda: _sse([], httpx.RemoteProtocolError("peer closed")),
        lambda: _sse([PREDICT_OK]),
    )
    result = _sync_client(server).predict({"prompt": "hi"})
    assert result.output_data["output"] == ["done"]
    assert len(server.requests) == 2


def test_async_submit_keeps_one_idempotency_key(waits):
    server = _Server(
        httpx.Response(503, json={"detail": "busy"}),
        httpx.Response(202, content=ASYNC_OK),
    )
    task = _sync_client(server).predict_async_task(
        {"prompt": "hi", "file": b"\x00\x01"}
    )
    assert task.prediction_id == "pred-async-1"
    keys = [r.headers.get("Idempotency-Key") for r in server.requests]
    assert len(keys) == 2 and keys[0] and keys[0] == keys[1]
    # The same multipart form both times (file names included): the gateway binds
    # the key to it.
    assert _form(server.requests[0]) == _form(server.requests[1])


def test_async_submit_uses_the_callers_key(waits):
    server = _Server(httpx.Response(202, content=ASYNC_OK))
    _sync_client(server).predict_async_task({"prompt": "hi"}, idempotency_key="k-1")
    assert server.requests[0].headers["Idempotency-Key"] == "k-1"


def test_async_submit_reused_key_is_not_sent_again(waits):
    body = {"code": "idempotency_key_reused", "detail": "used"}
    server = _Server(
        httpx.Response(409, json=body), httpx.Response(202, content=ASYNC_OK)
    )
    with pytest.raises(FlyMyAIAsyncTaskException) as caught:
        _sync_client(server).predict_async_task({"prompt": "hi"})
    assert caught.value.response.status_code == 409
    assert len(server.requests) == 1


# --- the asyncio client -----------------------------------------------------------


async def _adrain(stream):
    return [partial.output_data["output"][0] async for partial in stream]


@pytest.mark.asyncio
async def test_async_stream_cut_before_first_event_is_sent_again(waits):
    server = _Server(
        lambda: _sse([], httpx.ReadError("cut")),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    assert await _adrain(_async_client(server).stream({"prompt": "hi"})) == [
        "Hel",
        "lo",
    ]
    assert len(server.requests) == 2


@pytest.mark.asyncio
async def test_async_stream_cut_after_first_event_is_never_sent_again(waits):
    server = _Server(lambda: _sse([ID_EVENT, CHUNK_1], httpx.ReadError("cut")))
    stream = _async_client(server).stream({"prompt": "hi"})
    received = []
    with pytest.raises(FlyMyAIStreamInterrupted) as caught:
        async for partial in stream:
            received.append(partial.output_data["output"][0])
    assert received == ["Hel"]
    assert len(server.requests) == 1
    assert caught.value.prediction_id == "pred-1"


@pytest.mark.asyncio
async def test_async_stream_should_retry_header(waits):
    server = _Server(
        lambda: _sse([b"{}"], status=503, headers={"x-should-retry": "true"}),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    assert await _adrain(_async_client(server).stream({"prompt": "hi"})) == [
        "Hel",
        "lo",
    ]
    server = _Server(
        lambda: _sse([b"{}"], status=500, headers={"x-should-retry": "false"}),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    with pytest.raises(FlyMyAIPredictException):
        await _adrain(_async_client(server).stream({"prompt": "hi"}))
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_async_stream_error_event_and_retry_after(waits):
    server = _Server(lambda: _sse([ID_EVENT, CHUNK_1, ERROR_EVENT]))
    with pytest.raises(FlyMyAIStreamError):
        await _adrain(_async_client(server).stream({"prompt": "hi"}))
    server = _Server(
        lambda: _sse([b"{}"], status=429, headers={"retry-after-ms": "2500"}),
        lambda: _sse([ID_EVENT, CHUNK_1, CHUNK_2]),
    )
    assert await _adrain(_async_client(server).stream({"prompt": "hi"})) == [
        "Hel",
        "lo",
    ]
    assert 2.5 in waits


@pytest.mark.asyncio
async def test_async_predict_and_submit(waits):
    server = _Server(
        lambda: _sse([b"{}"], status=503, headers={"x-should-retry": "true"}),
        lambda: _sse([PREDICT_OK]),
    )
    result = await _async_client(server).predict({"prompt": "hi"})
    assert result.output_data["output"] == ["done"]
    server = _Server(
        httpx.Response(429, json={"detail": "slow down"}, headers={"retry-after": "1"}),
        httpx.Response(202, content=ASYNC_OK),
    )
    task = await _async_client(server).predict_async_task({"prompt": "hi"})
    assert task.prediction_id == "pred-async-1"
    keys = {r.headers.get("Idempotency-Key") for r in server.requests}
    assert len(keys) == 1 and None not in keys
