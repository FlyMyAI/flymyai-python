import dataclasses
import logging
from typing import Generator, AsyncGenerator, Union, Any

import pytest

from flymyai import FlyMyAIExceptionGroup
from flymyai.core.clients.AsyncClient import BaseAsyncClient
from flymyai.core.clients.SyncClient import BaseSyncClient


class MockedStream:
    _gen: Union[Generator[bytes, Any, None], AsyncGenerator[bytes, Any]]
    _http1_status: int
    headers: dict

    class StreamWrapper:
        def __init__(
            self,
            gen: Union[Generator[bytes, Any, None], AsyncGenerator[bytes, Any]],
        ):
            self.gen = gen

        def __next__(self):
            data = next(self.gen)
            return data.decode()

        async def __anext__(self):
            self.gen: AsyncGenerator[bytes, None]
            data = await self.gen.__anext__()
            return data.decode()

        def __iter__(self):
            return self

        def __aiter__(self):
            return self

    @dataclasses.dataclass
    class Request:
        url: str

    def __init__(
        self,
        data_generator: Union[Generator[bytes, Any, None], AsyncGenerator[bytes, Any]],
        status_code: int,
    ):
        self._gen = data_generator
        self.http1_status = status_code
        self.headers = {}

    def iter_lines(self):
        return self.StreamWrapper(self._gen)

    def aiter_lines(self):
        return self.StreamWrapper(self._gen)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.critical("MockedStream __exit__")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logging.critical("MockedStream __aexit__")

    @property
    def request(self) -> Request:
        return self.Request("AnyStr")

    @property
    def status_code(self) -> int:
        return self.http1_status


@pytest.fixture
def mock_SyncStreamIteratorWith_200_Details():
    client = BaseSyncClient("123", "123/123", max_retries=3)

    def mocked_stream(*args, **kwargs):
        yield b'data: {"details": "Correct", "status": 200}'
        yield b""

    client._client.stream = lambda *_, **__: MockedStream(mocked_stream(), 200)
    return client


@pytest.fixture
def mock_SyncStreamIteratorWith_5000():
    client = BaseSyncClient("123", "123/123", max_retries=3)

    def mocked_stream(*args, **kwargs):
        yield b'data: {"details": "Unexpected broker error! Contact support!", "status": 5000}'
        yield b""

    client._client.stream = lambda *_, **__: MockedStream(mocked_stream(), 200)
    return client


@pytest.fixture
def mock_SyncStreamIteratorWith_5320():
    client = BaseSyncClient("123", "123/123", max_retries=3)

    def mocked_stream(*args, **kwargs):
        yield b'data: {"details": "Broker is down. Try again later!", "status": 5320}'
        yield b""

    client._client.stream = lambda *_, **__: MockedStream(mocked_stream(), 200)
    return client


@pytest.fixture
def mock_AsyncStreamIteratorWith_200_Details():
    client = BaseAsyncClient("123", "123/123", max_retries=3)

    async def mocked_stream(*args, **kwargs):
        yield b'data: {"details": "Correct", "status": 200}'
        yield b""

    client._client.stream = lambda *_, **__: MockedStream(mocked_stream(), 200)
    return client


@pytest.fixture
def mock_AsyncStreamIteratorWith_5000():
    client = BaseAsyncClient("123", "123/123", max_retries=3)

    async def mocked_stream(*args, **kwargs):
        yield b'data: {"details": "Unexpected broker error! Contact support!", "status": 5000}'
        yield b""

    client._client.stream = lambda *_, **__: MockedStream(mocked_stream(), 200)
    return client


@pytest.fixture
def mock_AsyncStreamIteratorWith_5320():
    client = BaseAsyncClient("123", "123/123", max_retries=3)

    async def mocked_stream(*args, **kwargs):
        yield b'data: {"details": "Broker is down. Try again later!", "status": 5320}'
        yield b""

    client._client.stream = lambda *_, **__: MockedStream(mocked_stream(), 200)
    return client


def test_sync_unknown_error_handle_predict(mock_SyncStreamIteratorWith_200_Details):
    client = mock_SyncStreamIteratorWith_200_Details
    exc = None
    with pytest.raises(FlyMyAIExceptionGroup):
        try:
            result = client.predict({})
        except FlyMyAIExceptionGroup as e:
            exc = e
            raise e
        raise Exception(f"Should not reach this code: {result}")
    assert len(exc.errors) == 1
    assert exc.errors[0].response.status_code == 599
    assert "Timestamp" in str(exc)


def test_sync_broker_unknown_predict(mock_SyncStreamIteratorWith_5000):
    client = mock_SyncStreamIteratorWith_5000
    exc = None
    with pytest.raises(FlyMyAIExceptionGroup):
        try:
            result = client.predict({})
        except FlyMyAIExceptionGroup as e:
            exc = e
            raise e
        raise Exception(f"Should not reach this code: {result}")
    assert len(exc.errors) == 1
    assert exc.errors[0].response.status_code == 5000
    assert "Timestamp" in str(exc)


def test_sync_broker_disconnected_error_handle_predict(
    mock_SyncStreamIteratorWith_5320,
):
    client = mock_SyncStreamIteratorWith_5320
    exc = None
    with pytest.raises(FlyMyAIExceptionGroup):
        try:
            result = client.predict({})
        except FlyMyAIExceptionGroup as e:
            exc = e
            raise e
        raise Exception(f"Should not reach this code: {result}")
    assert len(exc.errors) == client.max_retries
    assert exc.errors[0].response.status_code == 5320
    assert "Timestamp" in str(exc)


@pytest.mark.asyncio
async def test_async_unknown_error_handle_predict(
    mock_AsyncStreamIteratorWith_200_Details,
):
    client = mock_AsyncStreamIteratorWith_200_Details
    exc = None
    with pytest.raises(FlyMyAIExceptionGroup):
        try:
            result = await client.predict({})
        except FlyMyAIExceptionGroup as e:
            exc = e
            raise e
        raise Exception(f"Should not reach this code: {result}")
    assert len(exc.errors) == 1
    assert exc.errors[0].response.status_code == 599
    assert "Timestamp" in str(exc)


@pytest.mark.asyncio
async def test_async_broker_unknown_predict(mock_AsyncStreamIteratorWith_5000):
    client = mock_AsyncStreamIteratorWith_5000
    exc = None
    with pytest.raises(FlyMyAIExceptionGroup):
        try:
            result = await client.predict({})
        except FlyMyAIExceptionGroup as e:
            exc = e
            raise e
        raise Exception(f"Should not reach this code: {result}")
    assert len(exc.errors) == 1
    assert exc.errors[0].response.status_code == 5000
    assert "Timestamp" in str(exc)


@pytest.mark.asyncio
async def test_async_broker_disconnected_error_handle_predict(
    mock_AsyncStreamIteratorWith_5320,
):
    client = mock_AsyncStreamIteratorWith_5320
    exc = None
    with pytest.raises(FlyMyAIExceptionGroup):
        try:
            result = await client.predict({})
        except FlyMyAIExceptionGroup as e:
            exc = e
            raise e
        raise Exception(f"Should not reach this code: {result}")
    assert len(exc.errors) == client.max_retries
    assert exc.errors[0].response.status_code == 5320
    assert "Timestamp" in str(exc)
