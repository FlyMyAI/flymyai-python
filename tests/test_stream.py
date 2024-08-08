import asyncio
import os
import threading

import pytest

from flymyai import client as sync_client, async_client
from flymyai.core.models.successful_responses import PredictionEvent
from flymyai.core.types.event_types import EventType

from tests.FixtureFactory import FixtureFactory

factory = FixtureFactory(__file__)


@pytest.fixture
def dsn():
    os.environ["FLYMYAI_DSN"] = factory("address_fixture")


@pytest.fixture
def stream_payload():
    return factory("stream_payload")


@pytest.fixture
def stream_auth():
    client_auth_fixture = factory("auth")
    auth_apikey_env = client_auth_fixture.pop("apikey_environ", "")
    if auth_apikey_env:
        client_auth_fixture["apikey"] = os.getenv(
            auth_apikey_env, client_auth_fixture.get("apikey")
        )
    return client_auth_fixture


@pytest.fixture
def output_field():
    return factory("output_field")


def test_stream(stream_auth, stream_payload, dsn, output_field):
    stream_iterator = sync_client(**stream_auth).stream(stream_payload)
    stream_iterator.follow_cancelling = True
    stream_iterator.set_on_event(print)
    try:
        for response in stream_iterator:
            assert response.status == 200
            assert response.output_data or hasattr(stream_iterator, "stream_details")
            if response.output_data.get(output_field):
                print(response.output_data[output_field].pop(), end="")
    except Exception as e:
        if hasattr(e, "msg"):
            print(e)
        raise e
    finally:
        print()
        print(getattr(stream_iterator, "stream_details", None))


@pytest.mark.asyncio
async def test_async_stream(stream_auth, stream_payload, dsn, output_field):
    stream_iterator = async_client(**stream_auth).stream(stream_payload)
    stream_iterator.follow_cancelling = True
    stream_iterator.set_on_event(print)
    try:
        async for response in stream_iterator:
            assert response.status == 200
            assert response.output_data or hasattr(stream_iterator, "stream_details")
            if response.output_data.get(output_field):
                print(response.output_data[output_field].pop(), end="")
    except Exception as e:
        if hasattr(e, "msg"):
            print(e)
        raise e
    finally:
        print()
        print(getattr(stream_iterator, "stream_details", None))


def test_stream_cancel(stream_auth, stream_payload, dsn, output_field):
    stream_iterator = sync_client(**stream_auth).stream(stream_payload)
    stream_iterator.follow_cancelling = False
    cancelling_obtained = threading.Event()

    def cancel_callback(event: PredictionEvent):
        if event.event_type == EventType.STREAM_ID:
            stream_iterator.cancel()
        else:
            cancelling_obtained.set()

    stream_iterator.set_on_event(cancel_callback)
    try:
        for response in stream_iterator:
            assert response.status == 200
            assert response.output_data or hasattr(stream_iterator, "stream_details")
            if response.output_data.get(output_field):
                print(response.output_data[output_field].pop(), end="")
    except Exception as e:
        if hasattr(e, "msg"):
            print(e)
        raise e
    finally:
        print()
        assert cancelling_obtained.is_set()
        print(getattr(stream_iterator, "stream_details", None))


def test_cancel_with_client(stream_auth, stream_payload, dsn, output_field):
    client = sync_client(**stream_auth)
    stream_iterator = client.stream(stream_payload)
    stream_iterator.follow_cancelling = True
    cancelling_obtained = threading.Event()

    def cancel_callback(event: PredictionEvent):
        if event.event_type == EventType.STREAM_ID:
            client.cancel_prediction(
                stream_iterator.prediction_id, model=stream_auth["model"]
            )
        else:
            cancelling_obtained.set()

    stream_iterator.set_on_event(cancel_callback)
    try:
        for response in stream_iterator:
            assert response.status == 200
            assert response.output_data or hasattr(stream_iterator, "stream_details")
            if response.output_data.get(output_field):
                print(response.output_data[output_field].pop(), end="")
    except Exception as e:
        if hasattr(e, "msg"):
            print(e)
        raise e
    finally:
        print()
        assert cancelling_obtained.is_set()
        print(getattr(stream_iterator, "stream_details", None))


@pytest.mark.asyncio
async def test_async_stream_cancel(stream_auth, stream_payload, dsn, output_field):
    client = async_client(**stream_auth)
    stream_iterator = client.stream(stream_payload)
    stream_iterator.follow_cancelling = False
    cancelling_obtained = asyncio.Event()

    async def cancel_callback(event: PredictionEvent):
        if event.event_type == EventType.STREAM_ID:
            await stream_iterator.cancel()
        else:
            cancelling_obtained.set()

    stream_iterator.set_on_event(cancel_callback)
    try:
        async for response in stream_iterator:
            assert response.status == 200
            assert response.output_data
            print(response.output_data[output_field].pop(), end="")
    except Exception as e:
        if hasattr(e, "msg"):
            print(e)
        raise e
    finally:
        print()
        assert cancelling_obtained.is_set()
        print(getattr(stream_iterator, "stream_details", None))


@pytest.mark.asyncio
async def test_async_stream_cancel_with_client(
    stream_auth, stream_payload, dsn, output_field
):
    client = async_client(**stream_auth)
    stream_iterator = client.stream(stream_payload)
    stream_iterator.follow_cancelling = False
    cancelling_obtained = asyncio.Event()

    async def cancel_callback(event: PredictionEvent):
        if event.event_type == EventType.STREAM_ID:
            await client.cancel_prediction(
                stream_iterator.prediction_id, model=stream_auth["model"]
            )
        else:
            cancelling_obtained.set()

    stream_iterator.set_on_event(cancel_callback)
    try:
        async for response in stream_iterator:
            assert response.status == 200
            assert response.output_data or hasattr(stream_iterator, "stream_details")
            if response.output_data.get(output_field):
                print(response.output_data[output_field].pop(), end="")
    except Exception as e:
        if hasattr(e, "msg"):
            print(e)
        raise e
    finally:
        print()
        assert cancelling_obtained.is_set()
        print(getattr(stream_iterator, "stream_details", None))
