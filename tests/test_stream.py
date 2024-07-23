import os

import pytest

from flymyai import client as sync_client, async_client

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
    return factory("auth")


@pytest.fixture
def output_field():
    return factory("output_field")


def test_stream(stream_auth, stream_payload, dsn, output_field):
    stream_iterator = sync_client(**stream_auth).stream(stream_payload)
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
        print(stream_iterator.stream_details)


@pytest.mark.asyncio
async def test_async_stream(stream_auth, stream_payload, dsn, output_field):
    stream_iterator = async_client(**stream_auth).stream(stream_payload)
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
        print(stream_iterator.stream_details)
