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


def test_vllm_stream(stream_auth, stream_payload, dsn):
    stream_iterator = sync_client(**stream_auth).stream(stream_payload)
    for response in stream_iterator:
        assert response.status == 200
        assert response.output_data
        print(response.output_data["o_text_output"].pop(), end="")
    print("\n")


@pytest.mark.asyncio
async def test_async_stream(stream_auth, stream_payload, dsn, output_field):
    try:
        stream_iterator = async_client(**stream_auth).stream(stream_payload)
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
