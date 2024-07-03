import os

import pytest

from flymyai import client as sync_client, async_client

from tests.FixtureFactory import FixtureFactory

factory = FixtureFactory(__file__)


@pytest.fixture
def dsn():
    os.environ["FLYMYAI_DSN"] = factory("address_fixture")


@pytest.fixture
def vllm_stream_payload():
    return factory("vllm_stream_payload")


@pytest.fixture
def vllm_stream_auth():
    return factory("vllm_auth")


def test_vllm_stream(vllm_stream_auth, vllm_stream_payload, dsn):
    stream_iterator = sync_client(auth=vllm_stream_auth).stream(vllm_stream_payload)
    for response in stream_iterator:
        assert response.status == 200
        assert response.output_data
        print(response.output_data["o_text_output"].pop(), end="")
    print("\n")


@pytest.mark.asyncio
async def test_vllm_async_stream(vllm_stream_auth, vllm_stream_payload, dsn):
    try:
        stream_iterator = async_client(auth=vllm_stream_auth).stream(
            vllm_stream_payload
        )
        async for response in stream_iterator:
            assert response.status == 200
            assert response.output_data
            print(response.output_data["o_text_output"].pop(), end="")
    except Exception as e:
        if hasattr(e, "msg"):
            print(e)
        raise e
    finally:
        print()
