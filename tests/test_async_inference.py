import os
import pathlib

import pytest

from flymyai.core.exceptions import (
    RetryTimeoutExceededException,
    BaseFlyMyAIException,
    FlyMyAIExceptionGroup,
)
from flymyai.core.models.successful_responses import (
    AsyncPredictionResponseList,
    AsyncPredictionTask,
)
from .FixtureFactory import FixtureFactory
from flymyai import client as sync_client, async_client

factory = FixtureFactory(__file__)


@pytest.fixture
def address_fixture():
    os.environ["FLYMYAI_DSN"] = factory("address_fixture")


@pytest.fixture
def binary_file_paths():
    return factory("binary_file_paths")


@pytest.fixture
def fake_payload_fixture(binary_file_paths) -> dict:
    files = {}
    for k, v in binary_file_paths.items():
        files[k] = pathlib.Path(v)
    payload = factory("fake_payload_fixture")
    payload.update(files)
    return payload


@pytest.fixture
def broken_payload_fixture(binary_file_paths) -> dict:
    files = {}
    for k, v in binary_file_paths.items():
        files[k] = pathlib.Path(v)
    payload = factory("broken_payload_fixture")
    payload.update(files)
    return payload


@pytest.fixture
def client_auth_fixture() -> dict:
    client_auth_fixture: dict = factory("client_auth_fixture")
    auth_apikey_env = client_auth_fixture.pop("apikey_environ", "")
    if auth_apikey_env:
        client_auth_fixture["apikey"] = os.getenv(
            auth_apikey_env, client_auth_fixture.get("apikey")
        )
    return client_auth_fixture


def test_sync_client_async_inference(
    address_fixture, fake_payload_fixture, client_auth_fixture
):
    client = sync_client(**client_auth_fixture)
    prediction_task = client.predict_async_task(payload=fake_payload_fixture)
    assert prediction_task.prediction_id is not None
    with pytest.raises(RetryTimeoutExceededException):
        res = prediction_task.result(0)
        assert res is None  # should not achieve this point
    res = prediction_task.result()
    assert isinstance(res, AsyncPredictionResponseList)
    assert all(map(lambda x: x.infer_details["status"] == 200, res.inference_responses))

    mocked_pred_task = AsyncPredictionTask(prediction_id="123")
    mocked_pred_task.set_client(client)
    with pytest.raises(FlyMyAIExceptionGroup):
        mocked_pred_task.result()


@pytest.mark.asyncio
async def test_async_client_async_inference(
    address_fixture, fake_payload_fixture, client_auth_fixture
):
    client = async_client(**client_auth_fixture)
    prediction_task = await client.predict_async_task(payload=fake_payload_fixture)
    assert prediction_task.prediction_id is not None

    with pytest.raises(RetryTimeoutExceededException):
        res = await prediction_task.result(0)
        assert res is None  # should not achieve this point
    res = await prediction_task.result()
    assert isinstance(res, AsyncPredictionResponseList), res
    assert all(map(lambda x: x.infer_details["status"] == 200, res.inference_responses))

    mocked_pred_task = AsyncPredictionTask(prediction_id="123")
    mocked_pred_task.set_client(client)
    with pytest.raises(FlyMyAIExceptionGroup):
        await mocked_pred_task.result()


def test_sync_client_async_inference_with_guaranteed_error(
    address_fixture, broken_payload_fixture, client_auth_fixture
):
    client = sync_client(**client_auth_fixture)
    prediction_task = client.predict_async_task(payload=broken_payload_fixture)
    assert prediction_task.prediction_id is not None
    with pytest.raises(FlyMyAIExceptionGroup):
        res = prediction_task.result()
        assert res is None  # should not achieve this point


@pytest.mark.asyncio
async def test_async_client_async_inference_with_guaranteed_error(
    address_fixture, broken_payload_fixture, client_auth_fixture
):
    client = async_client(**client_auth_fixture)
    prediction_task = await client.predict_async_task(payload=broken_payload_fixture)
    assert prediction_task.prediction_id is not None
    with pytest.raises(FlyMyAIExceptionGroup):
        res = await prediction_task.result()
        assert res is None  # should not achieve this point
