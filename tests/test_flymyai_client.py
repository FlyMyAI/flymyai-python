import os
import pathlib

import pytest
from flymyai import run as flymyai_sync_run
from flymyai import async_run as flymyai_async_run
from flymyai import client as flymyai_client
from .FixtureFactory import FixtureFactory

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
def client_auth_fixture() -> dict:
    return factory("client_auth_fixture")


def test_flymyai_client(address_fixture, fake_payload_fixture, client_auth_fixture):
    response = flymyai_sync_run(
        auth=client_auth_fixture,
        payload=fake_payload_fixture,
    )
    assert response


def test_flymyai_openapi(address_fixture, client_auth_fixture):
    response = flymyai_client(auth=client_auth_fixture).openapi_schema()
    assert response


@pytest.mark.asyncio
async def test_flymyai_async_run(
    address_fixture, client_auth_fixture, fake_payload_fixture
):
    response = await flymyai_async_run(
        auth=client_auth_fixture, payload=fake_payload_fixture
    )
    assert response
