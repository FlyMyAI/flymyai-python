import asyncio
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
        **client_auth_fixture,
        payload=fake_payload_fixture,
    )
    assert response


def test_flymyai_openapi(address_fixture, client_auth_fixture):
    response1 = flymyai_client(**client_auth_fixture).openapi_schema()
    response2 = flymyai_client(client_auth_fixture["apikey"]).openapi_schema(
        model=client_auth_fixture["model"]
    )
    assert response1.model_dump() == response2.model_dump()


@pytest.mark.asyncio
async def test_flymyai_async_run(
    address_fixture, client_auth_fixture, fake_payload_fixture
):
    response = await flymyai_async_run(
        **client_auth_fixture, payload=fake_payload_fixture
    )
    assert response


@pytest.mark.asyncio
async def test_doc_case(address_fixture, client_auth_fixture, fake_payload_fixture):
    tasks = [
        asyncio.create_task(flymyai_async_run(**client_auth_fixture, payload=prompt))
        for prompt in [fake_payload_fixture] * 3
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
