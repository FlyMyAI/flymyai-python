import os

import pytest
from flymyai import run as flymyai_sync_run
from flymyai import async_run as flymyai_async_run
from flymyai import client as flymyai_client


@pytest.fixture
def address_fixture():
    os.environ["FLYMYAI_DSN"] = "http://localhost:8088"


@pytest.fixture
def fake_payload_fixture():
    image_path = "/home/oleg/Downloads/Telegram Desktop/image_2023-12-07_23-42-17.png"
    payload = {"i_image": image_path}
    return payload


@pytest.fixture
def client_auth_fixture():
    return {
        "apikey": "fly-12e2wqfusodigih",
        "username": "d1",
        "project_name": "test2",
    }


def test_flymyai_client(address_fixture, fake_payload_fixture, client_auth_fixture):
    response = flymyai_sync_run(
        client_info=client_auth_fixture,
        payload=fake_payload_fixture,
    )
    assert response


def test_flymyai_openapi(address_fixture, client_auth_fixture):
    response = flymyai_client(client_info=client_auth_fixture).openapi_schema()
    assert response


@pytest.mark.asyncio
async def test_flymyai_async_run(
    address_fixture, client_auth_fixture, fake_payload_fixture
):
    response = await flymyai_async_run(
        client_info=client_auth_fixture, payload=fake_payload_fixture
    )
    assert response
