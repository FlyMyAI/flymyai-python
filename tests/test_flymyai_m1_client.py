import os
import pathlib

import pytest
import respx
import httpx
from flymyai import m1_client, async_m1_client
from flymyai.core.types.m1 import M1Role
from .FixtureFactory import FixtureFactory
from httpx import Response

factory = FixtureFactory(__file__)


@pytest.fixture
def m1_env_fixture():
    os.environ["FLYMYAI_M1_DSN"] = factory("m1_env_fixture")


@pytest.fixture
def test_prompt():
    return "Hello, generate something interesting!"


@pytest.fixture
def dummy_image_path():
    return pathlib.Path(__file__).parent / "fixtures" / "Untitled.png"


@pytest.fixture
def apikey_fixture():
    auth_data = factory("client_auth_fixture")
    return auth_data.get("apikey", "dummy-test-apikey")


@respx.mock
def test_generate_text_only(m1_env_fixture, test_prompt, apikey_fixture):
    base_url = os.getenv("FLYMYAI_M1_DSN")

    respx.post(f"{base_url}chat").mock(
        return_value=Response(200, json={"request_id": "abc123"})
    )
    respx.get(f"{base_url}chat-result/abc123").mock(
        return_value=Response(
            200, json={"success": True, "data": {"text": "This is a response"}}
        )
    )

    client = m1_client(apikey_fixture)
    response = client.generate(prompt=test_prompt)

    assert response.data.text == "This is a response"
    assert response.success
    assert client._m1_history._records[0].role == M1Role.user
    assert client._m1_history._records[1].role == M1Role.assistant


@respx.mock
def test_generate_with_image(
    m1_env_fixture, test_prompt, dummy_image_path, apikey_fixture
):
    base_url = os.getenv("FLYMYAI_M1_DSN")

    respx.post(f"{base_url}upload-image").mock(
        return_value=Response(200, json={"url": "/static/images/xyz.png"})
    )
    respx.post(f"{base_url}chat").mock(
        return_value=Response(200, json={"request_id": "img123"})
    )
    respx.get(f"{base_url}chat-result/img123").mock(
        return_value=Response(
            200, json={"success": True, "data": {"text": "Image-based response"}}
        )
    )

    client = m1_client(apikey_fixture)
    response = client.generate(prompt=test_prompt, image=dummy_image_path)

    assert response.data.text == "Image-based response"


@respx.mock
def test_image_upload(m1_env_fixture, dummy_image_path, apikey_fixture):
    base_url = os.getenv("FLYMYAI_M1_DSN")

    respx.post(f"{base_url}upload-image").mock(
        return_value=Response(200, json={"url": "/uploads/fake123.png"})
    )

    client = m1_client(apikey_fixture)
    image_url = client.upload_image(dummy_image_path)

    assert image_url.endswith("/uploads/fake123.png")


@pytest.mark.asyncio
@respx.mock
async def test_async_generate_text_only(m1_env_fixture, test_prompt, apikey_fixture):
    base_url = os.getenv("FLYMYAI_M1_DSN")

    respx.post(f"{base_url}chat").mock(
        return_value=Response(200, json={"request_id": "async123"})
    )
    respx.get(f"{base_url}chat-result/async123").mock(
        return_value=Response(
            200, json={"success": True, "data": {"text": "Async response"}}
        )
    )

    client = async_m1_client(apikey=apikey_fixture)
    async with client:
        response = await client.generate(prompt=test_prompt)

    assert response.data.text == "Async response"
    assert response.success
    assert client._m1_history._records[0].role == M1Role.user
    assert client._m1_history._records[1].role == M1Role.assistant


@pytest.mark.asyncio
@respx.mock
async def test_async_generate_with_image(
    m1_env_fixture, test_prompt, dummy_image_path, apikey_fixture
):
    base_url = os.getenv("FLYMYAI_M1_DSN")

    respx.post(f"{base_url}upload-image").mock(
        return_value=Response(200, json={"url": "/static/images/xyz.png"})
    )
    respx.post(f"{base_url}chat").mock(
        return_value=Response(200, json={"request_id": "img123"})
    )
    respx.get(f"{base_url}chat-result/img123").mock(
        return_value=Response(
            200, json={"success": True, "data": {"text": "Image-based async response"}}
        )
    )

    client = async_m1_client(apikey=apikey_fixture)
    async with client:
        response = await client.generate(prompt=test_prompt, image=dummy_image_path)

    assert response.data.text == "Image-based async response"


@pytest.mark.asyncio
@respx.mock
async def test_async_image_upload(m1_env_fixture, dummy_image_path, apikey_fixture):
    base_url = os.getenv("FLYMYAI_M1_DSN")

    respx.post(f"{base_url}upload-image").mock(
        return_value=Response(200, json={"url": "/uploads/fake123.png"})
    )

    client = async_m1_client(apikey=apikey_fixture)
    async with client:
        image_url = await client.upload_image(dummy_image_path)

    assert image_url.endswith("/uploads/fake123.png")
