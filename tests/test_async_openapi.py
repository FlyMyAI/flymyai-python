"""The public async schema method must reach the selected model over HTTP."""

import httpx
import pytest

from flymyai import async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, "other/selected"])
async def test_async_openapi_uses_default_or_overridden_model(override):
    calls = []
    schema = {"openapi": "3.0.0", "paths": {"/predict": {}}}

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=schema)

    async with async_client("fixture-key", model="owner/default", max_retries=1) as sdk:
        await sdk._client.aclose()
        sdk._client = httpx.AsyncClient(
            base_url="https://gateway.test", transport=httpx.MockTransport(handle)
        )
        result = await sdk.openapi_schema(model=override)

    assert result.openapi_schema == schema
    assert len(calls) == 1
    assert calls[0].url.path == "/api/v1/{}/openapi.json".format(
        override or "owner/default"
    )
    assert calls[0].headers["X-API-KEY"] == "fixture-key"
