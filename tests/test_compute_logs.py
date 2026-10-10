"""Bounded sync/async owner diagnostics, authentication errors and forward fields."""

import asyncio

import httpx
import pytest

from flymyai.agents import AsyncAgentClient, FlyMyAIAgentError, SyncAgentClient

SID = "00000000-0000-4000-8000-000000000001"
PAYLOAD = {
    "schema": "flymy.compute-logs.v1",
    "server_id": SID,
    "status": "sleeping",
    "availability": "sleeping",
    "health": None,
    "future_field": {"ok": True},
    "logs": {
        "lines": ["ValueError"],
        "bytes": 11,
        "retention": "provider_tail",
        "future_tail": 1,
    },
}


def test_sync_owner_snapshot_is_one_get_and_keeps_unknown_fields():
    calls = []

    def answer(request):
        calls.append(request)
        assert request.headers["accept-encoding"] == "identity"
        assert request.url.path == f"/api/v1/agents/compute/servers/{SID}/logs/"
        assert dict(request.url.params) == {"lines": "20", "max_bytes": "4096"}
        return httpx.Response(200, json=PAYLOAD)

    client = SyncAgentClient(api_key="test-key", base_url="https://test.invalid")
    client._http.close()
    client._http = httpx.Client(
        base_url="https://test.invalid", transport=httpx.MockTransport(answer)
    )
    with client:
        result = client.compute.logs(SID, lines=20, max_bytes=4096)
    assert len(calls) == 1 and calls[0].method == "GET"
    assert result.future_field == {"ok": True} and result.logs.future_tail == 1
    assert result.logs.lines == ["ValueError"]


def test_async_snapshot_and_error_do_not_retry():
    async def check():
        calls = []

        async def answer(request):
            calls.append(request)
            return httpx.Response(
                200 if len(calls) == 1 else 404,
                json=(
                    PAYLOAD
                    if len(calls) == 1
                    else {"code": "server_not_found", "message": "no such owner server"}
                ),
            )

        client = AsyncAgentClient(api_key="test-key", base_url="https://test.invalid")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://test.invalid", transport=httpx.MockTransport(answer)
        )
        async with client:
            assert (await client.compute.logs(SID)).availability == "sleeping"
            with pytest.raises(FlyMyAIAgentError) as error:
                await client.compute.logs(SID)
            assert error.value.status_code == 404
        assert len(calls) == 2

    asyncio.run(check())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"server_id": "../../escape"},
        {"server_id": SID, "lines": 201},
        {"server_id": SID, "lines": True},
        {"server_id": SID, "max_bytes": 16385},
    ],
)
def test_invalid_request_fails_before_dispatch(kwargs):
    with SyncAgentClient(api_key="test-key", base_url="https://test.invalid") as client:
        with pytest.raises(ValueError):
            client.compute.logs(**kwargs)


def test_oversized_response_is_closed_before_json_hydration():
    client = SyncAgentClient(api_key="test-key", base_url="https://test.invalid")
    client._http.close()
    client._http = httpx.Client(
        base_url="https://test.invalid",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"x" * 65537)
        ),
    )
    with client:
        with pytest.raises(FlyMyAIAgentError, match="64 KiB"):
            client.compute.logs(SID)


@pytest.mark.parametrize("async_client", [False, True])
def test_compressed_response_is_rejected_before_consumption(async_client):
    class NeverRead(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            raise AssertionError("compressed body must not be read")
            yield b""

        async def __aiter__(self):
            raise AssertionError("compressed body must not be read")
            yield b""

    def answer(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200, headers={"Content-Encoding": "gzip"}, stream=NeverRead()
        )

    if async_client:

        async def check():
            client = AsyncAgentClient(
                api_key="test-key", base_url="https://test.invalid"
            )
            await client._http.aclose()
            client._http = httpx.AsyncClient(
                base_url="https://test.invalid", transport=httpx.MockTransport(answer)
            )
            async with client:
                with pytest.raises(FlyMyAIAgentError, match="Compressed"):
                    await client.compute.logs(SID)

        asyncio.run(check())
    else:
        client = SyncAgentClient(api_key="test-key", base_url="https://test.invalid")
        client._http.close()
        client._http = httpx.Client(
            base_url="https://test.invalid", transport=httpx.MockTransport(answer)
        )
        with client:
            with pytest.raises(FlyMyAIAgentError, match="Compressed"):
                client.compute.logs(SID)
