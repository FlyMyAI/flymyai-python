import asyncio

import httpx
import pytest

from flymyai.agents import AppOperationStatus, AsyncAgentClient
from tests.test_apps import RELEASE
from tests.test_mcp_teams import client_with, response


KEY = "same apply ../?& +%"


@pytest.mark.parametrize("state", ["pending", "recorded", "not_found"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_read_only_reconciliation_keeps_fields_and_never_retries(state, asynchronous):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/v1/agents/apps/apply/"
        assert dict(request.url.params) == {"operation_key": KEY}
        assert request.headers.get("Idempotency-Key") is None
        assert not request.content
        return response({
            "schema": "flymy.app-operation.v1",
            "operation_key": KEY,
            "state": state,
            "retryable": False,
            "release_id": RELEASE if state == "recorded" else None,
            "future_field": "preserved",
        })

    async def go():
        client = AsyncAgentClient(api_key="fixture", base_url="https://example.test")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            return await client.apps.apply_status(KEY)
        finally:
            await client.close()

    if asynchronous:
        status = asyncio.run(go())
    else:
        with client_with(handler) as client:
            status = client.apps.apply_status(KEY)
    assert isinstance(status, AppOperationStatus)
    assert status.state == state
    assert status.retryable is False
    assert status.future_field == "preserved"
    assert len(seen) == 1


@pytest.mark.parametrize(
    "key", ["", " leading", "trailing ", "x" * 256, "🙂", "\u202e", "key\n"]
)
def test_invalid_keys_never_reach_transport(key):
    def handler(request):
        raise AssertionError("invalid key reached transport")

    with client_with(handler) as client, pytest.raises(ValueError):
        client.apps.apply_status(key)


@pytest.mark.parametrize("status_code", [401, 403, 404, 405, 503])
def test_unsupported_or_failed_lookup_has_no_apply_fallback(status_code):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.method == "GET"
        return response({"detail": "fixture failure"}, status_code)

    with client_with(handler) as client, pytest.raises(Exception) as failure:
        client.apps.apply_status(KEY)
    assert failure.value.status_code == status_code
    assert len(seen) == 1
