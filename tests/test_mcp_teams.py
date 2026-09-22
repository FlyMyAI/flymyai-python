import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from flymyai.agents._client import SyncAgentClient, AsyncAgentClient
from flymyai.agents._mcp_sharing import McpTeams, AsyncMcpTeams, McpTeamError


def response(payload, status=200):
    return httpx.Response(status, stream=httpx.ByteStream(json.dumps(payload).encode()))


def client_with(handler):
    client = SyncAgentClient(api_key="personal-key", base_url="https://example.test")
    client._http.close()
    client._http = httpx.Client(
        base_url="https://example.test",
        headers={"X-API-KEY": "personal-key"},
        transport=httpx.MockTransport(handler),
    )
    return client


def test_bounded_pages_query_parameters_and_personal_auth():
    def handler(request):
        assert request.headers["X-API-KEY"] == "personal-key"
        assert request.url.params["group"] == "action"
        assert request.url.params["cursor"] == "linear_list_issues"
        return response({
            "items": [{"key": "notion_search", "spent": "0.001", "calls": 1}],
            "next_cursor": None,
        })

    with client_with(handler) as client:
        page = client.teams.usage(uuid4(), group="action", cursor="linear_list_issues")
        assert page.items[0]["calls"] == 1


def test_device_is_one_time_secret_no_automatic_retry():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.headers["Idempotency-Key"] == "device-operation"
        return response({
            "id": str(uuid4()),
            "member_id": str(uuid4()),
            "label": "Laptop",
            "prefix": "fmst_abc",
            "expires_at": "2026-09-29T00:00:00Z",
            "token": "fmst_test-secret",
        })

    with client_with(handler) as client:
        device = client.teams.create_device(
            uuid4(), label="Laptop", idempotency_key="device-operation"
        )
        assert device.token.get_secret_value() == "fmst_test-secret"
        assert "test-secret" not in repr(device)
        assert "test-secret" not in device.model_dump_json()
    assert len(seen) == 1


@pytest.mark.parametrize(
    "status,payload",
    [
        (404, {"detail": "secret-from-server"}),
        (409, {"code": "stale_revision"}),
        (500, {"code": "secret-from-server"}),
        (302, {}),
    ],
)
def test_backend_errors_do_not_expose_payload_or_retry(status, payload):
    seen = []

    def handler(request):
        seen.append(request)
        return response(payload, status)

    with client_with(handler) as client, pytest.raises(McpTeamError) as caught:
        client.teams.get(uuid4())
    assert caught.value.status_code == status
    assert "secret-from-server" not in str(caught.value)
    assert len(seen) == 1


def test_response_limit_and_redirect_target_are_not_followed():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"x" * (256 * 1024 + 1)))

    with client_with(handler) as client, pytest.raises(
        McpTeamError, match="response_too_large"
    ):
        client.teams.list()
    assert len(seen) == 1


def test_invalid_path_and_missing_consent_are_not_dispatched():
    def handler(request):
        pytest.fail("Invalid input reached the transport")

    with client_with(handler) as client:
        with pytest.raises(ValueError):
            client.teams.get("../../users/")
        with pytest.raises(TypeError):
            client.teams.remove_member(uuid4(), uuid4())
        with pytest.raises(ValueError):
            client.teams.create_device(
                uuid4(), label="Laptop", idempotency_key="bad key"
            )


def test_async_surface_and_bounded_transport_match_sync():
    sync_names = {name for name in dir(McpTeams) if not name.startswith("_")}
    assert sync_names == {
        name for name in dir(AsyncMcpTeams) if not name.startswith("_")
    }

    async def run():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test",
            transport=httpx.MockTransport(
                lambda request: response({"items": [], "next_cursor": None})
            ),
        )
        try:
            assert (await client.teams.list()).items == []
        finally:
            await client._http.aclose()

    asyncio.run(run())
