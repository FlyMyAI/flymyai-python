import asyncio
import json

import httpx
import pytest

from flymyai.agents import (
    AsyncMcpServers,
    McpServer,
    McpServerAuthorization,
    McpServerOAuthError,
    McpServers,
)
from flymyai.agents._client import AsyncAgentClient
from tests.test_mcp_teams import client_with, response

REDIRECT = "https://backend.flymy.ai/api/v1/agents/mcp-servers/oauth/callback/"
OAUTH = {
    "schema": "flymy.custom-mcp-oauth/v1",
    "authorized": False,
    "reconnect_required": False,
    "issuer": None,
    "scopes": [],
    "expires_at": None,
    "client": None,
    "client_id": None,
    "has_client_secret": False,
    "requested_scopes": None,
    "redirect_uri": REDIRECT,
}
SERVER = {
    "id": 7,
    "public_id": "4b8f2b3e-2c1d-4e5f-9a6b-7c8d9e0f1a2b",
    "name": "Linear",
    "url": "https://mcp.linear.app/mcp",
    "transport_type": "streamable_http",
    "auth_type": "oauth",
    "has_auth_credentials": False,
    "oauth": OAUTH,
    "is_active": True,
    "status": "pending",
    "status_detail": "This MCP server needs OAuth authorization.",
    "discovered_tools": [],
    "created_at": "2026-10-06T00:00:00Z",
    "updated_at": "2026-10-06T00:00:00Z",
}


def recorder(answers):
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append(
            (
                request.method,
                request.url.path,
                request.headers.get("Idempotency-Key"),
                body,
            )
        )
        status, payload = answers(request)
        return response(payload, status)

    return seen, handler


def test_an_oauth_server_is_added_without_credentials_then_authorized():
    seen, handler = recorder(
        lambda request: (
            (
                200,
                {
                    "authorize_url": "https://mcp.linear.app/authorize?state=s",
                    "redirect_uri": REDIRECT,
                    "expires_at": "2026-10-06T00:15:00Z",
                    "server": SERVER,
                },
            )
            if request.url.path.endswith("/authorize/")
            else (201, SERVER)
        )
    )
    client = client_with(handler)

    server = client.mcp_servers.create(
        name="Linear", url="https://mcp.linear.app/mcp", auth_type="oauth"
    )
    authorization = client.mcp_servers.authorize(server.id)

    assert isinstance(client.mcp_servers, McpServers)
    assert isinstance(server, McpServer) and server.oauth.redirect_uri == REDIRECT
    assert isinstance(authorization, McpServerAuthorization)
    assert authorization.authorize_url.startswith("https://mcp.linear.app/authorize")
    assert seen == [
        (
            "POST",
            "/api/v1/agents/mcp-servers/",
            None,
            {
                "name": "Linear",
                "url": "https://mcp.linear.app/mcp",
                "transport_type": "streamable_http",
                "auth_type": "oauth",
                "is_active": True,
            },
        ),
        ("POST", "/api/v1/agents/mcp-servers/7/authorize/", None, None),
    ]


def test_a_provider_without_registration_raises_the_redirect_url_to_register():
    _, handler = recorder(
        lambda request: (
            400,
            {
                "code": "oauth_client_required",
                "detail": "This provider has no automatic client registration.",
                "redirect_uri": REDIRECT,
            },
        )
    )
    client = client_with(handler)

    with pytest.raises(McpServerOAuthError) as raised:
        client.mcp_servers.authorize(7)

    assert raised.value.code == "oauth_client_required"
    assert raised.value.redirect_uri == REDIRECT
    assert raised.value.status_code == 400


def test_the_own_oauth_app_is_saved_with_only_the_fields_given():
    seen, handler = recorder(lambda request: (200, SERVER))
    client = client_with(handler)

    client.mcp_servers.update(
        7, oauth_client_id="1234.5678", oauth_client_secret="app-secret"
    )

    assert seen == [
        (
            "PATCH",
            "/api/v1/agents/mcp-servers/7/",
            None,
            {"oauth_client_id": "1234.5678", "oauth_client_secret": "app-secret"},
        )
    ]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"name": "x", "url": "http://local.test/mcp", "auth_type": "oauth"},
            "https",
        ),
        (
            {
                "name": "x",
                "url": "https://x.test/mcp",
                "auth_type": "oauth",
                "auth_credentials": {"token": "t"},
            },
            "no auth_credentials",
        ),
        (
            {
                "name": "x",
                "url": "https://x.test/mcp",
                "auth_type": "api_key",
                "oauth_client_id": "c",
            },
            "need auth_type oauth",
        ),
        ({"name": "x", "url": "https://x.test/mcp", "auth_type": "magic"}, "auth_type"),
    ],
)
def test_invalid_settings_fail_before_any_request(kwargs, message):
    seen, handler = recorder(lambda request: (201, SERVER))
    client = client_with(handler)

    with pytest.raises(ValueError, match=message):
        client.mcp_servers.create(**kwargs)

    assert seen == []


def test_listing_connecting_and_a_direct_call_with_its_key():
    connected = {**SERVER, "status": "connected", "oauth": {**OAUTH, "authorized": True}}
    seen, handler = recorder(
        lambda request: (
            (200, {"results": [connected]})
            if request.method == "GET"
            else (200, {"result": {"ok": True}})
            if request.url.path.endswith("/call/")
            else (200, connected)
        )
    )
    client = client_with(handler)

    servers = client.mcp_servers.list()
    server = client.mcp_servers.connect(7)
    result = client.mcp_servers.call(
        7, "list_issues", {"first": 5}, idempotency_key="linear-issues-1"
    )

    assert [s.oauth.authorized for s in servers] == [True]
    assert server.status == "connected"
    assert result == {"result": {"ok": True}}
    assert seen[-1] == (
        "POST",
        "/api/v1/agents/mcp-servers/7/call/",
        "linear-issues-1",
        {"action": "list_issues", "arguments": {"first": 5}},
    )


def test_async_authorize():
    async def scenario():
        def handler(request):
            assert request.url.path == "/api/v1/agents/mcp-servers/7/authorize/"
            return response(
                {
                    "authorize_url": "https://mcp.example.test/authorize",
                    "redirect_uri": REDIRECT,
                    "expires_at": "2026-10-06T00:15:00Z",
                    "server": SERVER,
                }
            )

        client = AsyncAgentClient(api_key="personal-key", base_url="https://example.test")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test",
            headers={"X-API-KEY": "personal-key"},
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(client.mcp_servers, AsyncMcpServers)
        authorization = await client.mcp_servers.authorize(7)
        await client._http.aclose()
        return authorization

    authorization = asyncio.run(scenario())
    assert authorization.server.id == 7
