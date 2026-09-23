import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from flymyai.agents._client import AsyncAgentClient
from flymyai.agents._mcp_sharing import McpTeamError
from tests.test_mcp_teams import client_with, response


def share_payload(identifier):
    return {
        "id": identifier,
        "connection_id": identifier,
        "workspace_id": identifier,
        "label": "Linear / work",
        "source_id": identifier,
        "source_kind": "native",
        "owner": {"id": 1, "username": "owner"},
        "recipient": {"id": 2, "username": "friend"},
        "state": "active",
        "expires_at": "2030-01-01T00:00:00Z",
        "payer": "owner",
    }


def test_personal_invitation_requires_exact_source_and_stable_key():
    identifier = str(uuid4())
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["Idempotency-Key"] == "email-operation"
        assert request.headers["X-API-KEY"] == "personal-key"
        assert request.url.path == "/api/v1/mcp-share-invitations/"
        body = json.loads(request.content)
        assert body == {
            "source_id": identifier,
            "email": "friend@example.test",
            "accept_billing": True,
            "access_days": 7,
        }
        return response({
            "id": identifier,
            "label": "Linear / work",
            "email": body["email"],
            "owner": {"id": 1, "username": "owner"},
            "state": "pending",
            "expires_at": "2030-01-01T00:00:00Z",
            "access_days": 7,
        })

    with client_with(handler) as client:
        invitation = client.shares.invite(
            identifier,
            "friend@example.test",
            accept_billing=True,
            idempotency_key="email-operation",
        )
        assert str(invitation.id) == identifier
        with pytest.raises(ValueError):
            client.shares.invite(
                "../users",
                "friend@example.test",
                accept_billing=True,
                idempotency_key="email-operation",
            )
    assert len(calls) == 1


def test_personal_listing_accept_revoke_and_discovery_are_explicit_http_calls():
    identifier = str(uuid4())
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/v1/mcp-shares/":
            assert request.url.params["scope"] == "received"
            return response({"items": [share_payload(identifier)], "next_cursor": None})
        if request.url.path.endswith("/tools/"):
            return response({"tools": []})
        return response(share_payload(identifier))

    with client_with(handler) as client:
        assert client.shares.list().items[0].label == "Linear / work"
        client.shares.accept(identifier)
        client.shares.tools(identifier)
        client.shares.revoke(identifier)
    assert seen == [
        ("GET", "/api/v1/mcp-shares/"),
        ("POST", f"/api/v1/mcp-share-invitations/{identifier}/accept/"),
        ("GET", f"/api/v1/mcp-shares/{identifier}/tools/"),
        ("DELETE", f"/api/v1/mcp-shares/{identifier}/"),
    ]


def test_personal_runtime_preserves_idempotency_and_sanitizes_denial_without_retry():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        assert request.headers["Idempotency-Key"] == "read-operation"
        return response(
            {"code": "access_revoked", "detail": "owner-key-must-not-escape"}, 403
        )

    with client_with(handler) as client:
        with pytest.raises(McpTeamError) as failure:
            client.shares.call(
                uuid4(), "reviewed_read", {}, idempotency_key="read-operation"
            )
        assert "owner-key" not in str(failure.value)
        with pytest.raises(ValueError):
            client.shares.call(
                uuid4(), "read", {"huge": "x" * 65536}, idempotency_key="read-operation"
            )
    assert count == 1


def test_async_personal_parity_preserves_bounded_transport():
    identifier = str(uuid4())
    calls = []

    async def scenario():
        def handler(request):
            calls.append(request)
            return response(share_payload(identifier))

        client = AsyncAgentClient(api_key="personal", base_url="https://example.test")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        async with client:
            result = await client.shares.accept(identifier)
            assert result.recipient.username == "friend"
            await client.shares.revoke(identifier)

    asyncio.run(scenario())
    assert len(calls) == 2
