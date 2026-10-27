"""A copy made before its services are connected (backend SHARING.md 1, 2.3):
``agents.copy`` and ``projects.copy`` send ``connect_later`` only when asked, and read
back what is left to connect and the chat that asks for it."""

import asyncio
import json

import httpx
import pytest

from flymyai.agents import AgentCopy
from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

AGENT = "9d767ed3-167f-4a82-9764-ed1cff0cfbd2"
FLEET_ID = "fleet:0b9c6f36-3f1f-4f5b-9a55-1f2a7c1f0e11"
CONNECTION = "0f6c2a52-6f0e-4d5b-9a3e-1c2d3e4f5a6b"
TO_CONNECT = [{
    "toolkit": "googlesheets",
    "title": "Google Sheets",
    "connection_id": CONNECTION,
    "connect_url": (
        "https://app.flymy.ai/mcp-configs#googlesheets?connection_id=" + CONNECTION
    ),
}]
SETUP = {
    "chat": "trn-sfer-cht",
    "needs": {"connections": ["googlesheets"], "to_connect": TO_CONNECT},
}


def agent_copy(setup=None):
    row = {
        "uuid": "7f000000-0000-4000-8000-000000000001",
        "name": "Instagram Trend Watcher",
        "user_prompt": "Follow the guide.",
        "created_at": "2026-10-26T00:00:00Z",
        "updated_at": "2026-10-26T00:00:00Z",
        "execution_id": "wlc-ome-cht",
        "template": {"slug": "instagram-trend-watcher", "version": "1.1"},
    }
    if setup is not None:
        row["setup"] = setup
    return row


def handle(seen):
    def respond(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, request.headers.get("Idempotency-Key"), body))
        later = bool(body and body.get("connect_later"))
        if "/tasks/" in request.url.path:
            return httpx.Response(201, json=agent_copy(SETUP if later else None))
        return httpx.Response(
            201,
            json={
                "schema": "flymy.project-copy.v1",
                "project": {"id": "fleet:7f000000-0000-4000-8000-000000000001"},
                "agents": [],
                "setup": SETUP,
            },
        )

    return respond


def check_calls(seen):
    assert seen == [
        (f"/api/v1/agents/tasks/{AGENT}/copy/", None, {}),
        (f"/api/v1/agents/tasks/{AGENT}/copy/", None, {"connect_later": True}),
        (f"/api/v1/agents/projects/{FLEET_ID}/copy/", "copy-1", {}),
        (
            f"/api/v1/agents/projects/{FLEET_ID}/copy/",
            "copy-2",
            {"name": "watcher", "connect_later": True},
        ),
    ]


def check_answers(plain, later, project):
    assert isinstance(plain, AgentCopy) and plain.setup is None
    assert plain.execution_id == "wlc-ome-cht"
    assert plain.template == {"slug": "instagram-trend-watcher", "version": "1.1"}
    assert later.setup["chat"] == "trn-sfer-cht"
    assert later.setup["needs"]["to_connect"] == TO_CONNECT
    assert project.setup["needs"]["to_connect"][0]["connect_url"].endswith(CONNECTION)


def test_sync_copies_send_connect_later_only_when_asked():
    seen = []
    client = SyncAgentClient(api_key="test", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
    )
    try:
        plain = client.agents.copy(AGENT)
        later = client.agents.copy(AGENT, connect_later=True)
        client.projects.copy(FLEET_ID, idempotency_key="copy-1")
        project = client.projects.copy(
            FLEET_ID, idempotency_key="copy-2", name="watcher", connect_later=True
        )
    finally:
        client._http.close()
    check_answers(plain, later, project)
    check_calls(seen)


def test_async_copies_send_connect_later_only_when_asked():
    seen = []

    async def check():
        client = AsyncAgentClient(api_key="test", base_url="http://testserver")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
        )
        try:
            plain = await client.agents.copy(AGENT)
            later = await client.agents.copy(AGENT, connect_later=True)
            await client.projects.copy(FLEET_ID, idempotency_key="copy-1")
            project = await client.projects.copy(
                FLEET_ID, idempotency_key="copy-2", name="watcher", connect_later=True
            )
        finally:
            await client._http.aclose()
        return plain, later, project

    check_answers(*asyncio.run(check()))
    check_calls(seen)


def test_connect_later_must_be_a_bool_and_nothing_is_sent_otherwise():
    def refuse(request):
        raise AssertionError("no request")

    client = SyncAgentClient(api_key="test", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(refuse)
    )
    try:
        with pytest.raises(ValueError):
            client.agents.copy(AGENT, connect_later="yes")
        with pytest.raises(ValueError):
            client.projects.copy(FLEET_ID, idempotency_key="k", connect_later=1)
    finally:
        client._http.close()
