"""The agent engine (backend docs/protocols/VERSIONING.md) crosses both SDK
transports: sent on create, read back, and engine 1 when an older server omits it."""

import asyncio
import json

import httpx

from flymyai.agents._client import AsyncAgentClient, SyncAgentClient


def agent_row(engine="flymy/agent@1"):
    row = {
        "uuid": "agent-1",
        "name": "Trend watch",
        "user_prompt": "Watch the creators.",
        "created_at": "2026-11-06T00:00:00Z",
        "updated_at": "2026-11-06T00:00:00Z",
    }
    if engine is not None:
        row["engine"] = engine
    return row


def handle(seen):
    def respond(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.url.path.endswith("/legacy/"):
            # A backend before engines answers without the field.
            return httpx.Response(200, json=agent_row(engine=None))
        if request.method == "POST":
            return httpx.Response(201, json=agent_row(body.get("engine", "flymy/agent@1")))
        return httpx.Response(200, json=agent_row())

    return respond


EXPECTED = [
    (
        "POST",
        "/api/v1/agents/tasks/",
        {"name": "Trend watch", "user_prompt": "Watch the creators.", "engine": "flymy/agent@1"},
    ),
    ("PATCH", "/api/v1/agents/tasks/agent-1/", {"engine": "flymy/agent@1"}),
    ("GET", "/api/v1/agents/tasks/legacy/", None),
]


def test_sync_agent_engine():
    seen = []
    client = SyncAgentClient(api_key="test", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
    )
    try:
        assert client.agents.create(
            name="Trend watch", goal="Watch the creators.", engine="flymy/agent@1"
        ).engine == "flymy/agent@1"
        assert client.agents.update("agent-1", engine="flymy/agent@1").engine == "flymy/agent@1"
        assert client.agents.get("legacy").engine == "flymy/agent@1"
    finally:
        client._http.close()
    assert seen == EXPECTED


def test_async_agent_engine():
    seen = []

    async def check():
        client = AsyncAgentClient(api_key="test", base_url="http://testserver")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
        )
        try:
            created = await client.agents.create(
                name="Trend watch", goal="Watch the creators.", engine="flymy/agent@1"
            )
            assert created.engine == "flymy/agent@1"
            assert (await client.agents.update("agent-1", engine="flymy/agent@1")).engine == "flymy/agent@1"
            assert (await client.agents.get("legacy")).engine == "flymy/agent@1"
        finally:
            await client._http.aclose()

    asyncio.run(check())
    assert seen == EXPECTED
