"""An agent's preview crosses both SDK transports: written, pinned and read back."""

import asyncio
import json

import httpx

from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

PREVIEW = "Scores each creator's posts against their usual level and sends a digest."


def agent_row(preview="", pinned=False):
    row = {
        "uuid": "agent-1",
        "name": "Trend watch",
        "user_prompt": "Watch the creators.",
        "created_at": "2026-10-24T00:00:00Z",
        "updated_at": "2026-10-24T00:00:00Z",
    }
    if preview is not None:
        row["preview"] = preview
        row["preview_pinned"] = pinned
    return row


def handle(seen):
    def respond(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path.endswith("/legacy/"):
            # A backend before previews answers without either field.
            return httpx.Response(200, json=agent_row(preview=None))
        if request.method == "GET":
            return httpx.Response(200, json=agent_row(PREVIEW, pinned=True))
        if request.method == "POST":
            return httpx.Response(201, json=agent_row(body["preview"], pinned=True))
        text = body["preview"]
        return httpx.Response(200, json=agent_row(text, pinned=bool(text)))

    return respond


def check_calls(seen):
    assert seen == [
        (
            "POST",
            "/api/v1/agents/tasks/",
            {
                "name": "Trend watch",
                "user_prompt": "Watch the creators.",
                "preview": PREVIEW,
            },
        ),
        ("PATCH", "/api/v1/agents/tasks/agent-1/", {"preview": PREVIEW}),
        ("PATCH", "/api/v1/agents/tasks/agent-1/", {"preview": ""}),
        ("GET", "/api/v1/agents/tasks/agent-1/", None),
        ("GET", "/api/v1/agents/tasks/legacy/", None),
    ]


def test_sync_agent_preview():
    seen = []
    client = SyncAgentClient(api_key="test", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
    )
    try:
        created = client.agents.create(
            name="Trend watch", goal="Watch the creators.", preview=PREVIEW
        )
        assert (created.preview, created.preview_pinned) == (PREVIEW, True)
        pinned = client.agents.update("agent-1", preview=PREVIEW)
        assert (pinned.preview, pinned.preview_pinned) == (PREVIEW, True)
        handed_back = client.agents.update("agent-1", preview="")
        assert (handed_back.preview, handed_back.preview_pinned) == ("", False)
        detail = client.agents.get("agent-1")
        assert (detail.preview, detail.preview_pinned) == (PREVIEW, True)
        legacy = client.agents.get("legacy")
        assert (legacy.preview, legacy.preview_pinned) == ("", False)
    finally:
        client._http.close()
    check_calls(seen)


def test_async_agent_preview():
    seen = []

    async def check():
        client = AsyncAgentClient(api_key="test", base_url="http://testserver")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
        )
        try:
            created = await client.agents.create(
                name="Trend watch", goal="Watch the creators.", preview=PREVIEW
            )
            assert (created.preview, created.preview_pinned) == (PREVIEW, True)
            pinned = await client.agents.update("agent-1", preview=PREVIEW)
            assert (pinned.preview, pinned.preview_pinned) == (PREVIEW, True)
            handed_back = await client.agents.update("agent-1", preview="")
            assert (handed_back.preview, handed_back.preview_pinned) == ("", False)
            detail = await client.agents.get("agent-1")
            assert (detail.preview, detail.preview_pinned) == (PREVIEW, True)
            legacy = await client.agents.get("legacy")
            assert (legacy.preview, legacy.preview_pinned) == ("", False)
        finally:
            await client._http.aclose()

    asyncio.run(check())
    check_calls(seen)


def test_create_without_preview_sends_none():
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(201, json=agent_row())

    client = SyncAgentClient(api_key="test", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(respond)
    )
    try:
        client.agents.create(name="Trend watch", goal="Watch the creators.")
    finally:
        client._http.close()
    assert seen == [{"name": "Trend watch", "user_prompt": "Watch the creators."}]
