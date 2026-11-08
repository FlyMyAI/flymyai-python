"""What a frozen version pins (backend docs/protocols/VERSIONING.md 2.6): a version
reads back its ``pins``, ``agents.update`` names the version it saved next to the live
one in ``new_version``, and a saved-version row carries its model and effort. Every
field is optional, so older servers read as before."""

import asyncio
import json
from unittest.mock import Mock

import httpx
import pytest

from flymyai.agents import (
    Agent,
    AgentUpdate,
    Compilation,
    McpAccessMode,
    NewVersion,
    SavedVersionSummary,
    VersionPins,
)
from flymyai.agents._client import AsyncAgentClient, SyncAgentClient
from flymyai.agents._resources import Compilations

AGENT = "aaaaaaaa-0000-0000-0000-000000000001"
RESOURCE_SET = "cccccccc-0000-0000-0000-000000000001"
NEXT = (
    "The live version is unchanged. Test the new version with run_frozen"
    " (compilation_id 42), then make it live with set_live_version."
)
NEW_VERSION = {"id": 42, "live": False, "source_version_id": 7, "next": NEXT}
PINS = {
    "llm_model": "claude-opus-5-5",
    "effort": "high",
    "web_search": None,
    "mcp_access_mode": "scoped",
    "tools": [3, 5],
    "custom_mcp_servers": [9],
    "mcp_resource_sets": [RESOURCE_SET],
    "skills": [{"id": 11, "slug": "report", "version": 2}],
}


def agent_row(**extra):
    return {
        "uuid": AGENT,
        "name": "Trend watch",
        "user_prompt": "Watch the creators.",
        "created_at": "2026-11-07T00:00:00Z",
        "updated_at": "2026-11-07T00:00:00Z",
        **extra,
    }


def compilation_row(**extra):
    return {
        "id": 42,
        "execution": "abc-defg-hij",
        "status": "compiled",
        "instruction_md": "# Watch",
        "created_at": "2026-11-07T00:00:00Z",
        "updated_at": "2026-11-07T00:00:00Z",
        "is_live": False,
        "source_compilation": 7,
        "user_agent_task": AGENT,
        **extra,
    }


def handle(seen):
    def respond(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "PATCH" and "model" in body:
            # A pinned setting changed on an agent with a live version.
            return httpx.Response(200, json=agent_row(new_version=NEW_VERSION))
        if request.method == "PATCH":
            return httpx.Response(200, json=agent_row())
        if request.url.path.endswith("/42/"):
            return httpx.Response(200, json=compilation_row(pins=PINS))
        if request.url.path.endswith("/41/"):
            return httpx.Response(200, json=compilation_row(id=41, pins=None))
        # A server before pins sends no key at all.
        return httpx.Response(200, json=compilation_row(id=40))

    return respond


EXPECTED = [
    ("PATCH", f"/api/v1/agents/tasks/{AGENT}/", {"model": "claude-sonnet-5-5"}),
    ("PATCH", f"/api/v1/agents/tasks/{AGENT}/", {"name": "Trend watch"}),
    ("GET", "/api/v1/agents/compilations/42/", None),
    ("GET", "/api/v1/agents/compilations/41/", None),
    ("GET", "/api/v1/agents/compilations/40/", None),
]


def check(updated, renamed, pinned, unpinned, legacy):
    assert isinstance(updated, AgentUpdate) and isinstance(updated, Agent)
    assert isinstance(updated.new_version, NewVersion)
    assert updated.new_version.id == 42
    assert updated.new_version.live is False
    assert updated.new_version.source_version_id == 7
    assert updated.new_version.next == NEXT
    assert renamed.new_version is None

    assert isinstance(pinned.pins, VersionPins)
    assert pinned.pins.llm_model == "claude-opus-5-5"
    assert pinned.pins.effort == "high"
    assert pinned.pins.web_search is None
    assert pinned.pins.mcp_access_mode is McpAccessMode.SCOPED
    assert pinned.pins.tools == [3, 5]
    assert pinned.pins.custom_mcp_servers == [9]
    assert pinned.pins.mcp_resource_sets == [RESOURCE_SET]
    assert [(s.id, s.slug, s.version) for s in pinned.pins.skills] == [
        (11, "report", 2)
    ]
    assert pinned.pins.model_dump(mode="json") == PINS
    assert unpinned.pins is None
    assert legacy.pins is None


def test_sync_update_names_the_new_version_and_a_version_reads_its_pins():
    seen = []
    client = SyncAgentClient(api_key="test", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
    )
    try:
        check(
            client.agents.update(AGENT, model="claude-sonnet-5-5"),
            client.agents.update(AGENT, name="Trend watch"),
            client.compilations.get(42),
            client.compilations.get(41),
            client.compilations.get(40),
        )
    finally:
        client._http.close()
    assert seen == EXPECTED


def test_async_update_names_the_new_version_and_a_version_reads_its_pins():
    seen = []

    async def run():
        client = AsyncAgentClient(api_key="test", base_url="http://testserver")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="http://testserver", transport=httpx.MockTransport(handle(seen))
        )
        try:
            check(
                await client.agents.update(AGENT, model="claude-sonnet-5-5"),
                await client.agents.update(AGENT, name="Trend watch"),
                await client.compilations.get(42),
                await client.compilations.get(41),
                await client.compilations.get(40),
            )
        finally:
            await client._http.aclose()

    asyncio.run(run())
    assert seen == EXPECTED


def test_pins_keep_what_a_newer_server_adds_and_default_what_it_omits():
    pins = Compilation(
        **compilation_row(
            pins={"llm_model": "gpt-6", "input_schema": {"type": "object"}}
        )
    ).pins
    assert pins.llm_model == "gpt-6"
    assert pins.effort == "" and pins.web_search is None
    assert pins.mcp_access_mode is McpAccessMode.LEGACY
    assert pins.tools == [] and pins.skills == []
    assert pins.input_schema == {"type": "object"}
    skill = Compilation(
        **compilation_row(pins={"skills": [{"id": 1, "slug": "s", "version": None}]})
    ).pins.skills[0]
    assert skill.version is None


def saved_page(**row_extra):
    row = {
        "id": 42,
        "execution": "abc-defg-hij",
        "user_agent_task": AGENT,
        "status": "compiled",
        "last_run_status": None,
        "name": "Worker",
        "note": "Saved with the agent's new settings (update_agent).",
        "note_truncated": False,
        "is_live": False,
        "has_instruction": True,
        "cron_schedule": "",
        "timezone": "UTC",
        "schedule_active": False,
        "schedule_paused": False,
        "schedule_paused_reason": "",
        "schedule_paused_reason_truncated": False,
        "schedule_ready": True,
        "source_compilation": 7,
        "engine": "flymy/agent@1",
        "created_at": "2026-11-07T00:00:00Z",
        "updated_at": "2026-11-07T00:00:00Z",
        **row_extra,
    }
    return {
        "schema": "flymy.saved-versions.v1",
        "task": AGENT,
        "results": [row],
        "next_cursor": None,
    }


@pytest.mark.parametrize(
    "row_extra,model,effort",
    [
        ({"model": "claude-opus-5-5", "effort": "high"}, "claude-opus-5-5", "high"),
        ({"model": "claude-opus-5-5", "effort": ""}, "claude-opus-5-5", ""),
        # A version made before pins.
        ({"model": None, "effort": None}, None, None),
        # A server before pins sends neither key.
        ({}, None, None),
    ],
)
def test_a_saved_version_row_reads_its_model_and_effort(row_extra, model, effort):
    client = Mock()
    client._request.return_value = saved_page(**row_extra)
    row = Compilations(client).list_versions(AGENT).results[0]
    assert isinstance(row, SavedVersionSummary)
    assert (row.model, row.effort) == (model, effort)


@pytest.mark.parametrize(
    "row_extra",
    [{"model": 5}, {"effort": True}, {"model": "m" * 129}, {"effort": "e" * 17}],
)
def test_a_saved_version_row_with_a_malformed_model_or_effort_is_refused(row_extra):
    client = Mock()
    client._request.return_value = saved_page(**row_extra)
    with pytest.raises(ValueError):
        Compilations(client).list_versions(AGENT)
