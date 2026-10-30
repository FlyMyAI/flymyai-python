"""Saved-version discovery stays one-page, scoped, and separate from full details."""

from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from flymyai.agents import SavedVersionPage, SavedVersionSummary
from flymyai.agents._resources import AsyncCompilations, Compilations

AGENT = "aaaaaaaa-0000-0000-0000-000000000001"
OTHER = "bbbbbbbb-0000-0000-0000-000000000001"


def page():
    return {
        "schema": "flymy.saved-versions.v1",
        "task": AGENT,
        "results": [
            {
                "id": 2,
                "execution": "abc-defg-hij",
                "user_agent_task": AGENT,
                "status": "completed",
                "last_run_status": "completed",
                "name": "Worker",
                "note": "",
                "note_truncated": False,
                "is_live": True,
                "has_instruction": True,
                "cron_schedule": "",
                "timezone": "UTC",
                "schedule_active": False,
                "schedule_paused": False,
                "schedule_paused_reason": "",
                "schedule_paused_reason_truncated": False,
                "schedule_ready": True,
                "source_compilation": None,
                "created_at": "2026-10-30T00:00:00Z",
                "updated_at": "2026-10-30T00:00:00Z",
            }
        ],
        "next_cursor": "opaque-older",
    }


def test_sync_fetches_one_page_preserves_cursor_and_public_types():
    client = Mock()
    client._request.return_value = page()
    result = Compilations(client).list_versions(AGENT.upper(), limit=1)
    assert isinstance(result, SavedVersionPage)
    assert isinstance(result.results[0], SavedVersionSummary)
    assert result.results[0].id == 2
    assert not hasattr(result.results[0], "instruction_md")
    assert result.next_cursor == "opaque-older"
    assert result.schema_version == "flymy.saved-versions.v1"
    client._request.assert_called_once_with(
        "GET",
        "/api/v1/agents/compilations/versions/",
        params={"task": AGENT, "limit": 1},
    )
    client._request.return_value = {**page(), "results": [], "next_cursor": None}
    older = Compilations(client).list_versions(AGENT, cursor=result.next_cursor)
    assert older.results == [] and older.next_cursor is None
    client._request.assert_called_with(
        "GET",
        "/api/v1/agents/compilations/versions/",
        params={"task": AGENT, "limit": 20, "cursor": "opaque-older"},
    )


@pytest.mark.asyncio
async def test_async_fetches_one_page_without_following_cursor():
    client = Mock(_request=AsyncMock(return_value=page()))
    result = await AsyncCompilations(client).list_versions(AGENT, cursor="opaque-newer")
    assert result.next_cursor == "opaque-older" and result.results[0].is_live
    client._request.assert_awaited_once_with(
        "GET",
        "/api/v1/agents/compilations/versions/",
        params={"task": AGENT, "limit": 20, "cursor": "opaque-newer"},
    )


@pytest.mark.parametrize(
    "agent,kwargs",
    [
        ("", {}),
        ("worker", {}),
        (1, {}),
        (None, {}),
        *[(AGENT, {"limit": limit}) for limit in [0, 51, 1.2, True, "20", None]],
        *[(AGENT, {"cursor": cursor}) for cursor in ["", " ", "a\n", "x" * 513, 1]],
    ],
)
@pytest.mark.asyncio
async def test_invalid_input_refused_before_sync_or_async_dispatch(agent, kwargs):
    sync = Mock()
    async_client = Mock(_request=AsyncMock())
    with pytest.raises(ValueError):
        Compilations(sync).list_versions(agent, **kwargs)
    with pytest.raises(ValueError):
        await AsyncCompilations(async_client).list_versions(agent, **kwargs)
    sync._request.assert_not_called()
    async_client._request.assert_not_awaited()


def invalid_pages():
    p = page()
    row = p["results"][0]
    return [
        p["results"],
        {**p, "schema": "future"},
        {**p, "task": OTHER},
        {**p, "results": [{**row, "user_agent_task": OTHER}]},
        {**p, "results": [row, row]},
        {**p, "results": [row, {**row, "id": 3}]},
        {**p, "results": [{**row, "instruction_md": "private source"}]},
        {**p, "results": [{**row, "note": "x" * 241}]},
        {**p, "results": [{**row, "id": True}]},
        {**p, "results": [{**row, "is_live": "true"}]},
        {**p, "next_cursor": ""},
        {**p, "next_cursor": "x" * 513},
        {**p, "extra": "x" * 65536},
        {**p, "results": [{**row, "id": 100 - i} for i in range(21)]},
    ]


@pytest.mark.parametrize("payload", invalid_pages())
@pytest.mark.asyncio
async def test_invalid_response_never_falls_back_or_follows_another_endpoint(payload):
    sync = Mock()
    sync._request.return_value = deepcopy(payload)
    async_client = Mock(_request=AsyncMock(return_value=deepcopy(payload)))
    with pytest.raises(ValueError):
        Compilations(sync).list_versions(AGENT)
    with pytest.raises(ValueError):
        await AsyncCompilations(async_client).list_versions(AGENT)
    assert sync._request.call_count == 1
    assert async_client._request.await_count == 1


def test_repeated_cursor_refused_and_unrecognized_metadata_not_retained():
    client = Mock()
    client._request.return_value = page()
    with pytest.raises(ValueError, match="cursor"):
        Compilations(client).list_versions(AGENT, cursor="opaque-older")
    payload = page()
    payload["results"][0]["future_metadata"] = "not copied"
    client._request.return_value = payload
    result = Compilations(client).list_versions(AGENT)
    assert not hasattr(result.results[0], "future_metadata")


def test_json_escaping_counts_toward_response_limit():
    payload = page()
    row = payload["results"][0]
    payload["results"] = [
        {
            **row,
            "id": 100 - i,
            "name": "\x01" * 200,
            "note": "\x01" * 240,
            "schedule_paused_reason": "\x01" * 240,
        }
        for i in range(50)
    ]
    client = Mock()
    client._request.return_value = payload
    with pytest.raises(ValueError, match="64 KiB"):
        Compilations(client).list_versions(AGENT, limit=50)


@pytest.mark.asyncio
async def test_validation_errors_do_not_echo_private_summary_content():
    marker = "private saved-version note"
    payload = page()
    payload["results"][0]["note"] = marker * 30
    sync = Mock()
    sync._request.return_value = payload
    async_client = Mock(_request=AsyncMock(return_value=payload))
    with pytest.raises(ValueError) as sync_error:
        Compilations(sync).list_versions(AGENT)
    with pytest.raises(ValueError) as async_error:
        await AsyncCompilations(async_client).list_versions(AGENT)
    with pytest.raises(ValueError) as summary_error:
        SavedVersionSummary(**payload["results"][0])
    for error in (sync_error.value, async_error.value, summary_error.value):
        assert marker not in str(error)
