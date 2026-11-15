"""Offline HTTP contracts for bounded saved-version discovery, sync and async."""

import copy

import httpx
import pytest

from flymyai.agents import (
    AsyncAgentClient,
    CompilationSummaryPage,
    FlyMyAIAgentError,
    SyncAgentClient,
)

AGENT = "aaaaaaaa-0000-0000-0000-000000000001"
FOREIGN = "bbbbbbbb-0000-0000-0000-000000000001"
NOW = "2026-10-11T00:00:00Z"


def summary(version_id=2):
    return {
        "id": version_id,
        "execution": "abc-defg-hij",
        "user_agent_task": AGENT,
        "status": "failed",
        "last_run_status": "failed",
        "name": "Worker",
        "note": "Preview",
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
        "created_at": NOW,
        "updated_at": NOW,
    }


def page(*items, **overrides):
    return {
        "schema": "flymy.saved-versions.v1",
        "task": AGENT,
        "results": list(items),
        "next_cursor": None,
        **overrides,
    }


def install_sync_transport(client, handler):
    client._http.close()
    client._http = httpx.Client(
        base_url="https://backend.invalid",
        headers={"X-API-KEY": "offline-dummy"},
        transport=httpx.MockTransport(handler),
    )


def test_one_page_explicit_continuation_and_detail_preserve_unknown_metadata():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["X-API-KEY"] == "offline-dummy"
        if request.url.path.endswith("/versions/"):
            assert request.headers["Accept-Encoding"] == "identity"
        if request.url.path.endswith("/versions/"):
            assert request.url.params["task"] == AGENT
            assert request.url.params["limit"] == "20"
            if request.url.params.get("cursor"):
                assert request.url.params["cursor"] == "opaque-next"
                return httpx.Response(200, json=page(summary(1)))
            item = {**summary(), "future_metadata": "retained"}
            return httpx.Response(200, json=page(item, next_cursor="opaque-next"))
        assert request.url.path == "/api/v1/agents/compilations/2/"
        return httpx.Response(
            200,
            json={
                "id": 2,
                "execution": "abc-defg-hij",
                "status": "compiled",
                "instruction_md": "Explicit full instruction",
                "created_at": NOW,
                "updated_at": NOW,
            },
        )

    with SyncAgentClient(api_key="offline-dummy", max_retries=0) as client:
        install_sync_transport(client, handler)
        first = client.compilations.list_versions(AGENT)
        assert isinstance(first, CompilationSummaryPage)
        assert len(requests) == 1  # next_cursor never starts a second request
        assert first.results[0].model_extra["future_metadata"] == "retained"
        assert "instruction_md" not in first.results[0].model_dump()
        second = client.compilations.list_versions(AGENT, cursor=first.next_cursor)
        assert [item.id for item in second.results] == [1]
        assert len(requests) == 2
        assert client.compilations.get(2).instruction_md == "Explicit full instruction"
        assert len(requests) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent_id": "bad"},
        {"limit": 0},
        {"limit": 51},
        {"limit": 1.5},
        {"limit": True},
        {"cursor": ""},
        {"cursor": " "},
        {"cursor": "x\n"},
        {"cursor": "x" * 513},
    ],
)
def test_invalid_inputs_do_not_dispatch(kwargs):
    requests = []
    with SyncAgentClient(api_key="offline-dummy", max_retries=0) as client:
        install_sync_transport(client, lambda request: requests.append(request))
        with pytest.raises(ValueError):
            client.compilations.list_versions(**{"agent_id": AGENT, **kwargs})
    assert not requests


@pytest.mark.parametrize(
    "payload",
    [
        [summary()],
        page(summary(), schema="future"),
        page(summary(), task=FOREIGN),
        page({**summary(), "user_agent_task": FOREIGN}),
        page(summary(1), summary(2)),
        page(summary(2), summary(2)),
        page({**summary(), "instruction_md": "must fetch explicitly"}),
        page({**summary(), "note": "x" * 241}),
        page(summary(), next_cursor="x" * 513),
        page(*[summary(100 - i) for i in range(51)]),
        page(summary(), future_metadata="x" * (64 * 1024)),
    ],
)
def test_wrong_or_oversized_projection_fails_closed(payload):
    with SyncAgentClient(api_key="offline-dummy", max_retries=0) as client:
        install_sync_transport(
            client, lambda request: httpx.Response(200, json=copy.deepcopy(payload))
        )
        with pytest.raises(ValueError):
            client.compilations.list_versions(AGENT, limit=50)


@pytest.mark.asyncio
async def test_async_is_single_page_and_legacy_list_still_returns_full_instruction():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/versions/"):
            assert request.url.params["limit"] == "1"
            return httpx.Response(200, json=page(summary(), next_cursor="opaque-next"))
        assert request.url.path == "/api/v1/agents/compilations/"
        assert request.url.params["task"] == AGENT
        return httpx.Response(
            200,
            json=[{
                "id": 2,
                "execution": "abc-defg-hij",
                "instruction_md": "Legacy full text",
                "created_at": NOW,
                "updated_at": NOW,
            }],
        )

    async with AsyncAgentClient(api_key="offline-dummy", max_retries=0) as client:
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://backend.invalid",
            transport=httpx.MockTransport(handler),
        )
        first = await client.compilations.list_versions(AGENT, limit=1)
        assert first.next_cursor == "opaque-next"
        assert len(requests) == 1
        legacy = await client.compilations.list(agent_id=AGENT)
        assert legacy[0].instruction_md == "Legacy full text"
        assert len(requests) == 2


def test_streaming_cap_stops_reading_and_closes_response_before_parsing():
    read_bytes = []
    closed = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(1000):
                read_bytes.append(8192)
                yield b"x" * 8192

        def close(self):
            closed.append(True)

    with SyncAgentClient(api_key="offline-dummy", max_retries=0) as client:
        install_sync_transport(
            client, lambda request: httpx.Response(200, stream=Stream())
        )
        with pytest.raises(ValueError, match="byte limit"):
            client.compilations.list_versions(AGENT)
    assert sum(read_bytes) <= 64 * 1024 + 8192
    assert closed == [True]


@pytest.mark.asyncio
async def test_async_streaming_cap_also_stops_and_closes():
    read_bytes = []
    closed = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(1000):
                read_bytes.append(8192)
                yield b"x" * 8192

        async def aclose(self):
            closed.append(True)

    async with AsyncAgentClient(api_key="offline-dummy", max_retries=0) as client:
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://backend.invalid",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=Stream())
            ),
        )
        with pytest.raises(ValueError, match="byte limit"):
            await client.compilations.list_versions(AGENT)
    assert sum(read_bytes) <= 64 * 1024 + 8192
    assert closed == [True]


def test_unexpected_encoding_rejected_and_http_ownership_error_preserved():
    class MustNotRead(httpx.SyncByteStream):
        def __iter__(self):
            raise AssertionError("Unexpected encoding must fail before decoding")

    with SyncAgentClient(api_key="offline-dummy", max_retries=0) as client:
        install_sync_transport(
            client,
            lambda request: httpx.Response(
                200, headers={"Content-Encoding": "gzip"}, stream=MustNotRead()
            ),
        )
        with pytest.raises(ValueError, match="encoding"):
            client.compilations.list_versions(AGENT)
        install_sync_transport(
            client, lambda request: httpx.Response(404, json={"detail": "Not found."})
        )
        with pytest.raises(FlyMyAIAgentError) as error:
            client.compilations.list_versions(FOREIGN)
        assert error.value.status_code == 404
