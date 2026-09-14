"""Tests for flymyai.agents - SyncAgentClient, AsyncAgentClient, and helpers."""

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from flymyai.agents import (
    AgentClient,
    AgentStatus,
    AppendMessageResponse,
    AsyncAgentClient,
    CompilationStatus,
    ExecutionLogType,
    ExecutionStatus,
    FlyMyAIAgentError,
    SyncAgentClient,
)
from flymyai.agents._types import (
    Agent,
    AgentDetail,
    AvailableTool,
    Compilation,
    ExecutionLog,
    Run,
    RunDetail,
    RunStatus,
    RunStep,
    Tool,
)

NOW = datetime.now(tz=timezone.utc).isoformat()
RUN_ID = "won-gsfr-mxp"
PREVIOUS_RUN_ID = "won-prev-mxp"
COMPILATION_ID = "cmp-gsfr-mxp"


def _agent_payload(**overrides) -> dict:
    base = {
        "uuid": "aaaaaaaa-0000-0000-0000-000000000001",
        "name": "Test Agent",
        "user_prompt": "Do something useful",
        "available_tools": [],
        "all_tools_configured": True,
        "tools_need_to_configure": [],
        "generated_pipeline": {},
        "status": "draft",
        "created_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return base


def _run_payload(**overrides) -> dict:
    base = {
        "id": RUN_ID,
        "user_agent_task": 1,
        "previous_execution": None,
        "original_prompt": "Do something useful",
        "created_at": NOW,
        "updated_at": NOW,
        "messages": [],
        "status": "pending",
        "run_seq": 0,
        "error": None,
        "agent_result": None,
        "logs": [],
        "user_agent_task_uuid": "aaaaaaaa-0000-0000-0000-000000000001",
    }
    base.update(overrides)
    return base


def _append_message_payload(**overrides) -> dict:
    base = {
        "id": RUN_ID,
        "status": "running",
        "effort": "medium",
        "model": "claude-sonnet",
        "run_seq": 1,
        "error": None,
        "agent_result": None,
        "chat_files": [],
    }
    base.update(overrides)
    return base


def _log_payload(**overrides) -> dict:
    base = {
        "id": 1,
        "created_at": NOW,
        "updated_at": NOW,
        "type": "tool_called",
        "message": "Called search_web",
        "data": {},
    }
    base.update(overrides)
    return base


def _tool_payload(**overrides) -> dict:
    base = {
        "id": 7,
        "mcp_tool": "web_search",
        "user_config": {},
        "is_configured": True,
        "is_active": True,
        "unsafe_methods": [],
        "required_configuration_steps": [],
        "finished_configuration_steps": [],
        "next_configuration_step": None,
        "redirect_url": "",
        "response": "",
        "created_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return base


def _compilation_payload(**overrides) -> dict:
    base = {
        "id": COMPILATION_ID,
        "execution": RUN_ID,
        "status": "compiled",
        "script_code": "print('hello')",
        "result": None,
        "error": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return base


def _make_response(payload: Any, *, status_code: int = 200) -> httpx.Response:
    """Build a minimal httpx.Response whose .json() returns *payload*."""
    import json as _json

    raw = _json.dumps(payload).encode()
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=raw,
        request=httpx.Request("GET", "https://backend.flymy.ai/"),
    )


def _poll_status_payload(status="running", *, since=0, steps=(), run_seq=1, output=None):
    """Small canonical status pages for the existing wait/stream objectives."""
    assert len(steps) <= 2
    next_since = steps[-1]["id"] if steps else since
    settled = status == "completed"
    payload = {
        "view": "bounded_v1", "id": RUN_ID, "status": status, "run_seq": run_seq,
        "updated_at": NOW, "is_settled": settled, "step_count": next_since,
        "tool_step_count": next_since, "last_step_id": next_since or None,
        "new_steps": [{
            "id": step["id"], "type": step["type"], "message": step["message"],
            "message_size_bytes": len(step["message"].encode("utf-8")), "message_truncated": False,
            "label": step["message"], "label_size_bytes": len(step["message"].encode("utf-8")),
            "label_truncated": False,
        } for step in steps],
        "agent_surface_revision": 0, "page_size": 20, "has_more": False,
        "next_since": next_since, "poll_complete": settled, "step_count_has_more": False,
        "presentation_cursor_v1": {
            "version": "presentation_cursor_v1", "run_seq": run_seq, "as_of_seq": next_since,
        },
    }
    if output is not None:
        raw = _make_response(output).content
        digest = hashlib.sha256(raw).hexdigest()
        payload["result"] = {
            "version": "execution_resource_v1",
            "ref": f"execution-resource:v1:{RUN_ID}:{run_seq}:agent_result:sha256:{digest}:hmac-sha256:" + "a" * 64,
            "kind": "agent_result", "media_type": "application/json", "encoding": "utf-8",
            "size_bytes": len(raw), "sha256": digest, "truncated": False,
            "inline": {"format": "json", "value": output, "size_bytes": len(raw), "truncated": False},
            "retrieval": {"href": f"/api/v1/agents/executions/{RUN_ID}/resource/",
                          "accept_ranges": "bytes", "max_range_bytes": 65536},
            "receipt": {"kind": "chunked_postgres_v1", "chunk_bytes": 65536},
        }
    return payload


class _PollingBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content
        self.closed = False

    def __iter__(self):
        yield self.content

    async def __aiter__(self):
        yield self.content

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


class _PollingTransport(httpx.MockTransport):
    def __init__(self, pages):
        assert 1 <= len(pages) <= 3
        self.pages = pages
        self.requests = []
        self.bodies = []
        super().__init__(self._reply)

    def _reply(self, request):
        index = len(self.requests)
        assert index < len(self.pages), "Unexpected polling request or history fallback"
        since, payload = self.pages[index]
        assert request.method == "GET"
        assert request.url.scheme == "https" and request.url.host == "sdk.invalid"
        assert request.url.path == f"/api/v1/agents/executions/{RUN_ID}/status/"
        assert sorted(request.url.params.multi_items()) == [
            ("page_size", "20"), ("since", str(since)), ("view", "bounded_v1"),
        ]
        assert request.content == b""
        assert request.headers["X-API-KEY"] == "fly-test"
        assert request.headers["Accept-Encoding"] == "identity"
        assert request.headers.get("Idempotency-Key") is None
        self.requests.append(request)
        raw = _make_response(payload).content
        assert len(raw) <= 4096
        body = _PollingBody(raw)
        self.bodies.append(body)
        return httpx.Response(200, headers={"Content-Type": "application/json"},
                              stream=body, request=request)

    def assert_done(self):
        assert len(self.requests) == len(self.pages)
        assert all(body.closed for body in self.bodies)


def _polling_client(monkeypatch, transport, *, async_mode=False):
    """Preserve real SDK/httpx construction, substituting only its transport."""
    http_name = "AsyncClient" if async_mode else "Client"
    original = getattr(httpx, http_name)

    def with_transport(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, http_name, with_transport)
    client_type = AsyncAgentClient if async_mode else SyncAgentClient
    return client_type(api_key="fly-test", base_url="https://sdk.invalid", timeout=1)


def _sync_client(mock_http: MagicMock) -> SyncAgentClient:
    client = SyncAgentClient.__new__(SyncAgentClient)
    client._api_key = "fly-test"
    client._base_url = "https://backend.flymy.ai"
    client._max_retries = 2
    client._http = mock_http
    from flymyai.agents._resources import Agents, Compilations, Runs, Tools

    client.agents = Agents(client)
    client.runs = Runs(client)
    client.tools = Tools(client)
    client.compilations = Compilations(client)
    return client


def _async_client(mock_http: AsyncMock) -> AsyncAgentClient:
    client = AsyncAgentClient.__new__(AsyncAgentClient)
    client._api_key = "fly-test"
    client._base_url = "https://backend.flymy.ai"
    client._max_retries = 2
    client._http = mock_http
    from flymyai.agents._resources import (
        AsyncAgents,
        AsyncCompilations,
        AsyncRuns,
        AsyncTools,
    )

    client.agents = AsyncAgents(client)
    client.runs = AsyncRuns(client)
    client.tools = AsyncTools(client)
    client.compilations = AsyncCompilations(client)
    return client


class TestSyncClientConstruction:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("FLYMYAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key is required"):
            SyncAgentClient()

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FLYMYAI_API_KEY", "fly-env-key")
        client = SyncAgentClient()
        assert client._api_key == "fly-env-key"
        client.close()

    def test_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("FLYMYAI_API_KEY", raising=False)
        client = SyncAgentClient(api_key="fly-explicit")
        assert client._api_key == "fly-explicit"
        client.close()

    def test_alias_agent_client(self):
        assert AgentClient is SyncAgentClient

    def test_context_manager(self, monkeypatch):
        monkeypatch.setenv("FLYMYAI_API_KEY", "fly-ctx")
        with SyncAgentClient() as c:
            assert isinstance(c, SyncAgentClient)


class TestAsyncClientConstruction:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("FLYMYAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="api_key is required"):
            AsyncAgentClient()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, monkeypatch):
        monkeypatch.setenv("FLYMYAI_API_KEY", "fly-async")
        async with AsyncAgentClient() as c:
            assert isinstance(c, AsyncAgentClient)


class TestRaiseForStatus:
    def test_2xx_does_not_raise(self):
        from flymyai.agents._client import _raise_for_status

        resp = _make_response({"ok": True}, status_code=200)
        _raise_for_status(resp)  # should not raise

    def test_4xx_raises_with_detail(self):
        from flymyai.agents._client import _raise_for_status

        resp = _make_response({"detail": "Not found"}, status_code=404)
        with pytest.raises(FlyMyAIAgentError) as exc_info:
            _raise_for_status(resp)
        err = exc_info.value
        assert err.status_code == 404
        assert "Not found" in str(err)

    def test_5xx_raises_with_text_body(self):
        from flymyai.agents._client import _raise_for_status

        resp = httpx.Response(
            status_code=500,
            content=b"Internal Server Error",
            request=httpx.Request("GET", "https://backend.flymy.ai/"),
        )
        with pytest.raises(FlyMyAIAgentError) as exc_info:
            _raise_for_status(resp)
        assert exc_info.value.status_code == 500

    def test_error_repr(self):
        err = FlyMyAIAgentError("boom", status_code=422, response_body={"x": 1})
        assert "422" in repr(err)
        assert "boom" in repr(err)

    def test_204_returns_none(self):
        mock_http = MagicMock()
        mock_http.request.return_value = httpx.Response(
            status_code=204,
            content=b"",
            request=httpx.Request("DELETE", "https://backend.flymy.ai/"),
        )
        client = _sync_client(mock_http)
        result = client._request("DELETE", "/api/v1/agents/tasks/x/")
        assert result is None


class TestSyncAgents:
    def _client(self, payload, status_code=200):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(
            payload, status_code=status_code
        )
        return _sync_client(mock_http)

    def test_create_returns_agent(self):
        client = self._client(_agent_payload())
        agent = client.agents.create(name="Researcher", goal="Search the web")
        assert isinstance(agent, Agent)
        assert agent.name == "Test Agent"
        assert agent.goal == "Do something useful"
        assert agent.id == "aaaaaaaa-0000-0000-0000-000000000001"

    def test_create_sends_correct_body(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(_agent_payload())
        client = _sync_client(mock_http)
        client.agents.create(name="X", goal="Y", tools=[1, 2], status="active")
        _, call_kwargs = mock_http.request.call_args
        body = call_kwargs["json"]
        assert body["name"] == "X"
        assert body["user_prompt"] == "Y"
        assert body["available_tools"] == [1, 2]
        assert body["status"] == "active"

    def test_list_returns_agent_list(self):
        client = self._client([_agent_payload(), _agent_payload(uuid="bbbb-0002")])
        agents = client.agents.list()
        assert len(agents) == 2
        assert all(isinstance(a, Agent) for a in agents)

    def test_get_returns_agent_detail(self):
        client = self._client(_agent_payload())
        detail = client.agents.get("aaaaaaaa-0000-0000-0000-000000000001")
        assert isinstance(detail, AgentDetail)

    def test_update_translates_goal(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(_agent_payload())
        client = _sync_client(mock_http)
        client.agents.update("some-id", goal="new goal")
        _, call_kwargs = mock_http.request.call_args
        assert call_kwargs["json"]["user_prompt"] == "new goal"
        assert "goal" not in call_kwargs["json"]

    def test_delete_calls_correct_path(self):
        mock_http = MagicMock()
        mock_http.request.return_value = httpx.Response(
            204,
            content=b"",
            request=httpx.Request("DELETE", "https://backend.flymy.ai/"),
        )
        client = _sync_client(mock_http)
        client.agents.delete("some-uuid")
        args, _ = mock_http.request.call_args
        assert args[0] == "DELETE"
        assert "some-uuid" in args[1]

    def test_run_returns_run_detail(self):
        client = self._client(_run_payload())
        run = client.agents.run(
            "aaaaaaaa-0000-0000-0000-000000000001",
            idempotency_key="agent-run-sync-1",
        )
        assert isinstance(run, RunDetail)
        assert run.status == ExecutionStatus.PENDING


class TestSyncRuns:
    def _client_with_run(self, **overrides):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(_run_payload(**overrides))
        return _sync_client(mock_http)

    def test_get_returns_run_detail(self):
        client = self._client_with_run(status="running")
        run = client.runs.get(RUN_ID)
        assert isinstance(run, RunDetail)
        assert run.id == RUN_ID
        assert run.status == ExecutionStatus.RUNNING

    def test_list_returns_runs(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(
            [_run_payload(), _run_payload(id=43)]
        )
        client = _sync_client(mock_http)
        runs = client.runs.list()
        assert len(runs) == 2
        assert all(isinstance(r, Run) for r in runs)

    def test_list_follows_paginated_backend_response(self):
        mock_http = MagicMock()
        mock_http.request.side_effect = [
            _make_response({
                "next": "https://backend.flymy.ai/api/v1/agents/executions/?cursor=next-page",
                "previous": None,
                "results": [_run_payload()],
            }),
            _make_response({
                "next": None,
                "previous": "https://backend.flymy.ai/api/v1/agents/executions/?cursor=previous-page",
                "results": [_run_payload(id="run-second")],
            }),
        ]
        client = _sync_client(mock_http)

        runs = client.runs.list()

        assert [run.id for run in runs] == [RUN_ID, "run-second"]
        assert mock_http.request.call_count == 2
        assert mock_http.request.call_args_list[1].kwargs["params"] == {
            "cursor": "next-page",
            "view": "bounded_v1",
        }

    def test_cancel_calls_correct_endpoint(self):
        mock_http = MagicMock()
        mock_http.request.return_value = httpx.Response(
            204, content=b"", request=httpx.Request("POST", "https://backend.flymy.ai/")
        )
        client = _sync_client(mock_http)
        client.runs.cancel(RUN_ID)
        args, _ = mock_http.request.call_args
        assert "cancel" in args[1]
        assert RUN_ID in args[1]

    def test_append_message(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(
            _append_message_payload(agent_result={"answer": "done"})
        )
        client = _sync_client(mock_http)
        result = client.runs.append_message(
            RUN_ID,
            text="continue please",
            effort="high",
            model="gpt-5.6",
        )
        assert isinstance(result, AppendMessageResponse)
        assert result.output == {"answer": "done"}
        assert mock_http.request.call_count == 1
        _, call_kwargs = mock_http.request.call_args
        assert call_kwargs["json"] == {
            "text": "continue please",
            "effort": "high",
            "model": "gpt-5.6",
        }

    def test_wait_returns_on_completed(self, monkeypatch):
        transport = _PollingTransport([
            (0, _poll_status_payload()),
            (0, _poll_status_payload("completed", output={"answer": "42"})),
        ])
        with _polling_client(monkeypatch, transport) as client:
            result = client.runs.wait(RUN_ID, timeout=5, poll_interval=0, run_seq=1)
        assert isinstance(result, RunStatus)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.poll_complete and result.is_terminal
        assert result.output == {"answer": "42"}
        transport.assert_done()

    def test_wait_raises_on_timeout(self, monkeypatch):
        transport = _PollingTransport([(0, _poll_status_payload())])
        with _polling_client(monkeypatch, transport) as client:
            with pytest.raises(TimeoutError, match="observation timed out"):
                client.runs.wait(RUN_ID, timeout=0, poll_interval=0)
        transport.assert_done()

    def test_stream_events_yields_new_logs(self, monkeypatch):
        log1 = _log_payload(id=1)
        log2 = _log_payload(id=2, message="second")
        transport = _PollingTransport([
            (0, _poll_status_payload(steps=[log1])),
            (1, _poll_status_payload("completed", since=1, steps=[log2], run_seq=2)),
        ])
        events = []
        with _polling_client(monkeypatch, transport) as client:
            for event in client.runs.stream_events(RUN_ID, timeout=5, poll_interval=0, run_seq=1):
                assert len(events) < 2
                events.append(event)
        assert all(isinstance(event, RunStep) for event in events)
        assert [(event.id, event.observed_run_seq) for event in events] == [(1, 1), (2, 2)]
        assert events[0].message == "Called search_web"
        assert events[1].message == "second"
        transport.assert_done()

    def test_stream_events_no_duplicates(self, monkeypatch):
        log1 = _log_payload(id=1)
        transport = _PollingTransport([
            (0, _poll_status_payload(steps=[log1])),
            (1, _poll_status_payload(since=1, run_seq=2)),
            (1, _poll_status_payload("completed", since=1, run_seq=2)),
        ])
        events = []
        with _polling_client(monkeypatch, transport) as client:
            for event in client.runs.stream_events(RUN_ID, timeout=5, poll_interval=0, run_seq=1):
                assert len(events) < 1
                events.append(event)
        assert [event.id for event in events] == [1]
        assert events[0].observed_run_seq == 1
        transport.assert_done()


class TestSyncTools:
    def test_list_tools(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response([_tool_payload()])
        client = _sync_client(mock_http)
        tools = client.tools.list()
        assert len(tools) == 1
        assert isinstance(tools[0], Tool)
        assert tools[0].name == "web_search"

    def test_available_tools(self):
        available = {
            "name": "web_search",
            "type": "mcp",
            "title": "Web Search",
            "description": "Search the internet",
            "detail": "",
            "href": "",
            "categories": [],
            "instruction": None,
            "custom_class": None,
            "github_link": None,
            "configuration_steps": [],
        }
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response([available])
        client = _sync_client(mock_http)
        tools = client.tools.available()
        assert len(tools) == 1
        assert isinstance(tools[0], AvailableTool)

    def test_create_tool(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(_tool_payload())
        client = _sync_client(mock_http)
        tool = client.tools.create(mcp_tool="web_search")
        assert isinstance(tool, Tool)
        _, call_kwargs = mock_http.request.call_args
        assert call_kwargs["json"]["mcp_tool"] == "web_search"

    def test_update_tool(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(
            _tool_payload(user_config={"k": "v"})
        )
        client = _sync_client(mock_http)
        tool = client.tools.update(7, user_config={"k": "v"})
        assert tool.user_config == {"k": "v"}

    def test_delete_tool(self):
        mock_http = MagicMock()
        mock_http.request.return_value = httpx.Response(
            204,
            content=b"",
            request=httpx.Request("DELETE", "https://backend.flymy.ai/"),
        )
        client = _sync_client(mock_http)
        client.tools.delete(7)
        args, _ = mock_http.request.call_args
        assert args[0] == "DELETE"
        assert "7" in args[1]

    def test_provide_config(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(
            _tool_payload(is_configured=True)
        )
        client = _sync_client(mock_http)
        tool = client.tools.provide_config(7, user_response={"token": "abc"})
        assert tool.is_configured is True

    def test_call_tool(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response({"result": "found it"})
        client = _sync_client(mock_http)
        result = client.tools.call(
            7,
            action="search",
            arguments={"query": "AI"},
            idempotency_key="search-ai-v1",
        )
        assert result == {"result": "found it"}
        _, call_kwargs = mock_http.request.call_args
        assert call_kwargs["headers"]["Idempotency-Key"] == "search-ai-v1"


class TestSyncCompilations:
    def test_list_compilations(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response([_compilation_payload()])
        client = _sync_client(mock_http)
        comps = client.compilations.list()
        assert len(comps) == 1
        assert isinstance(comps[0], Compilation)

    def test_get_compilation(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(_compilation_payload())
        client = _sync_client(mock_http)
        comp = client.compilations.get(COMPILATION_ID)
        assert comp.id == COMPILATION_ID
        assert comp.execution == RUN_ID
        assert comp.script_code == "print('hello')"
        assert comp.status == CompilationStatus.COMPILED

    def test_compile_from_execution(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(_compilation_payload())
        client = _sync_client(mock_http)
        comp = client.compilations.compile(execution_id=RUN_ID)
        assert isinstance(comp, Compilation)
        assert comp.execution == RUN_ID
        args, _ = mock_http.request.call_args
        assert RUN_ID in args[1]

    def test_run_compilation(self):
        mock_http = MagicMock()
        client = _sync_client(mock_http)
        with pytest.raises(NotImplementedError, match="caller-owned replay contract"):
            client.compilations.run(COMPILATION_ID)
        mock_http.request.assert_not_called()


@pytest.mark.asyncio
class TestAsyncAgents:
    async def _client(self, payload):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(payload)
        return _async_client(mock_http)

    async def test_create(self):
        client = await self._client(_agent_payload())
        agent = await client.agents.create(name="Bot", goal="Do things")
        assert isinstance(agent, Agent)

    async def test_list(self):
        client = await self._client([_agent_payload()])
        agents = await client.agents.list()
        assert len(agents) == 1

    async def test_get(self):
        client = await self._client(_agent_payload())
        detail = await client.agents.get("aaaaaaaa-0000-0000-0000-000000000001")
        assert isinstance(detail, AgentDetail)

    async def test_update_translates_goal(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(_agent_payload())
        client = _async_client(mock_http)
        await client.agents.update("some-id", goal="new goal")
        _, call_kwargs = mock_http.request.call_args
        assert call_kwargs["json"]["user_prompt"] == "new goal"

    async def test_delete(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = httpx.Response(
            204,
            content=b"",
            request=httpx.Request("DELETE", "https://backend.flymy.ai/"),
        )
        client = _async_client(mock_http)
        await client.agents.delete("some-uuid")

    async def test_run(self):
        client = await self._client(_run_payload())
        run = await client.agents.run(
            "aaaaaaaa-0000-0000-0000-000000000001",
            idempotency_key="agent-run-async-1",
        )
        assert isinstance(run, RunDetail)


@pytest.mark.asyncio
class TestAsyncRuns:
    async def test_list_follows_paginated_backend_response(self):
        mock_http = AsyncMock()
        mock_http.request.side_effect = [
            _make_response({
                "next": "https://backend.flymy.ai/api/v1/agents/executions/?cursor=next-page",
                "previous": None,
                "results": [_run_payload()],
            }),
            _make_response({
                "next": None,
                "previous": "https://backend.flymy.ai/api/v1/agents/executions/?cursor=previous-page",
                "results": [_run_payload(id="run-second")],
            }),
        ]
        client = _async_client(mock_http)

        runs = await client.runs.list()

        assert [run.id for run in runs] == [RUN_ID, "run-second"]
        assert mock_http.request.await_count == 2
        assert mock_http.request.await_args_list[1].kwargs["params"] == {
            "cursor": "next-page",
            "view": "bounded_v1",
        }

    async def test_wait_completed(self, monkeypatch):
        transport = _PollingTransport([
            (0, _poll_status_payload()),
            (0, _poll_status_payload("completed")),
        ])
        async with _polling_client(monkeypatch, transport, async_mode=True) as client:
            result = await client.runs.wait(RUN_ID, timeout=5, poll_interval=0, run_seq=1)
        assert isinstance(result, RunStatus)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.poll_complete and result.is_terminal
        transport.assert_done()

    async def test_wait_timeout(self, monkeypatch):
        transport = _PollingTransport([(0, _poll_status_payload())])
        async with _polling_client(monkeypatch, transport, async_mode=True) as client:
            with pytest.raises(TimeoutError, match="observation timed out"):
                await client.runs.wait(RUN_ID, timeout=0, poll_interval=0)
        transport.assert_done()

    async def test_stream_events(self, monkeypatch):
        log1 = _log_payload(id=1)
        log2 = _log_payload(id=2)
        transport = _PollingTransport([
            (0, _poll_status_payload(steps=[log1])),
            (1, _poll_status_payload("completed", since=1, steps=[log2], run_seq=2)),
        ])
        events = []
        async with _polling_client(monkeypatch, transport, async_mode=True) as client:
            async for event in client.runs.stream_events(RUN_ID, timeout=5, poll_interval=0, run_seq=1):
                assert len(events) < 2
                events.append(event)
        assert all(isinstance(event, RunStep) for event in events)
        assert [(event.id, event.observed_run_seq) for event in events] == [(1, 1), (2, 2)]
        assert [event.message for event in events] == [log1["message"], log2["message"]]
        transport.assert_done()

    async def test_append_message(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(_append_message_payload())
        client = _async_client(mock_http)
        result = await client.runs.append_message(RUN_ID, text="go on")
        assert isinstance(result, AppendMessageResponse)
        assert mock_http.request.await_count == 1


@pytest.mark.asyncio
class TestAsyncTools:
    async def test_list(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response([_tool_payload()])
        client = _async_client(mock_http)
        tools = await client.tools.list()
        assert len(tools) == 1

    async def test_call(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response({"result": "ok"})
        client = _async_client(mock_http)
        r = await client.tools.call(
            7,
            action="ping",
            idempotency_key="async-ping-v1",
        )
        assert r == {"result": "ok"}
        _, call_kwargs = mock_http.request.await_args
        assert call_kwargs["headers"]["Idempotency-Key"] == "async-ping-v1"


@pytest.mark.asyncio
class TestAsyncCompilations:
    async def test_compile(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(_compilation_payload())
        client = _async_client(mock_http)
        comp = await client.compilations.compile(execution_id=RUN_ID)
        assert isinstance(comp, Compilation)
        assert comp.execution == RUN_ID

    async def test_run(self):
        mock_http = AsyncMock()
        client = _async_client(mock_http)
        with pytest.raises(NotImplementedError, match="caller-owned replay contract"):
            await client.compilations.run(COMPILATION_ID)
        mock_http.request.assert_not_awaited()


class TestModels:
    def test_run_is_terminal_true(self):
        run = RunDetail(**_run_payload(status="completed"))
        assert run.is_terminal is True

    def test_run_is_terminal_false(self):
        run = RunDetail(**_run_payload(status="running"))
        assert run.is_terminal is False

    def test_run_output_property(self):
        run = RunDetail(**_run_payload(agent_result={"key": "val"}))
        assert run.output == {"key": "val"}

    def test_run_accepts_string_previous_execution(self):
        run = RunDetail(**_run_payload(previous_execution=PREVIOUS_RUN_ID))
        assert run.previous_execution == PREVIOUS_RUN_ID

    def test_agent_id_property(self):
        agent = Agent(**_agent_payload())
        assert agent.id == agent.uuid

    def test_agent_goal_property(self):
        agent = Agent(**_agent_payload())
        assert agent.goal == agent.user_prompt

    def test_tool_name_property(self):
        tool = Tool(**_tool_payload())
        assert tool.name == "web_search"

    def test_execution_log_type_enum(self):
        log = ExecutionLog(**_log_payload(type="tool_called"))
        assert log.type == ExecutionLogType.TOOL_CALLED

    def test_agent_status_enum(self):
        agent = Agent(**_agent_payload(status="active"))
        assert agent.status == AgentStatus.ACTIVE

    def test_compilation_status_enum(self):
        comp = Compilation(**_compilation_payload(status="failed"))
        assert comp.status == CompilationStatus.FAILED


def test_idempotency_key_rejects_non_ascii_header_values():
    from flymyai.agents._resources import _idempotency_headers

    assert _idempotency_headers("run-42")["Idempotency-Key"] == "run-42"
    with pytest.raises(ValueError, match="printable ASCII"):
        _idempotency_headers("запуск-42")
    with pytest.raises(ValueError, match="leading or trailing spaces"):
        _idempotency_headers(" run-42")
    with pytest.raises(ValueError, match="leading or trailing spaces"):
        _idempotency_headers("run-42 ")
