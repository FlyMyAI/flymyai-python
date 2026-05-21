"""Tests for flymyai.agents — SyncAgentClient, AsyncAgentClient, and helpers."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from flymyai.agents import (
    AgentClient,
    AgentStatus,
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
        run = client.agents.run("aaaaaaaa-0000-0000-0000-000000000001")
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
        mock_http.request.return_value = _make_response(_run_payload())
        client = _sync_client(mock_http)
        result = client.runs.append_message(RUN_ID, text="continue please")
        assert isinstance(result, RunDetail)
        _, call_kwargs = mock_http.request.call_args
        assert call_kwargs["json"]["text"] == "continue please"

    def test_wait_returns_on_completed(self):
        mock_http = MagicMock()
        mock_http.request.side_effect = [
            _make_response(_run_payload(status="running")),
            _make_response(
                _run_payload(status="completed", agent_result={"answer": "42"})
            ),
        ]
        client = _sync_client(mock_http)
        with patch("time.sleep"):  # don't actually sleep
            result = client.runs.wait(42, poll_interval=0.01)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.output == {"answer": "42"}

    def test_wait_raises_on_timeout(self):
        mock_http = MagicMock()
        mock_http.request.return_value = _make_response(_run_payload(status="running"))
        client = _sync_client(mock_http)
        with patch("time.sleep"), patch("time.monotonic", side_effect=[0, 0, 1000]):
            with pytest.raises(TimeoutError, match="did not complete"):
                client.runs.wait(42, timeout=1.0, poll_interval=0.01)

    def test_stream_events_yields_new_logs(self):
        log1 = _log_payload(id=1)
        log2 = _log_payload(id=2, message="second")
        mock_http = MagicMock()
        mock_http.request.side_effect = [
            _make_response(_run_payload(status="running", logs=[log1])),
            _make_response(_run_payload(status="completed", logs=[log1, log2])),
        ]
        client = _sync_client(mock_http)
        with patch("time.sleep"):
            events = list(client.runs.stream_events(42, poll_interval=0.01))
        assert len(events) == 2
        assert events[0].message == "Called search_web"
        assert events[1].message == "second"

    def test_stream_events_no_duplicates(self):
        log1 = _log_payload(id=1)
        mock_http = MagicMock()
        mock_http.request.side_effect = [
            _make_response(_run_payload(status="running", logs=[log1])),
            _make_response(_run_payload(status="completed", logs=[log1])),
        ]
        client = _sync_client(mock_http)
        with patch("time.sleep"):
            events = list(client.runs.stream_events(42, poll_interval=0.01))
        assert len(events) == 1


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
        result = client.tools.call(7, action="search", arguments={"query": "AI"})
        assert result == {"result": "found it"}


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
        mock_http.request.return_value = _make_response(
            _compilation_payload(status="completed", result={"output": "done"})
        )
        client = _sync_client(mock_http)
        comp = client.compilations.run(COMPILATION_ID)
        assert comp.status == CompilationStatus.COMPLETED
        assert comp.result == {"output": "done"}


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
        run = await client.agents.run("aaaaaaaa-0000-0000-0000-000000000001")
        assert isinstance(run, RunDetail)


@pytest.mark.asyncio
class TestAsyncRuns:
    async def test_wait_completed(self):
        mock_http = AsyncMock()
        mock_http.request.side_effect = [
            _make_response(_run_payload(status="running")),
            _make_response(_run_payload(status="completed")),
        ]
        client = _async_client(mock_http)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client.runs.wait(42, poll_interval=0.01)
        assert result.status == ExecutionStatus.COMPLETED

    async def test_wait_timeout(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(_run_payload(status="running"))
        client = _async_client(mock_http)
        with patch("asyncio.sleep", new_callable=AsyncMock), patch(
            "time.monotonic", side_effect=[0, 0, 1000]
        ):
            with pytest.raises(TimeoutError):
                await client.runs.wait(42, timeout=1.0, poll_interval=0.01)

    async def test_stream_events(self):
        log1 = _log_payload(id=1)
        log2 = _log_payload(id=2)
        mock_http = AsyncMock()
        mock_http.request.side_effect = [
            _make_response(_run_payload(status="running", logs=[log1])),
            _make_response(_run_payload(status="completed", logs=[log1, log2])),
        ]
        client = _async_client(mock_http)
        events = []
        with patch("asyncio.sleep", new_callable=AsyncMock):
            async for event in client.runs.stream_events(42, poll_interval=0.01):
                events.append(event)
        assert len(events) == 2

    async def test_append_message(self):
        mock_http = AsyncMock()
        mock_http.request.return_value = _make_response(_run_payload())
        client = _async_client(mock_http)
        result = await client.runs.append_message(RUN_ID, text="go on")
        assert isinstance(result, RunDetail)


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
        r = await client.tools.call(7, action="ping")
        assert r == {"result": "ok"}


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
        mock_http.request.return_value = _make_response(
            _compilation_payload(status="completed")
        )
        client = _async_client(mock_http)
        comp = await client.compilations.run(COMPILATION_ID)
        assert comp.status == CompilationStatus.COMPLETED


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
