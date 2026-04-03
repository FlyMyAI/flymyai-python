from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from flymyai.agents._types import (
    Agent,
    AgentDetail,
    AvailableTool,
    Compilation,
    Run,
    RunDetail,
    Tool,
)

if TYPE_CHECKING:
    from flymyai.agents._client import AsyncAgentClient, SyncAgentClient


# ── helpers ──────────────────────────────────────────────────────────────────

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


# ══════════════════════════════════════════════════════════════════════════════
#  SYNC resources
# ══════════════════════════════════════════════════════════════════════════════


class Agents:
    """CRUD for agents. Maps to ``/api/v1/agents/tasks/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    # -- create ----------------------------------------------------------------

    def create(
        self,
        *,
        name: str,
        goal: str,
        tools: Optional[List[int]] = None,
        status: Optional[str] = None,
    ) -> Agent:
        """Create a new agent.

        Parameters
        ----------
        name:
            Human-readable agent name.
        goal:
            The agent's prompt / instructions (stored as ``user_prompt``).
        tools:
            List of ``UserMcpTool`` IDs to attach.
        status:
            Initial status (default ``draft``).
        """
        body: Dict[str, Any] = {"name": name, "user_prompt": goal}
        if tools is not None:
            body["available_tools"] = tools
        if status is not None:
            body["status"] = status
        data = self._c._request("POST", "/api/v1/agents/tasks/", json=body)
        return Agent(**data)

    # -- list ------------------------------------------------------------------

    def list(self) -> List[Agent]:
        """Return all non-archived agents for the current user."""
        data = self._c._request("GET", "/api/v1/agents/tasks/")
        return [Agent(**item) for item in data]

    # -- get -------------------------------------------------------------------

    def get(self, agent_id: str) -> AgentDetail:
        """Get agent by UUID (returns full detail with nested tools)."""
        data = self._c._request("GET", f"/api/v1/agents/tasks/{agent_id}/")
        return AgentDetail(**data)

    # -- update ----------------------------------------------------------------

    def update(self, agent_id: str, **kwargs: Any) -> Agent:
        """Partial update (PATCH).

        Use ``goal=`` to update ``user_prompt``.
        """
        if "goal" in kwargs:
            kwargs["user_prompt"] = kwargs.pop("goal")
        data = self._c._request(
            "PATCH", f"/api/v1/agents/tasks/{agent_id}/", json=kwargs
        )
        return Agent(**data)

    # -- delete ----------------------------------------------------------------

    def delete(self, agent_id: str) -> None:
        """Soft-delete (archive) an agent."""
        self._c._request("DELETE", f"/api/v1/agents/tasks/{agent_id}/")

    # -- run -------------------------------------------------------------------

    def run(self, agent_id: str) -> RunDetail:
        """Create an execution and start the agent loop.

        Returns the newly created :class:`RunDetail` (status will be ``pending``).
        """
        data = self._c._request(
            "POST", f"/api/v1/agents/tasks/{agent_id}/run-loop/"
        )
        return RunDetail(**data)


class Runs:
    """Manage agent executions (runs). Maps to ``/api/v1/agents/executions/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def create(self, *, agent_id: str) -> RunDetail:
        """Create a new run for the given agent.

        This is a convenience alias for ``client.agents.run(agent_id)``.
        """
        return self._c.agents.run(agent_id)

    def list(self) -> List[Run]:
        """List all executions for the current user (newest first)."""
        data = self._c._request("GET", "/api/v1/agents/executions/")
        return [Run(**item) for item in data]

    def get(self, run_id: int) -> RunDetail:
        """Get a single execution with logs."""
        data = self._c._request("GET", f"/api/v1/agents/executions/{run_id}/")
        return RunDetail(**data)

    def cancel(self, run_id: int) -> None:
        """Cancel a running execution."""
        self._c._request("POST", f"/api/v1/agents/executions/{run_id}/cancel/")

    def append_message(self, run_id: int, *, text: str) -> RunDetail:
        """Append a user message to the conversation and restart the agent loop."""
        data = self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/append-message/",
            json={"text": text},
        )
        return RunDetail(**data)

    def wait(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Poll until the run reaches a terminal status.

        Parameters
        ----------
        run_id:
            Execution ID.
        timeout:
            Max seconds to wait before raising ``TimeoutError``.
        poll_interval:
            Seconds between polls.
        """
        deadline = time.monotonic() + timeout
        while True:
            result = self.get(run_id)
            if result.status in _TERMINAL_STATUSES:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Run {run_id} did not complete within {timeout}s "
                    f"(last status: {result.status})"
                )
            time.sleep(poll_interval)

    def stream_events(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 1.0,
    ):
        """Yield new :class:`ExecutionLog` entries as they appear.

        Polls the execution detail endpoint and yields logs that haven't been
        seen yet.  Stops when the run reaches a terminal status.
        """
        seen_ids: set = set()
        deadline = time.monotonic() + timeout
        while True:
            detail = self.get(run_id)
            for log in detail.logs:
                if log.id not in seen_ids:
                    seen_ids.add(log.id)
                    yield log
            if detail.status in _TERMINAL_STATUSES:
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(poll_interval)


class Tools:
    """Manage MCP tools. Maps to ``/api/v1/agents/tools/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def list(self) -> List[Tool]:
        """List the user's configured tools."""
        data = self._c._request("GET", "/api/v1/agents/tools/")
        return [Tool(**item) for item in data]

    def available(self) -> List[AvailableTool]:
        """List available tools from the catalog (no auth required)."""
        data = self._c._request("GET", "/api/v1/agents/tools/available/")
        return [AvailableTool(**item) for item in data]

    def create(self, *, mcp_tool: str, **kwargs: Any) -> Tool:
        """Add a tool to the user's account."""
        body = {"mcp_tool": mcp_tool, **kwargs}
        data = self._c._request("POST", "/api/v1/agents/tools/", json=body)
        return Tool(**data)

    def get(self, tool_id: int) -> Tool:
        data = self._c._request("GET", f"/api/v1/agents/tools/{tool_id}/")
        return Tool(**data)

    def update(self, tool_id: int, **kwargs: Any) -> Tool:
        """Partial update (PATCH).  Pass ``user_config={...}`` to merge config."""
        data = self._c._request(
            "PATCH", f"/api/v1/agents/tools/{tool_id}/", json=kwargs
        )
        return Tool(**data)

    def delete(self, tool_id: int) -> None:
        self._c._request("DELETE", f"/api/v1/agents/tools/{tool_id}/")

    def provide_config(
        self, tool_id: int, *, user_response: Any
    ) -> Tool:
        """Answer the current ``ask_user`` configuration step."""
        data = self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/provide-config/",
            json={"user_response": user_response},
        )
        return Tool(**data)

    def call(
        self,
        tool_id: int,
        *,
        action: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Invoke a custom-class tool action directly."""
        data = self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/call/",
            json={"action": action, "arguments": arguments or {}},
        )
        return data


class Compilations:
    """Script compilations. Maps to ``/api/v1/agents/compilations/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def list(self) -> List[Compilation]:
        data = self._c._request("GET", "/api/v1/agents/compilations/")
        return [Compilation(**item) for item in data]

    def get(self, compilation_id: int) -> Compilation:
        data = self._c._request(
            "GET", f"/api/v1/agents/compilations/{compilation_id}/"
        )
        return Compilation(**data)

    def compile(self, *, execution_id: int) -> Compilation:
        """Create a script compilation from an execution."""
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/compile/{execution_id}/"
        )
        return Compilation(**data)

    def run(self, compilation_id: int) -> Compilation:
        """Execute a compiled script."""
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/{compilation_id}/run/"
        )
        return Compilation(**data)


# ══════════════════════════════════════════════════════════════════════════════
#  ASYNC resources
# ══════════════════════════════════════════════════════════════════════════════


class AsyncAgents:
    """Async variant of :class:`Agents`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def create(
        self,
        *,
        name: str,
        goal: str,
        tools: Optional[List[int]] = None,
        status: Optional[str] = None,
    ) -> Agent:
        body: Dict[str, Any] = {"name": name, "user_prompt": goal}
        if tools is not None:
            body["available_tools"] = tools
        if status is not None:
            body["status"] = status
        data = await self._c._request("POST", "/api/v1/agents/tasks/", json=body)
        return Agent(**data)

    async def list(self) -> List[Agent]:
        data = await self._c._request("GET", "/api/v1/agents/tasks/")
        return [Agent(**item) for item in data]

    async def get(self, agent_id: str) -> AgentDetail:
        data = await self._c._request("GET", f"/api/v1/agents/tasks/{agent_id}/")
        return AgentDetail(**data)

    async def update(self, agent_id: str, **kwargs: Any) -> Agent:
        if "goal" in kwargs:
            kwargs["user_prompt"] = kwargs.pop("goal")
        data = await self._c._request(
            "PATCH", f"/api/v1/agents/tasks/{agent_id}/", json=kwargs
        )
        return Agent(**data)

    async def delete(self, agent_id: str) -> None:
        await self._c._request("DELETE", f"/api/v1/agents/tasks/{agent_id}/")

    async def run(self, agent_id: str) -> RunDetail:
        data = await self._c._request(
            "POST", f"/api/v1/agents/tasks/{agent_id}/run-loop/"
        )
        return RunDetail(**data)


class AsyncRuns:
    """Async variant of :class:`Runs`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def create(self, *, agent_id: str) -> RunDetail:
        """Create a new run for the given agent (async)."""
        return await self._c.agents.run(agent_id)

    async def list(self) -> List[Run]:
        data = await self._c._request("GET", "/api/v1/agents/executions/")
        return [Run(**item) for item in data]

    async def get(self, run_id: int) -> RunDetail:
        data = await self._c._request(
            "GET", f"/api/v1/agents/executions/{run_id}/"
        )
        return RunDetail(**data)

    async def cancel(self, run_id: int) -> None:
        await self._c._request(
            "POST", f"/api/v1/agents/executions/{run_id}/cancel/"
        )

    async def append_message(self, run_id: int, *, text: str) -> RunDetail:
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/append-message/",
            json={"text": text},
        )
        return RunDetail(**data)

    async def wait(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        deadline = time.monotonic() + timeout
        while True:
            result = await self.get(run_id)
            if result.status in _TERMINAL_STATUSES:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Run {run_id} did not complete within {timeout}s "
                    f"(last status: {result.status})"
                )
            await asyncio.sleep(poll_interval)

    async def stream_events(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 1.0,
    ):
        seen_ids: set = set()
        deadline = time.monotonic() + timeout
        while True:
            detail = await self.get(run_id)
            for log in detail.logs:
                if log.id not in seen_ids:
                    seen_ids.add(log.id)
                    yield log
            if detail.status in _TERMINAL_STATUSES:
                return
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(poll_interval)


class AsyncTools:
    """Async variant of :class:`Tools`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def list(self) -> List[Tool]:
        data = await self._c._request("GET", "/api/v1/agents/tools/")
        return [Tool(**item) for item in data]

    async def available(self) -> List[AvailableTool]:
        data = await self._c._request("GET", "/api/v1/agents/tools/available/")
        return [AvailableTool(**item) for item in data]

    async def create(self, *, mcp_tool: str, **kwargs: Any) -> Tool:
        body = {"mcp_tool": mcp_tool, **kwargs}
        data = await self._c._request("POST", "/api/v1/agents/tools/", json=body)
        return Tool(**data)

    async def get(self, tool_id: int) -> Tool:
        data = await self._c._request("GET", f"/api/v1/agents/tools/{tool_id}/")
        return Tool(**data)

    async def update(self, tool_id: int, **kwargs: Any) -> Tool:
        data = await self._c._request(
            "PATCH", f"/api/v1/agents/tools/{tool_id}/", json=kwargs
        )
        return Tool(**data)

    async def delete(self, tool_id: int) -> None:
        await self._c._request("DELETE", f"/api/v1/agents/tools/{tool_id}/")

    async def provide_config(
        self, tool_id: int, *, user_response: Any
    ) -> Tool:
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/provide-config/",
            json={"user_response": user_response},
        )
        return Tool(**data)

    async def call(
        self,
        tool_id: int,
        *,
        action: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/call/",
            json={"action": action, "arguments": arguments or {}},
        )
        return data


class AsyncCompilations:
    """Async variant of :class:`Compilations`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def list(self) -> List[Compilation]:
        data = await self._c._request("GET", "/api/v1/agents/compilations/")
        return [Compilation(**item) for item in data]

    async def get(self, compilation_id: int) -> Compilation:
        data = await self._c._request(
            "GET", f"/api/v1/agents/compilations/{compilation_id}/"
        )
        return Compilation(**data)

    async def compile(self, *, execution_id: int) -> Compilation:
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/compile/{execution_id}/"
        )
        return Compilation(**data)

    async def run(self, compilation_id: int) -> Compilation:
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/{compilation_id}/run/"
        )
        return Compilation(**data)
