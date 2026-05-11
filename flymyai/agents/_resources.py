from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from flymyai.agents._types import (
    Agent,
    AgentDetail,
    AvailableTool,
    Compilation,
    CompilationStatus,
    ResourceID,
    Run,
    RunDetail,
    SchemaSuggestion,
    Tool,
)

if TYPE_CHECKING:
    from flymyai.agents._client import AsyncAgentClient, SyncAgentClient


_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class Agents:
    """CRUD for agents. Maps to ``/api/v1/agents/tasks/``."""

    def __init__(self, client: "SyncAgentClient") -> None:
        self._c = client

    def create(
        self,
        *,
        name: str,
        goal: str,
        tools: Optional[List[int]] = None,
        mcp_servers: Optional[List[int]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        input_description: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        output_description: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Agent:
        """Create a new agent.

        Parameters
        ----------
        name:
            Human-readable agent name.
        goal:
            The agent's prompt / instructions (stored as ``user_prompt``).
            May contain Jinja2 placeholders like ``{{ topic }}`` — provide
            matching ``input_schema`` so the backend can validate runtime
            ``variables``.
        tools:
            List of ``UserMcpTool`` IDs to attach.
        mcp_servers:
            List of custom MCP server IDs to attach.
        input_schema:
            JSON Schema describing runtime variables accepted by ``run``.
            Required whenever ``goal`` contains Jinja2 placeholders.
        input_description:
            Plain-text description of what the agent expects as input.
            Used together with ``input_schema`` at freeze time as the
            authoritative scope: chat steps outside the input→output
            canonical flow are dropped from the frozen instruction.
        output_schema:
            JSON Schema the agent must produce as its final result.
        output_description:
            Plain-text description of what the agent returns. Paired
            with ``output_schema`` to bound the canonical pipeline.
        status:
            Initial status (default ``draft``).
        """
        body: Dict[str, Any] = {"name": name, "user_prompt": goal}
        if tools is not None:
            body["available_tools"] = tools
        if mcp_servers is not None:
            body["available_custom_mcp_servers"] = mcp_servers
        if input_schema is not None:
            body["input_schema"] = input_schema
        if input_description is not None:
            body["input_description"] = input_description
        if output_schema is not None:
            body["output_schema"] = output_schema
        if output_description is not None:
            body["output_description"] = output_description
        if status is not None:
            body["status"] = status
        data = self._c._request("POST", "/api/v1/agents/tasks/", json=body)
        return Agent(**data)

    def list(self) -> List[Agent]:
        """Return all non-archived agents for the current user."""
        data = self._c._request("GET", "/api/v1/agents/tasks/")
        return [Agent(**item) for item in data]

    def get(self, agent_id: str) -> AgentDetail:
        """Get agent by UUID (returns full detail with nested tools)."""
        data = self._c._request("GET", f"/api/v1/agents/tasks/{agent_id}/")
        return AgentDetail(**data)

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

    def delete(self, agent_id: str) -> None:
        """Soft-delete (archive) an agent."""
        self._c._request("DELETE", f"/api/v1/agents/tasks/{agent_id}/")

    # -- run -------------------------------------------------------------------

    def run(
        self,
        agent_id: str,
        *,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Create an execution and start the agent loop.

        Parameters
        ----------
        agent_id:
            Agent UUID.
        variables:
            Runtime values to substitute into the ``goal`` Jinja2 template.
            Must match the agent's ``input_schema`` when one is set.

        Returns
        -------
        RunDetail
            Newly created run (status will be ``pending``/``running``).
        """
        body: Dict[str, Any] = {"variables": variables or {}}
        data = self._c._request(
            "POST", f"/api/v1/agents/tasks/{agent_id}/run-loop/", json=body
        )
        return RunDetail(**data)

    # -- freeze ----------------------------------------------------------------

    def freeze(self, run_id: int) -> Compilation:
        """Freeze a completed run into a reusable Markdown instruction.

        Distills the full chat + tool trace into an ``instruction_md`` that
        captures *what the agent does*. Re-run it later with
        :meth:`Compilations.run_instruction`.
        """
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{run_id}/"
        )
        return Compilation(**data)

    # -- suggest_schema --------------------------------------------------------

    def suggest_schema(
        self,
        *,
        user_prompt: str,
        input_hint: Optional[str] = None,
        output_hint: Optional[str] = None,
        generate_descriptions: bool = False,
    ) -> SchemaSuggestion:
        """Draft ``input_schema`` / ``output_schema`` from a prompt.

        Calls ``POST /api/v1/agents/tasks/suggest-schema/``. Server uses
        Anthropic to infer schemas from ``{{ placeholder }}`` references in
        the prompt and the optional natural-language hints. Does **not**
        persist anything — you have to PATCH the agent yourself.

        Parameters
        ----------
        user_prompt:
            The agent prompt. May contain ``{{ var }}`` placeholders.
        input_hint:
            One-sentence description of expected inputs. If empty and
            ``generate_descriptions`` is ``True``, the server drafts one.
        output_hint:
            Same for outputs.
        generate_descriptions:
            When ``True`` the response includes
            ``input_description`` / ``output_description``.
        """
        body: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "generate_descriptions": generate_descriptions,
        }
        if input_hint is not None:
            body["input_hint"] = input_hint
        if output_hint is not None:
            body["output_hint"] = output_hint
        data = self._c._request(
            "POST", "/api/v1/agents/tasks/suggest-schema/", json=body
        )
        return SchemaSuggestion(**data)

    # -- compile_from_run ------------------------------------------------------

    def compile_from_run(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Freeze a run and wait until the compilation is ready.

        Convenience wrapper: :meth:`freeze` + :meth:`Compilations.wait`.
        Raises :class:`FlyMyAIAgentError` if the compilation ends in
        ``FAILED`` state.
        """
        comp = self.freeze(run_id)
        comp = self._c.compilations.wait(
            comp.id, timeout=timeout, poll_interval=poll_interval
        )
        if comp.status == CompilationStatus.FAILED:
            from flymyai.agents._client import FlyMyAIAgentError
            raise FlyMyAIAgentError(
                f"Compilation {comp.id} failed: {comp.error or '(no error)'}",
                status_code=0,
                response_body=comp.model_dump(),
            )
        return comp


class Runs:
    """Manage agent executions (runs). Maps to ``/api/v1/agents/executions/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def create(
        self,
        *,
        agent_id: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Create a new run for the given agent.

        Convenience alias for ``client.agents.run(agent_id, variables=...)``.
        """
        return self._c.agents.run(agent_id, variables=variables)

    def list(self) -> List[Run]:
        """List all executions for the current user (newest first)."""
        data = self._c._request("GET", "/api/v1/agents/executions/")
        return [Run(**item) for item in data]

    def get(self, run_id: ResourceID) -> RunDetail:
        """Get a single execution with logs."""
        data = self._c._request("GET", f"/api/v1/agents/executions/{run_id}/")
        return RunDetail(**data)

    def cancel(self, run_id: ResourceID) -> None:
        """Cancel a running execution."""
        self._c._request("POST", f"/api/v1/agents/executions/{run_id}/cancel/")

    def append_message(self, run_id: ResourceID, *, text: str) -> RunDetail:
        """Append a user message to the conversation and restart the agent loop."""
        data = self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/append-message/",
            json={"text": text},
        )
        return RunDetail(**data)

    def suggest_schema(
        self,
        run_id: int,
        *,
        inputs_prompt: Optional[str] = None,
        outputs_prompt: Optional[str] = None,
    ) -> SchemaSuggestion:
        """Draft ``input_schema`` / ``output_schema`` from a finished run.

        Calls ``POST /api/v1/agents/executions/{id}/suggest-schema/``.
        The server uses the execution's chat history + tool trace to infer
        schemas.

        .. warning::
            **Side effect**: the server also saves the resulting schemas
            onto the agent (``user_agent_task.input_schema`` /
            ``output_schema``). Fetch the agent again with
            :meth:`Agents.get` to read them back.

        Parameters
        ----------
        inputs_prompt:
            Optional natural-language hint guiding the input schema draft.
        outputs_prompt:
            Same for outputs.
        """
        body: Dict[str, Any] = {}
        if inputs_prompt is not None:
            body["inputs_prompt"] = inputs_prompt
        if outputs_prompt is not None:
            body["outputs_prompt"] = outputs_prompt
        data = self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/suggest-schema/",
            json=body or None,
        )
        return SchemaSuggestion(**data)

    def wait(
        self,
        run_id: ResourceID,
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
        run_id: ResourceID,
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

    def provide_config(self, tool_id: int, *, user_response: Any) -> Tool:
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
    """Frozen agent instructions. Maps to ``/api/v1/agents/compilations/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def list(self) -> List[Compilation]:
        data = self._c._request("GET", "/api/v1/agents/compilations/")
        return [Compilation(**item) for item in data]

    def get(self, compilation_id: ResourceID) -> Compilation:
        data = self._c._request("GET", f"/api/v1/agents/compilations/{compilation_id}/")
        return Compilation(**data)

    def update(
        self,
        compilation_id: ResourceID,
        *,
        instruction_md: Optional[str] = None,
        cron_schedule: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Compilation:
        """Edit a frozen compilation (PATCH).

        Useful when you want to tweak the Markdown plan by hand after the
        backend froze it. Only allowed when the compilation is already in
        a terminal state (``compiled`` / ``running`` / ``completed``);
        editing during ``pending`` / ``compiling`` raises HTTP 400.

        Parameters
        ----------
        compilation_id:
            ID of the compilation to update.
        instruction_md:
            New Markdown instruction body. Pass a non-blank string.
        cron_schedule:
            New cron expression for scheduled re-runs (or ``""`` to clear).
        timezone:
            IANA timezone name for the cron schedule.
        """
        body: Dict[str, Any] = {}
        if instruction_md is not None:
            body["instruction_md"] = instruction_md
        if cron_schedule is not None:
            body["cron_schedule"] = cron_schedule
        if timezone is not None:
            body["timezone"] = timezone
        data = self._c._request(
            "PATCH",
            f"/api/v1/agents/compilations/{compilation_id}/",
            json=body,
        )
        return Compilation(**data)

    def compile(self, *, execution_id: ResourceID) -> Compilation:
        """Compile an execution into a reusable Python script.

        Note: this is the deterministic replay path (no variables). For
        parametric reuse, prefer :meth:`freeze` + :meth:`run_instruction`.
        """
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/compile/{execution_id}/"
        )
        return Compilation(**data)

    def freeze(self, *, execution_id: ResourceID) -> Compilation:
        """Freeze an execution into a reusable Markdown instruction.

        Alias for :meth:`Agents.freeze`.
        """
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{execution_id}/"
        )
        return Compilation(**data)

    def run(self, compilation_id: ResourceID) -> Compilation:
        """Re-execute a compiled script (deterministic replay, no variables)."""
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/{compilation_id}/run/"
        )
        return Compilation(**data)

    def run_instruction(
        self,
        compilation_id: int,
        *,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Run a frozen agent from its Markdown instruction.

        Spawns a fresh execution that follows the compiled instruction.
        Pass ``variables`` matching the source agent's ``input_schema``.
        Raises :class:`VariablesValidationError` on HTTP 400.
        """
        body: Dict[str, Any] = {}
        if variables:
            body["variables"] = variables
        data = self._c._request(
            "POST",
            f"/api/v1/agents/compilations/{compilation_id}/run-instruction/",
            json=body or None,
        )
        return RunDetail(**data)

    def run_instruction_and_wait(
        self,
        compilation_id: int,
        *,
        variables: Optional[Dict[str, Any]] = None,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Run an instruction and block until the resulting run finishes."""
        run = self.run_instruction(compilation_id, variables=variables)
        return self._c.runs.wait(
            run.id, timeout=timeout, poll_interval=poll_interval
        )

    def wait(
        self,
        compilation_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Poll until the compilation leaves the ``compiling`` state."""
        deadline = time.monotonic() + timeout
        while True:
            comp = self.get(compilation_id)
            if comp.status != CompilationStatus.COMPILING and comp.status != CompilationStatus.PENDING:
                return comp
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Compilation {compilation_id} still {comp.status} after {timeout}s"
                )
            time.sleep(poll_interval)


class AsyncAgents:
    """Async variant of :class:`Agents`."""

    def __init__(self, client: "AsyncAgentClient") -> None:
        self._c = client

    async def create(
        self,
        *,
        name: str,
        goal: str,
        tools: Optional[List[int]] = None,
        mcp_servers: Optional[List[int]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        input_description: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        output_description: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Agent:
        """Async variant of :meth:`Agents.create`. Same parameters."""
        body: Dict[str, Any] = {"name": name, "user_prompt": goal}
        if tools is not None:
            body["available_tools"] = tools
        if mcp_servers is not None:
            body["available_custom_mcp_servers"] = mcp_servers
        if input_schema is not None:
            body["input_schema"] = input_schema
        if input_description is not None:
            body["input_description"] = input_description
        if output_schema is not None:
            body["output_schema"] = output_schema
        if output_description is not None:
            body["output_description"] = output_description
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

    async def run(
        self,
        agent_id: str,
        *,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        body: Dict[str, Any] = {"variables": variables or {}}
        data = await self._c._request(
            "POST", f"/api/v1/agents/tasks/{agent_id}/run-loop/", json=body
        )
        return RunDetail(**data)

    async def freeze(self, run_id: int) -> Compilation:
        """Freeze a completed run into a reusable instruction."""
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{run_id}/"
        )
        return Compilation(**data)

    async def suggest_schema(
        self,
        *,
        user_prompt: str,
        input_hint: Optional[str] = None,
        output_hint: Optional[str] = None,
        generate_descriptions: bool = False,
    ) -> SchemaSuggestion:
        """Async variant of :meth:`Agents.suggest_schema`."""
        body: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "generate_descriptions": generate_descriptions,
        }
        if input_hint is not None:
            body["input_hint"] = input_hint
        if output_hint is not None:
            body["output_hint"] = output_hint
        data = await self._c._request(
            "POST", "/api/v1/agents/tasks/suggest-schema/", json=body
        )
        return SchemaSuggestion(**data)

    async def compile_from_run(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Async variant of :meth:`Agents.compile_from_run`."""
        comp = await self.freeze(run_id)
        comp = await self._c.compilations.wait(
            comp.id, timeout=timeout, poll_interval=poll_interval
        )
        if comp.status == CompilationStatus.FAILED:
            from flymyai.agents._client import FlyMyAIAgentError
            raise FlyMyAIAgentError(
                f"Compilation {comp.id} failed: {comp.error or '(no error)'}",
                status_code=0,
                response_body=comp.model_dump(),
            )
        return comp


class AsyncRuns:
    """Async variant of :class:`Runs`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def create(
        self,
        *,
        agent_id: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Create a new run for the given agent (async)."""
        return await self._c.agents.run(agent_id, variables=variables)

    async def list(self) -> List[Run]:
        data = await self._c._request("GET", "/api/v1/agents/executions/")
        return [Run(**item) for item in data]

    async def get(self, run_id: ResourceID) -> RunDetail:
        data = await self._c._request("GET", f"/api/v1/agents/executions/{run_id}/")
        return RunDetail(**data)

    async def cancel(self, run_id: ResourceID) -> None:
        await self._c._request("POST", f"/api/v1/agents/executions/{run_id}/cancel/")

    async def append_message(self, run_id: ResourceID, *, text: str) -> RunDetail:
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/append-message/",
            json={"text": text},
        )
        return RunDetail(**data)

    async def suggest_schema(
        self,
        run_id: int,
        *,
        inputs_prompt: Optional[str] = None,
        outputs_prompt: Optional[str] = None,
    ) -> SchemaSuggestion:
        """Async variant of :meth:`Runs.suggest_schema`.

        .. warning::
            Also saves the resulting schemas onto the source agent.
        """
        body: Dict[str, Any] = {}
        if inputs_prompt is not None:
            body["inputs_prompt"] = inputs_prompt
        if outputs_prompt is not None:
            body["outputs_prompt"] = outputs_prompt
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/suggest-schema/",
            json=body or None,
        )
        return SchemaSuggestion(**data)

    async def wait(
        self,
        run_id: ResourceID,
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
        run_id: ResourceID,
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

    async def provide_config(self, tool_id: int, *, user_response: Any) -> Tool:
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

    async def get(self, compilation_id: ResourceID) -> Compilation:
        data = await self._c._request(
            "GET", f"/api/v1/agents/compilations/{compilation_id}/"
        )
        return Compilation(**data)

    async def update(
        self,
        compilation_id: ResourceID,
        *,
        instruction_md: Optional[str] = None,
        cron_schedule: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Compilation:
        """Async variant of :meth:`Compilations.update`."""
        body: Dict[str, Any] = {}
        if instruction_md is not None:
            body["instruction_md"] = instruction_md
        if cron_schedule is not None:
            body["cron_schedule"] = cron_schedule
        if timezone is not None:
            body["timezone"] = timezone
        data = await self._c._request(
            "PATCH",
            f"/api/v1/agents/compilations/{compilation_id}/",
            json=body,
        )
        return Compilation(**data)

    async def compile(self, *, execution_id: ResourceID) -> Compilation:
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/compile/{execution_id}/"
        )
        return Compilation(**data)

    async def freeze(self, *, execution_id: ResourceID) -> Compilation:
        """Freeze an execution into a reusable Markdown instruction."""
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{execution_id}/"
        )
        return Compilation(**data)

    async def run(self, compilation_id: ResourceID) -> Compilation:
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/{compilation_id}/run/"
        )
        return Compilation(**data)

    async def run_instruction(
        self,
        compilation_id: int,
        *,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Run a frozen agent from its Markdown instruction.

        Raises :class:`VariablesValidationError` on HTTP 400.
        """
        body: Dict[str, Any] = {}
        if variables:
            body["variables"] = variables
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/compilations/{compilation_id}/run-instruction/",
            json=body or None,
        )
        return RunDetail(**data)

    async def run_instruction_and_wait(
        self,
        compilation_id: int,
        *,
        variables: Optional[Dict[str, Any]] = None,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Run an instruction and await the resulting run."""
        run = await self.run_instruction(compilation_id, variables=variables)
        return await self._c.runs.wait(
            run.id, timeout=timeout, poll_interval=poll_interval
        )

    async def wait(
        self,
        compilation_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Poll until the compilation leaves the ``compiling`` state."""
        deadline = time.monotonic() + timeout
        while True:
            comp = await self.get(compilation_id)
            if comp.status != CompilationStatus.COMPILING and comp.status != CompilationStatus.PENDING:
                return comp
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Compilation {compilation_id} still {comp.status} after {timeout}s"
                )
            await asyncio.sleep(poll_interval)
