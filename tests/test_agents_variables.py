"""Unit tests for the new variables / suggest-schema / freeze flow.

All tests use ``httpx.MockTransport`` so nothing is actually sent over the
wire. They cover only the Python client's behaviour: correct paths, request
bodies, response parsing, and error mapping.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from flymyai.agents._client import SyncAgentClient
from flymyai.agents._types import CompilationStatus, ExecutionStatus
from flymyai import (
    SchemaSuggestion,
    SuggestSchemaError,
    VariablesValidationError,
)


# ── helpers ─────────────────────────────────────────────────────────────────


Route = Callable[[httpx.Request], httpx.Response]


def _build_client(routes: Dict[tuple, Route]) -> SyncAgentClient:
    """Build a client whose HTTP layer is a MockTransport matching routes."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        route = routes.get(key)
        if route is None:
            return httpx.Response(
                status_code=404,
                json={"detail": f"No mock route for {key}"},
            )
        return route(request)

    client = SyncAgentClient(api_key="test-key", base_url="http://testserver")
    client._http = httpx.Client(
        base_url="http://testserver",
        headers={"X-API-KEY": "test-key"},
        transport=httpx.MockTransport(handler),
    )
    return client


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compilation_payload(
    *,
    comp_id: int = 1,
    execution: int = 100,
    status: str = "compiled",
    instruction_md: str = "# Instruction",
) -> Dict[str, Any]:
    return {
        "id": comp_id,
        "execution": execution,
        "status": status,
        "script_code": "",
        "instruction_md": instruction_md,
        "result": None,
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _run_payload(
    *,
    run_id: int = 100,
    status: str = "pending",
    variables: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": run_id,
        "user_agent_task": 42,
        "previous_execution": None,
        "original_prompt": "rendered",
        "variables": variables or {},
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "messages": [],
        "status": status,
        "run_seq": 0,
        "error": None,
        "agent_result": None,
        "logs": [],
    }


# ── Agents.suggest_schema ───────────────────────────────────────────────────


class TestAgentsSuggestSchema:
    def test_posts_to_tasks_suggest_schema(self):
        captured: List[Dict[str, Any]] = []

        def suggest(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "reasoning": "ok",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
            )

        client = _build_client(
            {("POST", "/api/v1/agents/tasks/suggest-schema/"): suggest}
        )

        result = client.agents.suggest_schema(
            user_prompt="Summarize {{ url }}.",
            input_hint="A URL to summarize",
            output_hint="A short summary",
        )

        assert isinstance(result, SchemaSuggestion)
        assert result.input_schema == {"type": "object"}
        assert result.output_schema == {"type": "object"}
        assert captured == [{
            "user_prompt": "Summarize {{ url }}.",
            "generate_descriptions": False,
            "input_hint": "A URL to summarize",
            "output_hint": "A short summary",
        }]

    def test_with_generate_descriptions_parses_both(self):
        def suggest(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["generate_descriptions"] is True
            return httpx.Response(
                200,
                json={
                    "reasoning": "r",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "input_description": "a url",
                    "output_description": "a summary",
                },
            )

        client = _build_client(
            {("POST", "/api/v1/agents/tasks/suggest-schema/"): suggest}
        )
        result = client.agents.suggest_schema(
            user_prompt="Summarize {{ url }}.",
            generate_descriptions=True,
        )
        assert result.input_description == "a url"
        assert result.output_description == "a summary"

    def test_maps_502_to_suggest_schema_error(self):
        def suggest(_: httpx.Request) -> httpx.Response:
            return httpx.Response(502, json={"detail": "ANTHROPIC_API_KEY missing"})

        client = _build_client(
            {("POST", "/api/v1/agents/tasks/suggest-schema/"): suggest}
        )
        with pytest.raises(SuggestSchemaError) as excinfo:
            client.agents.suggest_schema(user_prompt="x")
        assert excinfo.value.status_code == 502


# ── Runs.suggest_schema ─────────────────────────────────────────────────────


class TestRunsSuggestSchema:
    def test_posts_to_execution_suggest_schema(self):
        captured: List[Dict[str, Any]] = []

        def suggest(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "reasoning": "",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
            )

        client = _build_client(
            {("POST", "/api/v1/agents/executions/77/suggest-schema/"): suggest}
        )

        client.runs.suggest_schema(
            77,
            inputs_prompt="one URL",
            outputs_prompt="a summary",
        )
        assert captured == [{"inputs_prompt": "one URL", "outputs_prompt": "a summary"}]

    def test_empty_body_sends_no_hints(self):
        seen_body: List[Any] = []

        def suggest(request: httpx.Request) -> httpx.Response:
            seen_body.append(request.content)
            return httpx.Response(
                200,
                json={
                    "reasoning": "",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
            )

        client = _build_client(
            {("POST", "/api/v1/agents/executions/77/suggest-schema/"): suggest}
        )
        client.runs.suggest_schema(77)
        # When both hints are None, body is null (or absent)
        assert seen_body[0] in (b"null", b"")


# ── Variables validation error mapping ──────────────────────────────────────


class TestVariablesValidationError:
    def test_run_loop_400_raises_structured_error(self):
        def run_loop(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "variables": [
                        "'website_url' is a required property",
                        "'count' must be an integer",
                    ]
                },
            )

        client = _build_client({
            ("POST", "/api/v1/agents/tasks/abc-123/run-loop/"): run_loop,
        })
        with pytest.raises(VariablesValidationError) as excinfo:
            client.agents.run("abc-123", variables={})
        err = excinfo.value
        assert err.status_code == 400
        assert len(err.messages) == 2
        assert err.field_errors == {
            "website_url": "'website_url' is a required property",
            "count": "'count' must be an integer",
        }

    def test_run_instruction_400_raises_structured_error(self):
        def run_instruction(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "variables": ["This field is required when input_schema is set."]
                },
            )

        client = _build_client({
            (
                "POST",
                "/api/v1/agents/compilations/9/run-instruction/",
            ): run_instruction,
        })
        with pytest.raises(VariablesValidationError) as excinfo:
            client.compilations.run_instruction(9)
        assert excinfo.value.messages == [
            "This field is required when input_schema is set."
        ]
        # Message has no quoted field name, so field_errors is empty
        assert excinfo.value.field_errors == {}


# ── compile_from_run / run_instruction_and_wait helpers ────────────────────


class TestHighLevelHelpers:
    def test_compile_from_run_freezes_then_waits(self):
        # First GET returns "compiling", second returns "compiled"
        poll_state = {"i": 0}

        def freeze(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_compilation_payload(status="compiling"))

        def get_comp(_: httpx.Request) -> httpx.Response:
            poll_state["i"] += 1
            status = "compiled" if poll_state["i"] >= 2 else "compiling"
            return httpx.Response(200, json=_compilation_payload(status=status))

        client = _build_client({
            (
                "POST",
                "/api/v1/agents/compilations/freeze-instruction/100/",
            ): freeze,
            ("GET", "/api/v1/agents/compilations/1/"): get_comp,
        })
        comp = client.agents.compile_from_run(100, poll_interval=0.01)
        assert comp.status == CompilationStatus.COMPILED
        assert poll_state["i"] >= 2

    def test_run_instruction_and_wait_runs_then_waits(self):
        poll_state = {"i": 0}

        def run_instruction(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_run_payload(run_id=100, status="pending"))

        def get_run(_: httpx.Request) -> httpx.Response:
            poll_state["i"] += 1
            status = "completed" if poll_state["i"] >= 2 else "running"
            return httpx.Response(200, json=_run_payload(run_id=100, status=status))

        client = _build_client({
            (
                "POST",
                "/api/v1/agents/compilations/1/run-instruction/",
            ): run_instruction,
            ("GET", "/api/v1/agents/executions/100/"): get_run,
        })
        run = client.compilations.run_instruction_and_wait(
            1, variables={"x": 1}, poll_interval=0.01
        )
        assert run.status == ExecutionStatus.COMPLETED


# ── Agents.create with input/output descriptions ────────────────────────────


class TestAgentsCreateDescriptions:
    def test_create_sends_input_output_description(self):
        captured: List[Dict[str, Any]] = []

        def create(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                201,
                json={
                    "uuid": "u",
                    "name": "n",
                    "user_prompt": "p",
                    "input_description": "a URL to audit",
                    "output_description": "a numeric score",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                },
            )

        client = _build_client({("POST", "/api/v1/agents/tasks/"): create})
        agent = client.agents.create(
            name="n",
            goal="p",
            input_description="a URL to audit",
            output_description="a numeric score",
        )

        assert captured[0]["input_description"] == "a URL to audit"
        assert captured[0]["output_description"] == "a numeric score"
        assert agent.input_description == "a URL to audit"
        assert agent.output_description == "a numeric score"

    def test_create_omits_descriptions_when_not_provided(self):
        captured: List[Dict[str, Any]] = []

        def create(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                201,
                json={
                    "uuid": "u",
                    "name": "n",
                    "user_prompt": "p",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                },
            )

        client = _build_client({("POST", "/api/v1/agents/tasks/"): create})
        agent = client.agents.create(name="n", goal="p")

        assert "input_description" not in captured[0]
        assert "output_description" not in captured[0]
        # Default empty string when backend omits the field too.
        assert agent.input_description == ""
        assert agent.output_description == ""
