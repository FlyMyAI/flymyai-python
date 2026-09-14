"""Goal read-model contracts using parsed documents, without HTTP or database I/O."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Dict, List, Tuple, cast

import pytest

import flymyai.agents as agents
from flymyai.agents import (
    AppendMessageResponse,
    AsyncAgentClient,
    ExecutionStatus,
    Run,
    RunDetail,
    RunGoalProgress,
    SyncAgentClient,
)
from flymyai.agents._resources import AsyncRuns, Runs


def _goal_payload(**overrides: object) -> Dict[str, object]:
    return {
        "schema": "flymyai.goal-progress/v1",
        "goal_id": "8436470d-2d22-49df-82df-7d90815f0cf3",
        "status": "complete",
        "review_count": 2,
        "reason": "finalized_completed",
        "terminal": True,
        "run_seq": 4,
        "revision": 9,
        **overrides,
    }


def _run_payload(**overrides: object) -> Dict[str, object]:
    return {
        "id": "goal-read-abc",
        "user_agent_task": 1,
        "previous_execution": None,
        "original_prompt": "",
        "created_at": "2026-09-08T15:55:00Z",
        "updated_at": "2026-09-08T15:55:00Z",
        "messages": [],
        "logs": [],
        "status": "completed",
        "run_seq": 4,
        "error": None,
        "agent_result": {"answer": "done"},
        **overrides,
    }


@pytest.mark.parametrize("reader", ["model", "sync", "async"])
@pytest.mark.parametrize(
    "goal_fields",
    [
        pytest.param({}, id="legacy-omission"),
        pytest.param({"goal": None}, id="null-goal"),
        pytest.param({"goal": _goal_payload()}, id="complete"),
        pytest.param(
            {"goal": _goal_payload(status="verified", reason="verified", terminal=False)},
            id="verified-is-preserved",
        ),
        pytest.param(
            {"goal": _goal_payload(status="future_status", reason="future_reason")},
            id="future-strings",
        ),
        pytest.param(
            {"goal": _goal_payload(schema="flymyai.goal-progress/v2")},
            id="future-schema-preserved",
        ),
        pytest.param({"goal": _goal_payload(status=None)}, id="null-status"),
        pytest.param(
            {"goal": _goal_payload(
                goal_id=None,
                status=None,
                review_count=None,
                reason="unavailable",
                terminal=None,
                revision=None,
            )},
            id="unavailable",
        ),
    ],
)
def test_goal_detail_parsing_matches_sync_and_async_reads(reader, goal_fields):
    payload = _run_payload(**goal_fields)
    calls: List[Tuple[str, str]] = []

    # These fixtures hand an already-parsed document to the real resource method.
    # They do not stand in for a database, HTTP transport, or backend integration.
    def read_document(method: str, path: str) -> Dict[str, object]:
        calls.append((method, path))
        return payload

    async def read_document_async(method: str, path: str) -> Dict[str, object]:
        calls.append((method, path))
        return payload

    if reader == "model":
        run = RunDetail.model_validate_json(json.dumps(payload))
    elif reader == "sync":
        client = cast(SyncAgentClient, SimpleNamespace(_request=read_document))
        run = Runs(client).get("goal-read-abc")
    else:
        client_async = cast(
            AsyncAgentClient, SimpleNamespace(_request=read_document_async)
        )
        run = asyncio.run(AsyncRuns(client_async).get("goal-read-abc"))

    assert isinstance(run, RunDetail)
    assert run.id == "goal-read-abc"
    assert run.run_seq == 4
    assert run.status == ExecutionStatus.COMPLETED
    assert run.output == {"answer": "done"}
    assert calls == (
        [] if reader == "model"
        else [("GET", "/api/v1/agents/executions/goal-read-abc/")]
    )
    assert ("goal" in run.model_dump(exclude_unset=True)) == ("goal" in payload)
    if payload.get("goal") is None:
        assert run.goal is None
    else:
        assert isinstance(run.goal, RunGoalProgress)
        assert run.goal.model_dump(by_alias=True) == payload["goal"]
        assert run.model_dump(by_alias=True)["goal"] == payload["goal"]


def test_goal_public_export_and_wire_schema_alias_round_trip():
    payload = _goal_payload()

    goal = RunGoalProgress.model_validate_json(json.dumps(payload))
    restored = RunGoalProgress.model_validate_json(goal.model_dump_json(by_alias=True))

    assert "RunGoalProgress" in agents.__all__
    assert goal.schema_version == "flymyai.goal-progress/v1"
    assert goal.model_dump(by_alias=True) == payload
    assert restored == goal


def test_goal_model_ignores_extra_future_and_private_fields():
    payload = _goal_payload()

    goal = RunGoalProgress.model_validate({
        **payload,
        "future_field": {"hint": "new"},
        "objective": "private objective",
        "review_prompt": "private review",
        "evidence": ["private evidence"],
    })

    assert goal.model_dump(by_alias=True) == payload


@pytest.mark.parametrize("goal_status", ["verified", "complete"])
@pytest.mark.parametrize(
    "status, terminal",
    [
        ("pending", False),
        ("running", False),
        ("completed", True),
        ("failed", True),
        ("cancelled", True),
    ],
)
def test_goal_does_not_recompute_the_canonical_run_terminal(status, terminal, goal_status):
    run = RunDetail.model_validate(_run_payload(
        status=status,
        goal=_goal_payload(status=goal_status, terminal=goal_status == "complete"),
    ))

    assert run.status.value == status
    assert run.is_terminal is terminal
    assert run.goal is not None
    assert run.goal.status == goal_status


@pytest.mark.parametrize("model", [Run, AppendMessageResponse])
def test_optional_detail_goal_does_not_expand_list_or_control_models(model):
    payload = _run_payload(goal={"future_unrecognized_shape": ["ignored"]})

    run = model.model_validate(payload)

    assert run.id == "goal-read-abc"
    assert run.is_terminal is True
    assert "goal" not in run.model_dump()
