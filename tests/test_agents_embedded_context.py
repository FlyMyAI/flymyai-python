"""Compatibility tests for optional embedded-agent execution context."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from flymyai.agents._resources import (
    Agents,
    AsyncAgents,
    AsyncCompilations,
    AsyncDeployments,
    AsyncRuns,
    Compilations,
    Deployments,
    Runs,
    Versions,
)
from flymyai.agents._types import Compilation, CompilationStatus, RunDetail

NOW = datetime.now(tz=timezone.utc).isoformat()
AGENT_ID = "85eeaa4b-ecfa-4ef6-8d9e-80bea838aa5e"
VERSION_ID = "f8fe116e-c9d0-4df5-92c3-41d6d7302046"
DEPLOYMENT_ID = "6d3181eb-3176-44e2-84dd-548a828d0dbd"
CONNECTION_ID = "8335876a-ee78-45db-9d49-0ae148bd0158"
SECOND_CONNECTION_ID = "d7031e94-df84-4dce-ac14-b4d67fc847f6"
REQUIREMENT_ID = "7273ebaa-cd6e-4089-acb2-133f7417e876"


def _run_payload(*, status: str = "pending") -> dict:
    return {
        "id": "run-123",
        "user_agent_task": 1,
        "previous_execution": None,
        "original_prompt": "Do the work",
        "variables": {},
        "created_at": NOW,
        "updated_at": NOW,
        "messages": [],
        "status": status,
        "run_seq": 0,
        "error": None,
        "agent_result": None,
        "logs": [],
    }


def _sync_resources():
    client = MagicMock()
    client._request.return_value = _run_payload()
    client.agents = Agents(client)
    client.runs = Runs(client)
    client.compilations = Compilations(client)
    return client


def _async_resources():
    client = MagicMock()
    client._request = AsyncMock(return_value=_run_payload())
    client.agents = AsyncAgents(client)
    client.runs = AsyncRuns(client)
    client.compilations = AsyncCompilations(client)
    return client


def _version_payload() -> dict:
    return {
        "public_id": VERSION_ID,
        "agent_task": AGENT_ID,
        "source_compilation": 7,
        "version_number": 1,
        "instruction_md": "# Plan",
        "runtime_manifest": {},
        "input_schema": None,
        "output_schema": None,
        "llm_model": "",
        "effort": "",
        "created_at": NOW,
    }


def _deployment_payload(*, status: str = "draft") -> dict:
    return {
        "public_id": DEPLOYMENT_ID,
        "agent_task": AGENT_ID,
        "active_version": VERSION_ID,
        "candidate_version": None,
        "name": "Production",
        "status": status,
        "publish_mode": "embedded",
        "created_at": NOW,
        "updated_at": NOW,
    }


def test_agent_run_requires_and_forwards_idempotency_header():
    client = _sync_resources()

    client.agents.run(AGENT_ID, idempotency_key="agent-run-42")

    client._request.assert_called_once_with(
        "POST",
        f"/api/v1/agents/tasks/{AGENT_ID}/run-loop/",
        json={"variables": {}},
        headers={"Idempotency-Key": "agent-run-42"},
    )


def test_run_instruction_forwards_required_key_with_empty_body():
    client = _sync_resources()

    client.compilations.run_instruction(7, idempotency_key="frozen-run-empty-7")

    client._request.assert_called_once_with(
        "POST",
        "/api/v1/agents/compilations/7/run-instruction/",
        json=None,
        headers={"Idempotency-Key": "frozen-run-empty-7"},
    )


def test_run_instruction_forwards_embedded_context_and_idempotency_header():
    client = _sync_resources()

    client.compilations.run_instruction(
        7,
        variables={"campaign": "fall"},
        external_user_id="customer-42",
        deployment_id=DEPLOYMENT_ID,
        connections={"sender_inbox": CONNECTION_ID},
        idempotency_key="frozen-run-42",
    )

    client._request.assert_called_once_with(
        "POST",
        "/api/v1/agents/compilations/7/run-instruction/",
        json={
            "variables": {"campaign": "fall"},
            "external_user_id": "customer-42",
            "deployment_id": DEPLOYMENT_ID,
            "connections": {"sender_inbox": CONNECTION_ID},
        },
        headers={"Idempotency-Key": "frozen-run-42"},
    )


def test_run_instruction_and_wait_forwards_context_to_initial_request():
    client = _sync_resources()
    completed = _run_payload(status="completed")
    client.runs.wait = MagicMock(return_value=completed)

    client.compilations.run_instruction_and_wait(
        7,
        external_user_id="customer-42",
        deployment_id=DEPLOYMENT_ID,
        connections={},
        idempotency_key="frozen-run-42",
        timeout=15,
        poll_interval=0.1,
    )

    client._request.assert_called_once_with(
        "POST",
        "/api/v1/agents/compilations/7/run-instruction/",
        json={
            "external_user_id": "customer-42",
            "deployment_id": DEPLOYMENT_ID,
            "connections": {},
        },
        headers={"Idempotency-Key": "frozen-run-42"},
    )
    client.runs.wait.assert_called_once_with(
        "run-123",
        timeout=15,
        poll_interval=0.1,
    )


def test_async_run_instruction_and_wait_forwards_context():
    client = _async_resources()
    client.runs.wait = AsyncMock(return_value=_run_payload(status="completed"))

    asyncio.run(
        client.compilations.run_instruction_and_wait(
            7,
            variables={"campaign": "fall"},
            external_user_id="customer-42",
            deployment_id=DEPLOYMENT_ID,
            connections={"sender_inbox": CONNECTION_ID},
            idempotency_key="frozen-run-42",
            timeout=15,
            poll_interval=0.1,
        )
    )

    client._request.assert_awaited_once_with(
        "POST",
        "/api/v1/agents/compilations/7/run-instruction/",
        json={
            "variables": {"campaign": "fall"},
            "external_user_id": "customer-42",
            "deployment_id": DEPLOYMENT_ID,
            "connections": {"sender_inbox": CONNECTION_ID},
        },
        headers={"Idempotency-Key": "frozen-run-42"},
    )
    client.runs.wait.assert_awaited_once_with(
        "run-123",
        timeout=15,
        poll_interval=0.1,
    )


def test_versions_follow_paginated_backend_response():
    client = MagicMock()
    second_version = {
        **_version_payload(),
        "public_id": "4f68c6c2-182e-4cf9-a48e-3812a6541441",
        "version_number": 2,
    }
    client._request.side_effect = [
        {
            "results": [_version_payload()],
            "next": "http://internal-backend/api/v1/agents/versions/?cursor=next-page",
        },
        {"results": [second_version], "next": None},
    ]
    versions = Versions(client)

    result = versions.list(agent_id=AGENT_ID)

    assert [version.id for version in result] == [
        VERSION_ID,
        second_version["public_id"],
    ]
    assert client._request.call_args_list == [
        (
            ("GET", "/api/v1/agents/versions/"),
            {"params": {"agent_task": AGENT_ID}},
        ),
        (
            ("GET", "/api/v1/agents/versions/"),
            {
                "params": {
                    "agent_task": AGENT_ID,
                    "cursor": "next-page",
                }
            },
        ),
    ]


def test_versions_reject_repeated_pagination_cursor():
    client = MagicMock()
    client._request.return_value = {
        "results": [_version_payload()],
        "next": "http://internal-backend/api/v1/agents/versions/?cursor=repeated",
    }
    versions = Versions(client)

    with pytest.raises(RuntimeError, match="repeated a cursor"):
        versions.list(agent_id=AGENT_ID)

    assert client._request.call_count == 2
    assert client._request.call_args_list[1].kwargs["params"] == {
        "agent_task": AGENT_ID,
        "cursor": "repeated",
    }


def test_deployments_bound_cursor_pagination():
    client = MagicMock()
    page_number = 0

    def next_page(*_args, **_kwargs):
        nonlocal page_number
        page_number += 1
        return {
            "results": [],
            "next": (
                "http://internal-backend/api/v1/agents/deployments/"
                f"?cursor=page-{page_number}"
            ),
        }

    client._request.side_effect = next_page
    deployments = Deployments(client)

    with pytest.raises(RuntimeError, match="exceeded 100 pages"):
        deployments.list(agent_id=AGENT_ID, status="active")

    assert client._request.call_count == 100


def test_async_deployments_reject_repeated_pagination_cursor():
    async def scenario():
        client = MagicMock()
        client._request = AsyncMock(
            return_value={
                "results": [_deployment_payload()],
                "next": (
                    "http://internal-backend/api/v1/agents/deployments/?cursor=repeated"
                ),
            }
        )
        deployments = AsyncDeployments(client)

        with pytest.raises(RuntimeError, match="repeated a cursor"):
            await deployments.list(agent_id=AGENT_ID, status="active")

        assert client._request.await_count == 2
        assert client._request.await_args_list[1].kwargs["params"] == {
            "agent_task": AGENT_ID,
            "status": "active",
            "cursor": "repeated",
        }

    asyncio.run(scenario())


def test_deployments_publish_and_create_customer_connection_link():
    client = MagicMock()
    deployments = Deployments(client)
    client._request.side_effect = [
        _deployment_payload(status="active"),
        {
            "redirect_url": "https://provider.example/connect",
            "expires_at": NOW,
            "provider": "composio",
        },
    ]

    published = deployments.publish(
        DEPLOYMENT_ID,
        publish_mode="private",
        version_id=VERSION_ID,
    )
    session = deployments.create_connection_link(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
        slot="sender_inbox",
        alias="Sales",
    )

    assert published.status == "active"
    assert session.provider == "composio"
    assert client._request.call_args_list == [
        (
            (
                "POST",
                f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/publish/",
            ),
            {
                "json": {
                    "publish_mode": "private",
                    "candidate_version": VERSION_ID,
                }
            },
        ),
        (
            (
                "POST",
                f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/connect-session/",
            ),
            {
                "json": {
                    "external_user_id": "customer-42",
                    "slot": "sender_inbox",
                    "alias": "Sales",
                }
            },
        ),
    ]


def test_deployment_access_exposes_exact_provider_preflight():
    client = MagicMock()
    client._request.return_value = {
        "deployment": _deployment_payload(),
        "active_version": _version_payload(),
        "requirements": [{
            "public_id": REQUIREMENT_ID,
            "agent_version": VERSION_ID,
            "slot": "sender_inbox",
            "toolkit_slug": "gmail",
            "adapter_provider": "composio",
            "connection_required": True,
            "hosted_setup_supported": True,
            "cardinality": "exactly_one",
            "exact_actions": ["custom--gmail--GMAIL_SEND_EMAIL"],
            "inferred_actions": ["custom--gmail--GMAIL_SEND_EMAIL"],
            "risk_class": "non_idempotent_write",
            "policy": {},
            "created_at": NOW,
            "updated_at": NOW,
        }],
        "principals": [],
        "connections": [],
        "bindings": [],
    }
    deployments = Deployments(client)

    access = deployments.access(DEPLOYMENT_ID)

    requirement = access.requirements[0]
    assert requirement.adapter_provider == "composio"
    assert requirement.connection_required is True
    assert requirement.hosted_setup_supported is True


def test_async_deployment_connection_link():
    async def scenario():
        client = MagicMock()
        client._request = AsyncMock(
            return_value={
                "redirect_url": "https://provider.example/connect",
                "expires_at": NOW,
                "provider": "zernio",
            }
        )
        deployments = AsyncDeployments(client)
        result = await deployments.create_connection_link(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            slot="social",
        )
        assert result.provider == "zernio"
        client._request.assert_awaited_once_with(
            "POST",
            f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/connect-session/",
            json={
                "external_user_id": "customer-42",
                "slot": "social",
            },
        )

    asyncio.run(scenario())


def test_deployment_run_uses_stable_route_without_compilation_id():
    client = MagicMock()
    client._request.return_value = _run_payload()
    deployments = Deployments(client)

    result = deployments.run(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
        variables={"campaign": "fall"},
        connections={
            "sender_inbox": [CONNECTION_ID, SECOND_CONNECTION_ID],
        },
        idempotency_key="deployment-run-42",
    )

    assert result.id == "run-123"
    client._request.assert_called_once_with(
        "POST",
        f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/run/",
        json={
            "external_user_id": "customer-42",
            "variables": {"campaign": "fall"},
            "connections": {
                "sender_inbox": [CONNECTION_ID, SECOND_CONNECTION_ID],
            },
        },
        headers={"Idempotency-Key": "deployment-run-42"},
    )


def test_deployment_run_and_wait_forwards_only_deployment_context():
    client = MagicMock()
    client._request.return_value = _run_payload()
    client.runs.wait = MagicMock(
        return_value=RunDetail(**_run_payload(status="completed"))
    )
    deployments = Deployments(client)

    completed = deployments.run_and_wait(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
        idempotency_key="deployment-run-wait-42",
        timeout=15,
        poll_interval=0.1,
    )

    assert completed.status == "completed"
    client._request.assert_called_once_with(
        "POST",
        f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/run/",
        json={"external_user_id": "customer-42"},
        headers={"Idempotency-Key": "deployment-run-wait-42"},
    )
    client.runs.wait.assert_called_once_with(
        "run-123",
        timeout=15,
        poll_interval=0.1,
    )


def test_async_deployment_run_uses_stable_route():
    async def scenario():
        client = MagicMock()
        client._request = AsyncMock(return_value=_run_payload())
        deployments = AsyncDeployments(client)

        result = await deployments.run(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            variables={},
            connections={"sender_inbox": CONNECTION_ID},
            idempotency_key="deployment-run-42",
        )

        assert result.id == "run-123"
        client._request.assert_awaited_once_with(
            "POST",
            f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/run/",
            json={
                "external_user_id": "customer-42",
                "variables": {},
                "connections": {"sender_inbox": CONNECTION_ID},
            },
            headers={"Idempotency-Key": "deployment-run-42"},
        )

    asyncio.run(scenario())


def test_async_deployment_run_and_wait_uses_run_resource():
    async def scenario():
        client = MagicMock()
        client._request = AsyncMock(return_value=_run_payload())
        client.runs.wait = AsyncMock(
            return_value=RunDetail(**_run_payload(status="completed"))
        )
        deployments = AsyncDeployments(client)

        completed = await deployments.run_and_wait(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            idempotency_key="deployment-run-wait-42",
            timeout=15,
            poll_interval=0.1,
        )

        assert completed.status == "completed"
        client._request.assert_awaited_once_with(
            "POST",
            f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/run/",
            json={"external_user_id": "customer-42"},
            headers={"Idempotency-Key": "deployment-run-wait-42"},
        )
        client.runs.wait.assert_awaited_once_with(
            "run-123",
            timeout=15,
            poll_interval=0.1,
        )

    asyncio.run(scenario())


def test_compilation_is_ready_matches_backend_instruction_contract():
    base = {
        "id": 7,
        "execution": "run-123",
        "created_at": NOW,
        "updated_at": NOW,
    }

    assert Compilation(**base, status=CompilationStatus.COMPILED).is_ready is True
    assert Compilation(**base, status=CompilationStatus.COMPLETED).is_ready is True
    assert (
        Compilation(
            **base,
            status=CompilationStatus.RUNNING,
            instruction_md="# Frozen plan",
        ).is_ready
        is True
    )
    assert (
        Compilation(
            **base,
            status=CompilationStatus.FAILED,
            instruction_md="# Frozen plan",
        ).is_ready
        is True
    )
    assert Compilation(**base, status=CompilationStatus.RUNNING).is_ready is False
    assert Compilation(**base, status=CompilationStatus.FAILED).is_ready is False
    assert Compilation(**base, status=CompilationStatus.PENDING).is_ready is False
