"""Browser Use profile and explicit execution context SDK contracts."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from flymyai.agents import (
    BrowserUseProfile,
    BrowserUseProfileBinding,
    BrowserUseProfileStatus,
)
from flymyai.agents._resources import AsyncTools, Tools


NOW = datetime.now(tz=timezone.utc).isoformat()
CONNECTION_ID = "11111111-1111-4111-8111-111111111111"
PROFILE_ID = "22222222-2222-4222-8222-222222222222"


def _binding_payload(status: str = "bound") -> dict:
    profile = None
    if status == "bound":
        profile = {
            "profile_id": PROFILE_ID,
            "name": "Research browser",
            "cookie_domains": ["instagram.com"],
            "created_at": NOW,
            "updated_at": NOW,
            "last_used_at": None,
            "note": "Cookie domains do not prove login state.",
        }
    return {
        "connection_id": CONNECTION_ID,
        "status": status,
        "profile": profile,
        "code": None,
        "detail": None,
        "recovery_intent_id": None,
    }


def test_sync_browser_profile_methods_use_exact_connection_routes_and_bounds():
    client = MagicMock()
    client._request.side_effect = [
        _binding_payload("unbound"),
        _binding_payload(),
        _binding_payload(),
    ]
    tools = Tools(client)

    inspected = tools.get_browser_profile(17)
    created = tools.create_browser_profile(17, name="  Research browser  ")
    reconciled = tools.reconcile_browser_profile(17)

    assert inspected.status is BrowserUseProfileStatus.UNBOUND
    assert created.status is BrowserUseProfileStatus.BOUND
    assert created.profile is not None
    assert created.profile.cookie_domains == ["instagram.com"]
    assert reconciled.connection_id == CONNECTION_ID
    assert client._request.call_args_list[0].args == (
        "GET",
        "/api/v1/agents/tools/17/browser-profile/",
    )
    assert client._request.call_args_list[0].kwargs == {"timeout": 70.0}
    assert client._request.call_args_list[1].args == (
        "POST",
        "/api/v1/agents/tools/17/browser-profile/",
    )
    assert client._request.call_args_list[1].kwargs == {
        "json": {"name": "Research browser"},
        "timeout": 70.0,
    }
    assert client._request.call_args_list[2].args == (
        "POST",
        "/api/v1/agents/tools/17/browser-profile-reconcile/",
    )
    assert client._request.call_args_list[2].kwargs == {"timeout": 70.0}


@pytest.mark.asyncio
async def test_async_browser_profile_methods_match_sync_contract():
    client = MagicMock()
    client._request = AsyncMock(
        side_effect=[
            _binding_payload("unbound"),
            _binding_payload(),
            _binding_payload(),
        ]
    )
    tools = AsyncTools(client)

    inspected = await tools.get_browser_profile(17)
    created = await tools.create_browser_profile(17, name="Research browser")
    reconciled = await tools.reconcile_browser_profile(17)

    assert inspected.status is BrowserUseProfileStatus.UNBOUND
    assert created.profile is not None
    assert created.profile.profile_id == PROFILE_ID
    assert reconciled.status is BrowserUseProfileStatus.BOUND
    assert client._request.await_args_list[0].args == (
        "GET",
        "/api/v1/agents/tools/17/browser-profile/",
    )
    assert client._request.await_args_list[1].kwargs == {
        "json": {"name": "Research browser"},
        "timeout": 70.0,
    }
    assert client._request.await_args_list[2].args == (
        "POST",
        "/api/v1/agents/tools/17/browser-profile-reconcile/",
    )


@pytest.mark.parametrize("tool_id", [True, False, 0, -1, "17", 17.0])
def test_profile_methods_reject_non_positive_integer_connection_ids(tool_id):
    client = MagicMock()
    tools = Tools(client)

    with pytest.raises(ValueError, match="positive integer"):
        tools.get_browser_profile(tool_id)
    assert client._request.call_count == 0


@pytest.mark.parametrize("name", ["", "   ", "x" * 101, None, 17])
def test_profile_create_rejects_invalid_names_before_dispatch(name):
    client = MagicMock()
    tools = Tools(client)

    with pytest.raises(ValueError, match="1 to 100 characters"):
        tools.create_browser_profile(17, name=name)
    assert client._request.call_count == 0


def test_profile_models_reject_unknown_status_and_do_not_expose_cookie_values():
    binding = BrowserUseProfileBinding(**_binding_payload())

    assert isinstance(binding.profile, BrowserUseProfile)
    assert binding.profile is not None
    assert set(binding.profile.model_dump()) == {
        "profile_id",
        "name",
        "cookie_domains",
        "created_at",
        "updated_at",
        "last_used_at",
        "note",
    }
    with pytest.raises(ValidationError):
        BrowserUseProfileBinding(**_binding_payload("provisioning"))


def test_sync_tool_call_forwards_optional_execution_context_and_replay_key():
    client = MagicMock()
    client._request.return_value = {"ok": True}
    tools = Tools(client)

    assert tools.call(
        17,
        action="run_browser_task",
        arguments={"task": "Open example.com"},
        execution_id="abc-defg-hij",
        idempotency_key="browser-run-1",
    ) == {"ok": True}
    client._request.assert_called_once_with(
        "POST",
        "/api/v1/agents/tools/17/call/",
        json={
            "action": "run_browser_task",
            "arguments": {"task": "Open example.com"},
            "execution_id": "abc-defg-hij",
        },
        headers={"Idempotency-Key": "browser-run-1"},
    )


def test_sync_tool_call_preserves_legacy_body_when_execution_context_is_omitted():
    client = MagicMock()
    client._request.return_value = {"ok": True}
    tools = Tools(client)

    tools.call(
        17,
        action="list_browser_profiles",
        arguments=None,
        idempotency_key="browser-read-1",
    )

    assert client._request.call_args.kwargs["json"] == {
        "action": "list_browser_profiles",
        "arguments": {},
    }


@pytest.mark.parametrize("execution_id", ["", "   ", 17])
def test_tool_call_rejects_invalid_explicit_execution_context(execution_id):
    client = MagicMock()
    tools = Tools(client)

    with pytest.raises(ValueError, match="execution_id"):
        tools.call(
            17,
            action="run_browser_task",
            execution_id=execution_id,
            idempotency_key="browser-run-invalid",
        )
    assert client._request.call_count == 0


@pytest.mark.asyncio
async def test_async_tool_call_forwards_the_same_execution_context():
    client = MagicMock()
    client._request = AsyncMock(return_value={"ok": True})
    tools = AsyncTools(client)

    await tools.call(
        17,
        action="run_browser_task",
        arguments={"task": "Open example.com"},
        execution_id="abc-defg-hij",
        idempotency_key="browser-run-async-1",
    )

    client._request.assert_awaited_once_with(
        "POST",
        "/api/v1/agents/tools/17/call/",
        json={
            "action": "run_browser_task",
            "arguments": {"task": "Open example.com"},
            "execution_id": "abc-defg-hij",
        },
        headers={"Idempotency-Key": "browser-run-async-1"},
    )
