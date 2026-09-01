"""HTTP contract tests for exact MCP instances, sets, groups, and mappings."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from flymyai import (
    ExternalPrincipal as PublicExternalPrincipal,
    IntegrationConnection as PublicIntegrationConnection,
    McpAccessMode as PublicMcpAccessMode,
    McpResourceSetAuthorityType as PublicMcpResourceSetAuthorityType,
    McpResourceSetManagementMode as PublicMcpResourceSetManagementMode,
    McpResourceSetSummary as PublicMcpResourceSetSummary,
    RuntimeConnections as PublicRuntimeConnections,
)
from flymyai.agents._client import (
    McpResourceSetStaleRevisionError,
    SyncAgentClient,
)
from flymyai.agents._resources import (
    AgentGroups,
    Agents,
    AsyncAgentGroups,
    AsyncAgents,
    AsyncCompilations,
    AsyncDeployments,
    AsyncMcpResourceSets,
    AsyncTools,
    Compilations,
    Deployments,
    McpResourceSets,
    Tools,
    _compact_cursor_page,
    _idempotency_headers,
)
from flymyai.agents._types import (
    AgentDeploymentAccess,
    McpAccessMode,
    McpResourceSetAuthorityType,
    McpResourceSetManagementMode,
    McpResourceSetMemberInput,
    McpResourceSetSummary,
    McpResourceType,
    RuntimeConnections,
)

NOW = datetime.now(tz=timezone.utc).isoformat()
AGENT_ID = "85eeaa4b-ecfa-4ef6-8d9e-80bea838aa5e"
DEPLOYMENT_ID = "6d3181eb-3176-44e2-84dd-548a828d0dbd"
OWNER_SET_ID = "11a12f0c-c71b-447d-a8c5-3294f0bdd9b0"
CUSTOMER_SET_ID = "d4949a5c-94e1-4529-9bc7-13c95d9916bb"
GROUP_ID = "d222e023-a246-45fa-830f-2ef848231db0"
PRINCIPAL_ID = "351fcf99-440f-4138-bc6e-a28ef09304fc"
TOOL_IDS = (
    "d5f79850-2c0f-4ce9-a865-d7061bc00182",
    "1afbe6dd-382f-40e2-8019-d405b2383437",
    "fa5e1b2b-86a5-4669-b617-bcaaee3c262e",
)
CONNECTION_ID = "8335876a-ee78-45db-9d49-0ae148bd0158"
BINDING_ID = "9df8683b-b094-4e12-bc35-5880a20c9260"


def _tool_payload(index: int, alias: str) -> dict:
    return {
        "id": index,
        "public_id": TOOL_IDS[index - 1],
        "mcp_tool": "gmail",
        "alias": alias,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _member_payload(
    *,
    public_id: str,
    resource_type: str,
    resource_id: str,
    alias: str,
    position: int,
    slot: str = "mailboxes",
) -> dict:
    return {
        "public_id": public_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "toolkit_slug": "gmail",
        "alias": alias,
        "display_name": f"gmail - {alias}",
        "slot": slot,
        "allowed_actions": ["SEARCH", "SEND"],
        "position": position,
        "created_at": NOW,
        "updated_at": NOW,
        "credentials": {"refresh_token": "must-not-surface"},
    }


def _resource_set_payload(
    *,
    public_id: str = OWNER_SET_ID,
    revision: int = 1,
    authority_type: str = "owner",
    management_mode: str = "flymyai",
    principal_id=None,
    members=None,
) -> dict:
    member_rows = members or []
    return {
        "public_id": public_id,
        "name": "Mail operations",
        "description": "Exact mailboxes",
        "status": "active",
        "management_mode": management_mode,
        "authority_type": authority_type,
        "principal_id": principal_id,
        "revision": revision,
        "member_count": len(member_rows),
        "members": member_rows,
        "created_at": NOW,
        "updated_at": NOW,
        "credentials": {"access_token": "must-not-surface"},
    }


def _resource_set_summary_payload(**kwargs) -> dict:
    payload = _resource_set_payload(**kwargs)
    payload.pop("members")
    return payload


def _group_payload() -> dict:
    return {
        "public_id": GROUP_ID,
        "name": "Mail agents",
        "description": "Agents with mailbox access",
        "is_active": True,
        "agent_ids": [AGENT_ID],
        "resource_set_ids": [OWNER_SET_ID],
        "created_at": NOW,
        "updated_at": NOW,
    }


def _run_payload() -> dict:
    return {
        "id": "run-123",
        "user_agent_task": 1,
        "previous_execution": None,
        "original_prompt": "Do the work",
        "variables": {},
        "created_at": NOW,
        "updated_at": NOW,
        "messages": [],
        "status": "pending",
        "run_seq": 0,
        "error": None,
        "agent_result": None,
        "logs": [],
    }


def _deployment_payload() -> dict:
    return {
        "public_id": DEPLOYMENT_ID,
        "agent_task": AGENT_ID,
        "active_version": None,
        "candidate_version": None,
        "name": "Production",
        "status": "active",
        "publish_mode": "embedded",
        "created_at": NOW,
        "updated_at": NOW,
    }


def _principal_payload(*, external_user_id: str = "customer-42") -> dict:
    return {
        "public_id": PRINCIPAL_ID,
        "deployment": DEPLOYMENT_ID,
        "external_user_id": external_user_id,
        "display_name": "Customer 42",
        "metadata": {"plan": "pro"},
        "status": "active",
        "created_at": NOW,
        "updated_at": NOW,
    }


def _connection_payload(*, status: str = "active") -> dict:
    return {
        "public_id": CONNECTION_ID,
        "principal": PRINCIPAL_ID,
        "toolkit_slug": "gmail",
        "alias": "sender",
        "status": status,
        "provider": "composio",
        "provider_connection_id": "provider-connection",
        "provider_account_id": "provider-account",
        "granted_scopes": ["gmail.send"],
        "legacy_user_mcp_tool": None,
        "credentials_configured": status == "active",
        "credential_revision": 1,
        "metadata": {},
        "last_error": "",
        "expires_at": None,
        "revoked_at": None,
        "created_at": NOW,
        "updated_at": NOW,
    }


def _access_payload(*, connection_status: str = "active") -> dict:
    return {
        "deployment": _deployment_payload(),
        "active_version": None,
        "candidate_version": None,
        "requirements": [],
        "principals": [_principal_payload()],
        "connections": [_connection_payload(status=connection_status)],
        "bindings": [{
            "public_id": BINDING_ID,
            "principal": PRINCIPAL_ID,
            "slot": "sender_inbox",
            "connections": [CONNECTION_ID],
            "created_at": NOW,
            "updated_at": NOW,
        }],
    }


def test_tools_keep_three_exact_same_type_aliases_and_legacy_create_shape():
    assert PublicMcpAccessMode is McpAccessMode
    assert PublicMcpResourceSetAuthorityType is McpResourceSetAuthorityType
    assert PublicMcpResourceSetManagementMode is McpResourceSetManagementMode
    assert PublicMcpResourceSetSummary is McpResourceSetSummary
    assert PublicRuntimeConnections is RuntimeConnections

    client = MagicMock()
    tools = Tools(client)
    client._request.side_effect = [
        {
            "next_cursor": "tool-page-2",
            "previous_cursor": None,
            "results": [
                _tool_payload(1, "personal"),
                _tool_payload(2, "sales"),
            ],
        },
        {
            "next_cursor": None,
            "previous_cursor": "tool-page-1",
            "results": [_tool_payload(3, "support")],
        },
    ]

    connections = tools.list(page_size=2, mcp_tool="gmail")

    assert [tool.public_id for tool in connections] == list(TOOL_IDS)
    assert [tool.alias for tool in connections] == ["personal", "sales", "support"]
    assert {tool.mcp_tool for tool in connections} == {"gmail"}
    assert client._request.call_args_list[0].kwargs["params"] == {
        "mcp_tool": "gmail",
        "page_size": 2,
    }
    assert client._request.call_args_list[1].kwargs["params"] == {
        "mcp_tool": "gmail",
        "page_size": 2,
        "cursor": "tool-page-2",
    }

    client.reset_mock()
    client._request.side_effect = None
    client._request.return_value = _tool_payload(1, "personal")
    tools.create(mcp_tool="gmail", alias="personal")
    client._request.assert_called_once_with(
        "POST",
        "/api/v1/agents/tools/",
        json={"mcp_tool": "gmail", "alias": "personal"},
    )

    client.reset_mock()
    client._request.return_value = _tool_payload(1, "default")
    tools.create(mcp_tool="gmail")
    client._request.assert_called_once_with(
        "POST",
        "/api/v1/agents/tools/",
        json={"mcp_tool": "gmail"},
    )


def test_owner_resource_set_and_agent_group_use_stable_ids_atomically():
    members = [
        _member_payload(
            public_id=f"00000000-0000-4000-8000-00000000000{index}",
            resource_type="user_mcp_tool",
            resource_id=resource_id,
            alias=alias,
            position=index,
        )
        for index, (resource_id, alias) in enumerate(
            zip(TOOL_IDS, ("personal", "sales", "support"))
        )
    ]
    client = MagicMock()
    client._request.side_effect = [
        _resource_set_payload(),
        _resource_set_payload(revision=2, members=members),
        _group_payload(),
        _group_payload(),
    ]
    resource_sets = McpResourceSets(client)
    groups = AgentGroups(client)

    created = resource_sets.create(
        name="Mail operations",
        idempotency_key="owner-set-create-v1",
        description="Exact mailboxes",
    )
    replaced = resource_sets.replace_members(
        created.id,
        expected_revision=created.revision,
        members=[
            McpResourceSetMemberInput(
                resource_type=McpResourceType.USER_MCP_TOOL,
                resource_id=resource_id,
                slot="mailboxes",
                allowed_actions=["SEARCH", "SEND"],
                position=position,
            )
            for position, resource_id in enumerate(TOOL_IDS)
        ],
    )
    group = groups.create(
        name="Mail agents",
        idempotency_key="agent-group-create-v1",
        description="Agents with mailbox access",
        agent_ids=[AGENT_ID],
        resource_set_ids=[created.id],
    )
    assigned = groups.replace_assignments(
        group.id,
        agent_ids=[AGENT_ID],
        resource_set_ids=[created.id],
    )

    assert replaced.revision == 2
    assert [member.resource_id for member in replaced.members] == list(TOOL_IDS)
    assert "credentials" not in replaced.model_dump()
    assert all("credentials" not in item.model_dump() for item in replaced.members)
    assert assigned.agent_ids == [AGENT_ID]
    assert client._request.call_args_list[0].kwargs["json"] == {
        "name": "Mail operations",
        "description": "Exact mailboxes",
        "management_mode": "flymyai",
    }
    assert client._request.call_args_list[0].kwargs["headers"] == {
        "Idempotency-Key": "owner-set-create-v1"
    }
    assert client._request.call_args_list[1].kwargs["json"] == {
        "expected_revision": 1,
        "members": [
            {
                "resource_type": "user_mcp_tool",
                "resource_id": resource_id,
                "slot": "mailboxes",
                "allowed_actions": ["SEARCH", "SEND"],
                "position": position,
            }
            for position, resource_id in enumerate(TOOL_IDS)
        ],
    }
    assert client._request.call_args_list[2].kwargs["headers"] == {
        "Idempotency-Key": "agent-group-create-v1"
    }
    assert client._request.call_args_list[3].kwargs["json"] == {
        "agent_ids": [AGENT_ID],
        "resource_set_ids": [OWNER_SET_ID],
    }


def test_member_identity_includes_slot_and_keeps_action_ceilings_independent():
    client = MagicMock()
    client._request.return_value = _resource_set_payload(revision=2)
    resource_sets = McpResourceSets(client)
    read_member = McpResourceSetMemberInput(
        resource_type=McpResourceType.USER_MCP_TOOL,
        resource_id=TOOL_IDS[0],
        slot="read_mail",
        allowed_actions=["SEARCH"],
        position=0,
    )
    send_member = McpResourceSetMemberInput(
        resource_type=McpResourceType.USER_MCP_TOOL,
        resource_id=TOOL_IDS[0],
        slot="send_mail",
        allowed_actions=["SEND"],
        position=1,
    )

    resource_sets.replace_members(
        OWNER_SET_ID,
        expected_revision=1,
        members=[read_member, send_member],
    )

    assert client._request.call_args.kwargs["json"]["members"] == [
        {
            "resource_type": "user_mcp_tool",
            "resource_id": TOOL_IDS[0],
            "slot": "read_mail",
            "allowed_actions": ["SEARCH"],
            "position": 0,
        },
        {
            "resource_type": "user_mcp_tool",
            "resource_id": TOOL_IDS[0],
            "slot": "send_mail",
            "allowed_actions": ["SEND"],
            "position": 1,
        },
    ]

    client.reset_mock()
    with pytest.raises(ValueError, match="resource_type, resource_id, slot"):
        resource_sets.replace_members(
            OWNER_SET_ID,
            expected_revision=1,
            members=[read_member, read_member.model_copy(update={"position": 1})],
        )
    client._request.assert_not_called()

    with pytest.raises(ValueError, match="action ceilings"):
        resource_sets.replace_members(
            OWNER_SET_ID,
            expected_revision=1,
            members=[
                read_member,
                McpResourceSetMemberInput(
                    resource_type=McpResourceType.USER_MCP_TOOL,
                    resource_id=TOOL_IDS[1],
                    slot="read_mail",
                    allowed_actions=["SEND"],
                    position=1,
                ),
            ],
        )
    client._request.assert_not_called()


def test_principal_customer_resource_set_uses_integration_connection_ids():
    customer_member = _member_payload(
        public_id="00000000-0000-4000-8000-000000000099",
        resource_type="integration_connection",
        resource_id=CONNECTION_ID,
        alias="customer-sales",
        position=0,
    )
    client = MagicMock()
    client._request.side_effect = [
        _resource_set_payload(
            public_id=CUSTOMER_SET_ID,
            authority_type="external_principal",
            management_mode="customer",
            principal_id=PRINCIPAL_ID,
        ),
        _resource_set_payload(
            public_id=CUSTOMER_SET_ID,
            revision=2,
            authority_type="external_principal",
            management_mode="customer",
            principal_id=PRINCIPAL_ID,
            members=[customer_member],
        ),
    ]
    resource_sets = McpResourceSets(client)

    created = resource_sets.create(
        name="Mail operations",
        idempotency_key="customer-set-create-v1",
        principal_id=PRINCIPAL_ID,
        management_mode=McpResourceSetManagementMode.CUSTOMER,
    )
    replaced = resource_sets.replace_members(
        created.id,
        expected_revision=1,
        members=[{
            "resource_type": "integration_connection",
            "resource_id": CONNECTION_ID,
            "slot": "mailboxes",
            "allowed_actions": ["SEARCH", "SEND"],
            "position": 0,
        }],
    )

    assert created.principal_id == PRINCIPAL_ID
    assert replaced.members[0].resource_id == CONNECTION_ID
    assert client._request.call_args_list[0].kwargs["json"] == {
        "name": "Mail operations",
        "management_mode": "customer",
        "principal_id": PRINCIPAL_ID,
    }


def test_deployment_access_types_exact_principal_and_ready_slot_fail_closed():
    payload = _access_payload()
    payload["connections"][0]["credentials"] = {"token": "must-not-surface"}
    access = AgentDeploymentAccess(**payload)

    principal = access.principal_for_external_user("customer-42")
    assert isinstance(principal, PublicExternalPrincipal)
    assert principal.id == PRINCIPAL_ID
    assert isinstance(access.connections[0], PublicIntegrationConnection)
    assert "credentials" not in access.connections[0].model_dump()
    assert access.ready_connection_ids_for_slot(
        principal_id=principal.id,
        slot="sender_inbox",
    ) == [CONNECTION_ID]

    with pytest.raises(ValueError, match="exactly one external principal"):
        access.principal_for_external_user("missing-customer")
    pending = AgentDeploymentAccess(**_access_payload(connection_status="pending"))
    with pytest.raises(ValueError, match="not ready"):
        pending.ready_connection_ids_for_slot(
            principal_id=PRINCIPAL_ID,
            slot="sender_inbox",
        )
    duplicated = AgentDeploymentAccess(**{
        **_access_payload(),
        "principals": [_principal_payload(), _principal_payload()],
    })
    with pytest.raises(ValueError, match="found 2"):
        duplicated.principal_for_external_user("customer-42")


def test_exact_customer_mapping_lifecycle_uses_typed_ids_and_revision():
    customer_member = _member_payload(
        public_id="00000000-0000-4000-8000-000000000099",
        resource_type="integration_connection",
        resource_id=CONNECTION_ID,
        alias="sender",
        position=0,
        slot="sender_inbox",
    )
    client = MagicMock()
    client._request.side_effect = [
        {
            "redirect_url": "https://provider.example/connect",
            "expires_at": NOW,
            "provider": "composio",
        },
        _access_payload(),
        _resource_set_payload(
            public_id=CUSTOMER_SET_ID,
            authority_type="external_principal",
            management_mode="customer",
            principal_id=PRINCIPAL_ID,
        ),
        _resource_set_payload(
            public_id=CUSTOMER_SET_ID,
            revision=2,
            authority_type="external_principal",
            management_mode="customer",
            principal_id=PRINCIPAL_ID,
            members=[customer_member],
        ),
        _run_payload(),
    ]
    deployments = Deployments(client)
    resource_sets = McpResourceSets(client)

    session = deployments.create_connection_link(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
        slot="sender_inbox",
    )
    assert session.redirect_url == "https://provider.example/connect"
    access = deployments.access(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
    )
    principal = access.principal_for_external_user("customer-42")
    connection_ids = access.ready_connection_ids_for_slot(
        principal_id=principal.id,
        slot="sender_inbox",
    )
    mapping = resource_sets.create(
        name="Customer 42 mail",
        idempotency_key="customer-42-mapping-create-v1",
        management_mode=McpResourceSetManagementMode.CUSTOMER,
        principal_id=principal.id,
    )
    mapping = resource_sets.replace_members(
        mapping.id,
        expected_revision=mapping.revision,
        members=[
            McpResourceSetMemberInput(
                resource_type=McpResourceType.INTEGRATION_CONNECTION,
                resource_id=connection_ids[0],
                slot="sender_inbox",
                allowed_actions=["GMAIL_SEND_EMAIL"],
            )
        ],
    )
    deployments.run(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
        resource_set_id=mapping.id,
        resource_set_revision=mapping.revision,
        idempotency_key="customer-42-send-v1",
    )

    assert client._request.call_args_list[1].kwargs["params"] == {
        "external_user_id": "customer-42"
    }
    assert client._request.call_args_list[2].kwargs["json"]["principal_id"] == (
        PRINCIPAL_ID
    )
    assert client._request.call_args_list[3].kwargs["json"] == {
        "expected_revision": 1,
        "members": [{
            "resource_type": "integration_connection",
            "resource_id": CONNECTION_ID,
            "slot": "sender_inbox",
            "allowed_actions": ["GMAIL_SEND_EMAIL"],
        }],
    }
    assert client._request.call_args_list[4].kwargs == {
        "json": {
            "external_user_id": "customer-42",
            "resource_set_id": CUSTOMER_SET_ID,
            "resource_set_revision": 2,
        },
        "headers": {"Idempotency-Key": "customer-42-send-v1"},
    }


def test_resource_set_crud_and_stale_revision_http_error():
    client = MagicMock()
    client._request.side_effect = [
        [_resource_set_summary_payload()],
        _resource_set_payload(),
        _resource_set_payload(revision=2),
        _resource_set_payload(revision=3),
        None,
    ]
    resource_sets = McpResourceSets(client)

    assert resource_sets.list()[0].id == OWNER_SET_ID
    assert resource_sets.get(OWNER_SET_ID).id == OWNER_SET_ID
    resource_sets.update(OWNER_SET_ID, expected_revision=1, status="active")
    resource_sets.replace(
        OWNER_SET_ID,
        expected_revision=2,
        name="Mail operations",
        description="Exact mailboxes",
        status="active",
    )
    resource_sets.delete(OWNER_SET_ID)

    assert client._request.call_args_list[2].args[:2] == (
        "PATCH",
        f"/api/v1/agents/mcp-resource-sets/{OWNER_SET_ID}/",
    )
    assert client._request.call_args_list[2].kwargs["json"] == {
        "expected_revision": 1,
        "status": "active",
    }
    assert client._request.call_args_list[3].args[:2] == (
        "PUT",
        f"/api/v1/agents/mcp-resource-sets/{OWNER_SET_ID}/",
    )
    assert client._request.call_args_list[3].kwargs["json"] == {
        "expected_revision": 2,
        "name": "Mail operations",
        "description": "Exact mailboxes",
        "status": "active",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(
            f"/mcp-resource-sets/{OWNER_SET_ID}/replace-members/"
        )
        return httpx.Response(
            409,
            json={
                "code": "stale_revision",
                "detail": "The MCP resource set changed. Reload and retry.",
                "current_revision": 4,
            },
        )

    http_client = SyncAgentClient(api_key="test", base_url="https://example.test")
    http_client._http.close()
    http_client._http = httpx.Client(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(McpResourceSetStaleRevisionError) as exc_info:
            http_client.mcp_resource_sets.replace_members(
                OWNER_SET_ID,
                expected_revision=3,
                members=[],
            )
    finally:
        http_client.close()

    assert exc_info.value.status_code == 409
    assert exc_info.value.current_revision == 4
    assert exc_info.value.response_body["code"] == "stale_revision"


def test_resource_set_authority_and_revision_inputs_fail_before_dispatch():
    client = MagicMock()
    resource_sets = McpResourceSets(client)

    with pytest.raises(ValueError, match="principal_id requires"):
        resource_sets.create(
            name="Wrong owner",
            idempotency_key="wrong-owner-create-v1",
            principal_id=PRINCIPAL_ID,
        )
    with pytest.raises(ValueError, match="requires the exact principal_id"):
        resource_sets.create(
            name="Missing principal",
            idempotency_key="missing-principal-create-v1",
            management_mode=McpResourceSetManagementMode.CUSTOMER,
        )
    with pytest.raises(ValueError, match="management_mode must be"):
        resource_sets.create(
            name="Unknown mode",
            idempotency_key="unknown-mode-create-v1",
            management_mode="ambient",
        )
    strict_create = cast(Any, resource_sets.create)
    with pytest.raises(TypeError, match="external_principal_id"):
        strict_create(
            name="Deprecated authority",
            idempotency_key="deprecated-authority-create-v1",
            external_principal_id=PRINCIPAL_ID,
        )
    strict_update = cast(Any, resource_sets.update)
    with pytest.raises(TypeError, match="management_mode"):
        strict_update(
            OWNER_SET_ID,
            expected_revision=1,
            management_mode="customer",
        )
    with pytest.raises(ValueError, match="integer of at least 1"):
        resource_sets.update(OWNER_SET_ID, expected_revision=0, name="No write")
    with pytest.raises(ValueError, match="integer of at least 1"):
        resource_sets.replace_members(
            OWNER_SET_ID,
            expected_revision=True,
            members=[],
        )
    with pytest.raises(ValueError, match="authority_type"):
        resource_sets.list(authority_type="ambient")
    with pytest.raises(ValueError, match="cannot be combined"):
        resource_sets.list(
            authority_type=McpResourceSetAuthorityType.OWNER,
            principal_id=PRINCIPAL_ID,
        )
    with pytest.raises(ValueError, match="query"):
        resource_sets.list(query="mail\nbox")
    with pytest.raises(ValueError, match="principal_id"):
        resource_sets.list(principal_id="   ")

    client._request.assert_not_called()


def test_agent_group_crud_routes_keep_public_ids():
    disabled_group = {**_group_payload(), "is_active": False}
    client = MagicMock()
    client._request.side_effect = [
        [_group_payload()],
        _group_payload(),
        disabled_group,
        None,
    ]
    groups = AgentGroups(client)

    assert groups.list()[0].id == GROUP_ID
    assert groups.get(GROUP_ID).id == GROUP_ID
    assert groups.update(GROUP_ID, is_active=False).is_active is False
    groups.delete(GROUP_ID)

    assert client._request.call_args_list[-1].args == (
        "DELETE",
        f"/api/v1/agents/agent-groups/{GROUP_ID}/",
    )


def test_resource_set_and_group_lists_follow_compact_cursor_pages():
    second_set_id = "22a12f0c-c71b-447d-a8c5-3294f0bdd9b0"
    second_group_id = "e333e023-a246-45fa-830f-2ef848231db0"
    client = MagicMock()
    client._request.side_effect = [
        {
            "next_cursor": "sets-next",
            "previous_cursor": None,
            "results": [_resource_set_summary_payload()],
        },
        {
            "next_cursor": None,
            "previous_cursor": "sets-previous",
            "results": [_resource_set_summary_payload(public_id=second_set_id)],
        },
        {
            "next_cursor": "groups-next",
            "previous_cursor": None,
            "results": [_group_payload()],
        },
        {
            "next_cursor": None,
            "previous_cursor": "groups-previous",
            "results": [{**_group_payload(), "public_id": second_group_id}],
        },
    ]

    resource_sets = McpResourceSets(client).list(
        page_size=1,
        query="  customer mail  ",
        authority_type=McpResourceSetAuthorityType.EXTERNAL_PRINCIPAL,
        principal_id=PRINCIPAL_ID,
    )
    groups = AgentGroups(client).list(page_size=1, query="  mail agents  ")

    assert [item.id for item in resource_sets] == [OWNER_SET_ID, second_set_id]
    assert resource_sets[0].member_count == 0
    assert not hasattr(resource_sets[0], "members")
    assert [item.id for item in groups] == [GROUP_ID, second_group_id]
    resource_set_filters = {
        "query": "customer mail",
        "authority_type": "external_principal",
        "principal_id": PRINCIPAL_ID,
    }
    assert client._request.call_args_list[0].kwargs["params"] == {
        **resource_set_filters,
        "page_size": 1,
    }
    assert client._request.call_args_list[1].kwargs["params"] == {
        **resource_set_filters,
        "page_size": 1,
        "cursor": "sets-next",
    }
    assert client._request.call_args_list[2].kwargs["params"] == {
        "query": "mail agents",
        "page_size": 1,
    }
    assert client._request.call_args_list[3].kwargs["params"] == {
        "query": "mail agents",
        "page_size": 1,
        "cursor": "groups-next",
    }


def test_compact_cursor_lists_reject_malformed_repeated_and_unbounded_responses():
    client = MagicMock()
    resource_sets = McpResourceSets(client)

    client._request.return_value = {"next_cursor": None, "results": []}
    with pytest.raises(RuntimeError, match="missing: previous_cursor"):
        resource_sets.list()

    with pytest.raises(RuntimeError, match="unexpected fields: count"):
        _compact_cursor_page(
            {
                "next_cursor": None,
                "previous_cursor": None,
                "results": [],
                "count": 0,
            },
            resource_name="MCP resource sets",
        )

    client._request.reset_mock()
    repeated_page = {
        "next_cursor": "same-cursor",
        "previous_cursor": None,
        "results": [],
    }
    client._request.side_effect = [repeated_page, repeated_page]
    with pytest.raises(RuntimeError, match="repeated a cursor"):
        resource_sets.list()
    assert client._request.call_count == 2

    client._request.reset_mock()
    client._request.side_effect = None
    client._request.return_value = {
        "next_cursor": "   ",
        "previous_cursor": None,
        "results": [],
    }
    with pytest.raises(RuntimeError, match="next_cursor"):
        resource_sets.list()

    client._request.reset_mock()
    client._request.side_effect = None
    client._request.return_value = [{}] * 101
    with pytest.raises(RuntimeError, match="legacy response exceeded 100 rows"):
        resource_sets.list()

    client._request.reset_mock()
    client._request.return_value = {
        "next_cursor": None,
        "previous_cursor": None,
        "results": [{}] * 101,
    }
    with pytest.raises(RuntimeError, match="cursor page exceeded 100 rows"):
        resource_sets.list()

    for cursor in (
        "line\nbreak",
        "zero\u200bwidth",
        "non\u00a0breaking",
    ):
        with pytest.raises(RuntimeError, match="printable"):
            _compact_cursor_page(
                {
                    "next_cursor": cursor,
                    "previous_cursor": None,
                    "results": [],
                },
                resource_name="MCP resource sets",
            )

    page, cursor = _compact_cursor_page(
        {
            "next_cursor": "😀" * 1_024,
            "previous_cursor": None,
            "results": [{}] * 100,
        },
        resource_name="MCP resource sets",
    )
    assert len(page) == 100
    assert cursor == "😀" * 1_024
    with pytest.raises(RuntimeError, match="at most 1024"):
        _compact_cursor_page(
            {
                "next_cursor": "😀" * 1_025,
                "previous_cursor": None,
                "results": [],
            },
            resource_name="MCP resource sets",
        )

    client._request.reset_mock()
    with pytest.raises(ValueError, match="page_size"):
        resource_sets.list(page_size=True)
    client._request.assert_not_called()


def test_resource_set_members_have_a_separate_bounded_cursor_collection():
    first = _member_payload(
        public_id="00000000-0000-4000-8000-000000000041",
        resource_type="user_mcp_tool",
        resource_id=TOOL_IDS[0],
        alias="personal",
        position=0,
    )
    second = _member_payload(
        public_id="00000000-0000-4000-8000-000000000042",
        resource_type="user_mcp_tool",
        resource_id=TOOL_IDS[1],
        alias="sales",
        position=1,
    )
    client = MagicMock()
    client._request.side_effect = [
        {
            "next_cursor": "member-page-2",
            "previous_cursor": None,
            "results": [first],
        },
        {
            "next_cursor": None,
            "previous_cursor": "member-page-1",
            "results": [second],
        },
    ]

    members = McpResourceSets(client).list_members(OWNER_SET_ID, page_size=1)

    assert [member.resource_id for member in members] == [TOOL_IDS[0], TOOL_IDS[1]]
    assert client._request.call_args_list[0].args == (
        "GET",
        f"/api/v1/agents/mcp-resource-sets/{OWNER_SET_ID}/members/",
    )
    assert client._request.call_args_list[1].kwargs["params"] == {
        "page_size": 1,
        "cursor": "member-page-2",
    }


def test_agent_access_mode_reads_writes_and_preserves_legacy_default():
    client = MagicMock()
    client._request.return_value = {
        "uuid": AGENT_ID,
        "name": "Mail analyst",
        "user_prompt": "Compare the mailboxes",
        "mcp_resource_set_ids": [OWNER_SET_ID],
        "mcp_access_mode": "scoped",
        "created_at": NOW,
        "updated_at": NOW,
    }
    agents = Agents(client)

    agent = agents.create(
        name="Mail analyst",
        goal="Compare the mailboxes",
        mcp_resource_set_ids=[OWNER_SET_ID],
        mcp_access_mode=McpAccessMode.SCOPED,
    )

    assert agent.mcp_resource_set_ids == [OWNER_SET_ID]
    assert agent.mcp_access_mode is McpAccessMode.SCOPED
    client._request.assert_called_once_with(
        "POST",
        "/api/v1/agents/tasks/",
        json={
            "name": "Mail analyst",
            "user_prompt": "Compare the mailboxes",
            "mcp_resource_set_ids": [OWNER_SET_ID],
            "mcp_access_mode": "scoped",
        },
    )

    client.reset_mock()
    client._request.return_value = {
        "uuid": AGENT_ID,
        "name": "Mail analyst",
        "user_prompt": "Compare the mailboxes",
        "created_at": NOW,
        "updated_at": NOW,
    }
    legacy = agents.update(AGENT_ID, name="Mail analyst")
    assert legacy.mcp_access_mode is McpAccessMode.LEGACY
    assert client._request.call_args.kwargs["json"] == {"name": "Mail analyst"}

    client.reset_mock()
    with pytest.raises(ValueError, match="legacy.*scoped"):
        agents.update(AGENT_ID, mcp_access_mode="ambient")
    client._request.assert_not_called()


def test_sync_agent_list_follows_bounded_cursor_pages():
    first_agent_id = "11111111-1111-4111-8111-111111111111"
    second_agent_id = "22222222-2222-4222-8222-222222222222"

    def payload(agent_id: str) -> Dict[str, Any]:
        return {
            "uuid": agent_id,
            "name": f"Agent {agent_id[0]}",
            "user_prompt": "Do bounded work",
            "created_at": NOW,
            "updated_at": NOW,
        }

    client = MagicMock()
    client._request.side_effect = [
        {
            "next_cursor": "agent-page-2",
            "previous_cursor": None,
            "results": [payload(first_agent_id)],
        },
        {
            "next_cursor": None,
            "previous_cursor": "agent-page-1",
            "results": [payload(second_agent_id)],
        },
    ]

    agents = Agents(client).list(page_size=1)

    assert [agent.id for agent in agents] == [first_agent_id, second_agent_id]
    assert client._request.call_args_list[0].kwargs["params"] == {
        "pagination": "cursor",
        "page_size": 1,
    }
    assert client._request.call_args_list[1].kwargs["params"] == {
        "pagination": "cursor",
        "page_size": 1,
        "cursor": "agent-page-2",
    }


def test_async_agent_list_follows_bounded_cursor_pages():
    async def scenario() -> None:
        first_agent_id = "33333333-3333-4333-8333-333333333333"
        second_agent_id = "44444444-4444-4444-8444-444444444444"

        def payload(agent_id: str) -> Dict[str, Any]:
            return {
                "uuid": agent_id,
                "name": f"Agent {agent_id[0]}",
                "user_prompt": "Do bounded work",
                "created_at": NOW,
                "updated_at": NOW,
            }

        client = MagicMock()
        client._request = AsyncMock(
            side_effect=[
                {
                    "next_cursor": "agent-page-2",
                    "previous_cursor": None,
                    "results": [payload(first_agent_id)],
                },
                {
                    "next_cursor": None,
                    "previous_cursor": "agent-page-1",
                    "results": [payload(second_agent_id)],
                },
            ]
        )

        agents = await AsyncAgents(client).list(page_size=1)

        assert [agent.id for agent in agents] == [first_agent_id, second_agent_id]
        assert client._request.await_args_list[0].kwargs["params"] == {
            "pagination": "cursor",
            "page_size": 1,
        }
        assert client._request.await_args_list[1].kwargs["params"] == {
            "pagination": "cursor",
            "page_size": 1,
            "cursor": "agent-page-2",
        }

    asyncio.run(scenario())


def test_scoped_to_legacy_clears_all_grants_before_separate_mode_patch():
    group_without_agent = {**_group_payload(), "agent_ids": []}
    scoped_without_direct_grants = {
        "uuid": AGENT_ID,
        "name": "Mail analyst",
        "user_prompt": "Compare the mailboxes",
        "mcp_resource_set_ids": [],
        "mcp_access_mode": "scoped",
        "created_at": NOW,
        "updated_at": NOW,
    }
    legacy_agent = {**scoped_without_direct_grants, "mcp_access_mode": "legacy"}
    client = MagicMock()
    client._request.side_effect = [
        {
            "next_cursor": None,
            "previous_cursor": None,
            "results": [_group_payload()],
        },
        group_without_agent,
        scoped_without_direct_grants,
        legacy_agent,
    ]
    groups = AgentGroups(client)
    agents = Agents(client)

    for group in groups.list():
        if AGENT_ID in group.agent_ids:
            groups.replace_assignments(
                group.id,
                agent_ids=[item for item in group.agent_ids if item != AGENT_ID],
                resource_set_ids=group.resource_set_ids,
            )
    cleared = agents.update(AGENT_ID, mcp_resource_set_ids=[])
    legacy = agents.update(AGENT_ID, mcp_access_mode=McpAccessMode.LEGACY)

    assert cleared.mcp_access_mode is McpAccessMode.SCOPED
    assert legacy.mcp_access_mode is McpAccessMode.LEGACY
    assert client._request.call_args_list[1].kwargs["json"] == {
        "agent_ids": [],
        "resource_set_ids": [OWNER_SET_ID],
    }
    assert client._request.call_args_list[2].kwargs["json"] == {
        "mcp_resource_set_ids": []
    }
    assert client._request.call_args_list[3].kwargs["json"] == {
        "mcp_access_mode": "legacy"
    }
    assert all(
        call.kwargs.get("json")
        != {"mcp_resource_set_ids": [], "mcp_access_mode": "legacy"}
        for call in client._request.call_args_list
    )


@pytest.mark.parametrize(
    "invalid_alias",
    ["", "mail box", "mail.box", "x" * 65],
)
def test_alias_and_slot_slugs_are_rejected_before_dispatch(invalid_alias):
    client = MagicMock()
    tools = Tools(client)
    deployments = Deployments(client)

    with pytest.raises(ValueError, match="alias"):
        tools.create(mcp_tool="gmail", alias=invalid_alias)
    with pytest.raises(ValueError, match="alias"):
        tools.update(1, alias=invalid_alias)
    with pytest.raises(ValueError, match="slot"):
        deployments.create_connection_link(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            slot="sales inbox",
        )
    with pytest.raises(ValueError, match="slot"):
        McpResourceSetMemberInput(
            resource_type=McpResourceType.USER_MCP_TOOL,
            resource_id=TOOL_IDS[0],
            slot="sales/inbox",
        )

    client._request.assert_not_called()


def test_deployment_and_instruction_runs_select_one_mapping_mode():
    client = MagicMock()
    client._request.return_value = _run_payload()
    deployments = Deployments(client)

    deployments.run(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
        idempotency_key="deployment-named-mapping-42",
        resource_set_id=CUSTOMER_SET_ID,
        resource_set_revision=2,
    )

    client._request.assert_called_once_with(
        "POST",
        f"/api/v1/agents/deployments/{DEPLOYMENT_ID}/run/",
        json={
            "external_user_id": "customer-42",
            "resource_set_id": CUSTOMER_SET_ID,
            "resource_set_revision": 2,
        },
        headers={"Idempotency-Key": "deployment-named-mapping-42"},
    )

    client.reset_mock()
    with pytest.raises(ValueError, match="either resource_set_id or connections"):
        deployments.run(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            idempotency_key="invalid-dual-mapping-42",
            connections={"mailboxes": CONNECTION_ID},
            resource_set_id=CUSTOMER_SET_ID,
        )
    client._request.assert_not_called()


def test_sync_run_methods_require_validate_and_forward_caller_keys():
    client = MagicMock()
    client._request.return_value = _run_payload()
    agents = Agents(client)
    compilations = Compilations(client)
    deployments = Deployments(client)
    tools = Tools(client)
    exact_key = "K" * 255

    agents.run(AGENT_ID, idempotency_key=exact_key)
    assert client._request.call_args.kwargs["headers"] == {"Idempotency-Key": exact_key}
    client.reset_mock()
    compilations.run_instruction(7, idempotency_key=" frozen-caller-key ")
    assert client._request.call_args.kwargs["headers"] == {
        "Idempotency-Key": " frozen-caller-key "
    }
    client.reset_mock()
    deployments.run(
        DEPLOYMENT_ID,
        external_user_id="customer-42",
        idempotency_key="deployment-caller-key",
    )
    assert client._request.call_args.kwargs["headers"] == {
        "Idempotency-Key": "deployment-caller-key"
    }
    client.reset_mock()
    tools.call(
        7,
        action="GMAIL_SEND_EMAIL",
        arguments={"to": "customer@example.com"},
        idempotency_key="direct-tool-caller-key",
    )
    assert client._request.call_args.kwargs["headers"] == {
        "Idempotency-Key": "direct-tool-caller-key"
    }

    for operation in (
        lambda key: agents.run(AGENT_ID, idempotency_key=key),
        lambda key: compilations.run_instruction(7, idempotency_key=key),
        lambda key: deployments.run(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            idempotency_key=key,
        ),
        lambda key: tools.call(
            7,
            action="GMAIL_SEND_EMAIL",
            idempotency_key=key,
        ),
    ):
        for invalid_key in ("", "   ", "x" * 256, "line\nbreak", "nul\x00key"):
            client.reset_mock()
            with pytest.raises(ValueError, match="idempotency_key"):
                operation(invalid_key)
            client._request.assert_not_called()

    with pytest.raises(TypeError, match="idempotency_key"):
        cast(Any, agents.run)(AGENT_ID)
    with pytest.raises(TypeError, match="idempotency_key"):
        cast(Any, compilations.run_instruction)(7)
    with pytest.raises(TypeError, match="idempotency_key"):
        cast(Any, deployments.run)(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
        )
    with pytest.raises(TypeError, match="idempotency_key"):
        cast(Any, tools.call)(7, action="GMAIL_SEND_EMAIL")


def test_sync_resource_scope_creates_require_validate_and_forward_caller_keys():
    client = MagicMock()
    resource_sets = McpResourceSets(client)
    groups = AgentGroups(client)

    client._request.return_value = _resource_set_payload()
    resource_sets.create(
        name="Mail operations",
        idempotency_key="resource-set-create-v1",
    )
    assert client._request.call_args.kwargs["headers"] == {
        "Idempotency-Key": "resource-set-create-v1"
    }

    client.reset_mock()
    client._request.return_value = _group_payload()
    groups.create(
        name="Mail agents",
        idempotency_key="agent-group-create-v1",
    )
    assert client._request.call_args.kwargs["headers"] == {
        "Idempotency-Key": "agent-group-create-v1"
    }

    for operation in (
        lambda key: resource_sets.create(
            name="Mail operations",
            idempotency_key=key,
        ),
        lambda key: groups.create(
            name="Mail agents",
            idempotency_key=key,
        ),
    ):
        for invalid_key in ("", "   ", "x" * 256, "line\nbreak", "zero\u200bwidth"):
            client.reset_mock()
            with pytest.raises(ValueError, match="idempotency_key"):
                operation(invalid_key)
            client._request.assert_not_called()

    with pytest.raises(TypeError, match="idempotency_key"):
        cast(Any, resource_sets.create)(name="Mail operations")
    with pytest.raises(TypeError, match="idempotency_key"):
        cast(Any, groups.create)(name="Mail agents")


def test_operation_key_shared_conformance_fixture_and_astral_boundary():
    fixture_path = (
        Path(__file__).parent / "fixtures" / "operation-key-conformance-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def conformance_value(entry: dict) -> str:
        value = entry.get("value")
        if isinstance(value, str):
            return value
        repeat = entry["repeat"]
        return str(repeat["value"]) * int(repeat["count"])

    for entry in fixture["accepted"]:
        value = conformance_value(entry)
        assert _idempotency_headers(value) == {"Idempotency-Key": value}, entry["name"]
    for entry in fixture["rejected"]:
        with pytest.raises(ValueError, match="idempotency_key"):
            _idempotency_headers(conformance_value(entry))

    astral_boundary = conformance_value(
        next(
            entry
            for entry in fixture["accepted"]
            if entry["name"] == "astral-255-code-points"
        )
    )
    assert len(astral_boundary) == 255


def test_deprecated_compilation_run_fails_locally_with_migration_guidance():
    client = MagicMock()
    compilations = Compilations(client)

    with pytest.raises(NotImplementedError) as exc_info:
        compilations.run(7)

    message = str(exc_info.value)
    assert "run_instruction" in message
    assert "Deployments.run" in message
    assert "idempotency_key" in message
    client._request.assert_not_called()


def test_runtime_customer_pair_revision_and_connections_fail_before_dispatch():
    client = MagicMock()
    compilations = Compilations(client)
    deployments = Deployments(client)

    with pytest.raises(ValueError, match="must be supplied together"):
        compilations.run_instruction(
            7,
            idempotency_key="external-only",
            external_user_id="customer-42",
        )
    with pytest.raises(ValueError, match="must be supplied together"):
        compilations.run_instruction(
            7,
            idempotency_key="deployment-only",
            deployment_id=DEPLOYMENT_ID,
        )
    with pytest.raises(ValueError, match="integer of at least 1"):
        deployments.run(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            idempotency_key="bool-revision",
            resource_set_id=CUSTOMER_SET_ID,
            resource_set_revision=True,
        )

    invalid_connections = (
        {"mail boxes": CONNECTION_ID},
        {"mailboxes": "   "},
        {"mailboxes": [CONNECTION_ID] * 26},
        {"mailboxes": [CONNECTION_ID, CONNECTION_ID]},
        {"mailboxes": cast(Any, 7)},
    )
    for index, connections in enumerate(invalid_connections):
        with pytest.raises(ValueError, match="connections|slot"):
            deployments.run(
                DEPLOYMENT_ID,
                external_user_id="customer-42",
                idempotency_key=f"invalid-connections-{index}",
                connections=connections,
            )
    client._request.assert_not_called()

    compilations = Compilations(client)
    with pytest.raises(ValueError, match="requires resource_set_id"):
        compilations.run_instruction(
            7,
            idempotency_key="invalid-revision-without-set-42",
            external_user_id="customer-42",
            deployment_id=DEPLOYMENT_ID,
            resource_set_revision=2,
        )
    client._request.assert_not_called()


def test_runtime_connections_and_resource_members_enforce_uuid_and_size_bounds():
    client = MagicMock()
    deployments = Deployments(client)
    resource_sets = McpResourceSets(client)

    invalid_connections = (
        {"mailboxes": "not-a-uuid"},
        {"mailboxes": [CONNECTION_ID, "not-a-uuid"]},
        {f"slot_{index}": CONNECTION_ID for index in range(101)},
    )
    for index, connections in enumerate(invalid_connections):
        with pytest.raises(ValueError, match="UUID|100 slots"):
            deployments.run(
                DEPLOYMENT_ID,
                external_user_id="customer-42",
                idempotency_key=f"invalid-bounds-{index}",
                connections=connections,
            )

    def member(index: int, *, slot: str) -> dict:
        return {
            "resource_type": "user_mcp_tool",
            "resource_id": f"00000000-0000-4000-8000-{index:012d}",
            "slot": slot,
            "allowed_actions": [],
            "position": index,
        }

    with pytest.raises(ValueError, match="at most 100 entries"):
        resource_sets.replace_members(
            OWNER_SET_ID,
            expected_revision=1,
            members=[member(index, slot=f"slot_{index}") for index in range(101)],
        )
    with pytest.raises(ValueError, match="at most 25 resources"):
        resource_sets.replace_members(
            OWNER_SET_ID,
            expected_revision=1,
            members=[member(index, slot="mailboxes") for index in range(26)],
        )
    invalid_member = member(1, slot="mailboxes")
    invalid_member["resource_id"] = "not-a-uuid"
    with pytest.raises(ValueError, match="resource_id must be a valid UUID"):
        resource_sets.replace_members(
            OWNER_SET_ID,
            expected_revision=1,
            members=[invalid_member],
        )

    client._request.assert_not_called()


def test_async_resource_sets_groups_and_named_mapping_parity():
    async def scenario():
        customer_member = _member_payload(
            public_id="00000000-0000-4000-8000-000000000099",
            resource_type="integration_connection",
            resource_id=CONNECTION_ID,
            alias="customer-sales",
            position=0,
        )
        client = MagicMock()
        client._request = AsyncMock(
            side_effect=[
                _access_payload(),
                _resource_set_payload(
                    public_id=CUSTOMER_SET_ID,
                    authority_type="external_principal",
                    management_mode="customer",
                    principal_id=PRINCIPAL_ID,
                ),
                _resource_set_payload(
                    public_id=CUSTOMER_SET_ID,
                    revision=2,
                    authority_type="external_principal",
                    management_mode="customer",
                    principal_id=PRINCIPAL_ID,
                    members=[customer_member],
                ),
                _group_payload(),
                _group_payload(),
                _run_payload(),
            ]
        )
        resource_sets = AsyncMcpResourceSets(client)
        groups = AsyncAgentGroups(client)
        deployments = AsyncDeployments(client)

        access = await deployments.access(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
        )
        principal = access.principal_for_external_user("customer-42")
        connection_id = access.ready_connection_ids_for_slot(
            principal_id=principal.id,
            slot="sender_inbox",
        )[0]
        created = await resource_sets.create(
            name="Mail operations",
            idempotency_key="async-customer-set-create-v1",
            principal_id=principal.id,
            management_mode=McpResourceSetManagementMode.CUSTOMER,
        )
        await resource_sets.replace_members(
            created.id,
            expected_revision=1,
            members=[
                McpResourceSetMemberInput(
                    resource_type=McpResourceType.INTEGRATION_CONNECTION,
                    resource_id=connection_id,
                    slot="mailboxes",
                    allowed_actions=["SEARCH", "SEND"],
                    position=0,
                )
            ],
        )
        group = await groups.create(
            name="Mail agents",
            idempotency_key="async-agent-group-create-v1",
            agent_ids=[AGENT_ID],
            resource_set_ids=[OWNER_SET_ID],
        )
        await groups.replace_assignments(
            group.id,
            agent_ids=[AGENT_ID],
            resource_set_ids=[OWNER_SET_ID],
        )
        await deployments.run(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            idempotency_key="async-deployment-named-mapping-42",
            resource_set_id=CUSTOMER_SET_ID,
            resource_set_revision=2,
        )

        assert client._request.await_args_list[-1].kwargs["json"] == {
            "external_user_id": "customer-42",
            "resource_set_id": CUSTOMER_SET_ID,
            "resource_set_revision": 2,
        }
        assert client._request.await_args_list[-1].kwargs["headers"] == {
            "Idempotency-Key": "async-deployment-named-mapping-42"
        }
        assert client._request.await_args_list[1].kwargs["headers"] == {
            "Idempotency-Key": "async-customer-set-create-v1"
        }
        assert client._request.await_args_list[3].kwargs["headers"] == {
            "Idempotency-Key": "async-agent-group-create-v1"
        }

    asyncio.run(scenario())


def test_async_lists_and_run_keys_match_sync_contracts():
    async def scenario():
        second_set_id = "22a12f0c-c71b-447d-a8c5-3294f0bdd9b0"
        second_group_id = "e333e023-a246-45fa-830f-2ef848231db0"
        list_client = MagicMock()
        list_client._request = AsyncMock(
            side_effect=[
                {
                    "next_cursor": "sets-next",
                    "previous_cursor": None,
                    "results": [_resource_set_summary_payload()],
                },
                {
                    "next_cursor": None,
                    "previous_cursor": "sets-previous",
                    "results": [_resource_set_summary_payload(public_id=second_set_id)],
                },
                {
                    "next_cursor": "groups-next",
                    "previous_cursor": None,
                    "results": [_group_payload()],
                },
                {
                    "next_cursor": None,
                    "previous_cursor": "groups-previous",
                    "results": [{**_group_payload(), "public_id": second_group_id}],
                },
            ]
        )
        resource_sets = await AsyncMcpResourceSets(list_client).list(
            page_size=1,
            query="  customer mail  ",
            authority_type=McpResourceSetAuthorityType.EXTERNAL_PRINCIPAL,
            principal_id=PRINCIPAL_ID,
        )
        groups = await AsyncAgentGroups(list_client).list(
            page_size=1,
            query="  mail agents  ",
        )
        assert [item.id for item in resource_sets] == [OWNER_SET_ID, second_set_id]
        assert [item.id for item in groups] == [GROUP_ID, second_group_id]
        assert list_client._request.await_args_list[1].kwargs["params"] == {
            "query": "customer mail",
            "authority_type": "external_principal",
            "principal_id": PRINCIPAL_ID,
            "page_size": 1,
            "cursor": "sets-next",
        }
        assert list_client._request.await_args_list[3].kwargs["params"] == {
            "query": "mail agents",
            "page_size": 1,
            "cursor": "groups-next",
        }

        run_client = MagicMock()
        run_client._request = AsyncMock(return_value=_run_payload())
        agents = AsyncAgents(run_client)
        compilations = AsyncCompilations(run_client)
        deployments = AsyncDeployments(run_client)
        tools = AsyncTools(run_client)

        await agents.run(AGENT_ID, idempotency_key=" async-agent-key ")
        assert run_client._request.await_args.kwargs["headers"] == {
            "Idempotency-Key": " async-agent-key "
        }
        run_client._request.reset_mock()
        await compilations.run_instruction(7, idempotency_key="async-frozen-key")
        assert run_client._request.await_args.kwargs["headers"] == {
            "Idempotency-Key": "async-frozen-key"
        }
        run_client._request.reset_mock()
        await deployments.run(
            DEPLOYMENT_ID,
            external_user_id="customer-42",
            idempotency_key="async-deployment-key",
        )
        assert run_client._request.await_args.kwargs["headers"] == {
            "Idempotency-Key": "async-deployment-key"
        }
        run_client._request.reset_mock()
        await tools.call(
            7,
            action="GMAIL_SEND_EMAIL",
            idempotency_key="async-direct-tool-key",
        )
        assert run_client._request.await_args.kwargs["headers"] == {
            "Idempotency-Key": "async-direct-tool-key"
        }

        for operation in (
            lambda key: agents.run(AGENT_ID, idempotency_key=key),
            lambda key: compilations.run_instruction(7, idempotency_key=key),
            lambda key: deployments.run(
                DEPLOYMENT_ID,
                external_user_id="customer-42",
                idempotency_key=key,
            ),
            lambda key: tools.call(
                7,
                action="GMAIL_SEND_EMAIL",
                idempotency_key=key,
            ),
        ):
            for invalid_key in (
                "",
                "   ",
                "x" * 256,
                "line\nbreak",
                "nul\x00key",
            ):
                run_client._request.reset_mock()
                with pytest.raises(ValueError, match="idempotency_key"):
                    await operation(invalid_key)
                run_client._request.assert_not_awaited()

        with pytest.raises(TypeError, match="idempotency_key"):
            cast(Any, agents.run)(AGENT_ID)
        with pytest.raises(TypeError, match="idempotency_key"):
            cast(Any, compilations.run_instruction)(7)
        with pytest.raises(TypeError, match="idempotency_key"):
            cast(Any, deployments.run)(
                DEPLOYMENT_ID,
                external_user_id="customer-42",
            )
        with pytest.raises(TypeError, match="idempotency_key"):
            cast(Any, tools.call)(7, action="GMAIL_SEND_EMAIL")

        create_client = MagicMock()
        create_client._request = AsyncMock(
            side_effect=[_resource_set_payload(), _group_payload()]
        )
        async_resource_sets = AsyncMcpResourceSets(create_client)
        async_groups = AsyncAgentGroups(create_client)
        await async_resource_sets.create(
            name="Mail operations",
            idempotency_key="async-resource-set-create-v1",
        )
        await async_groups.create(
            name="Mail agents",
            idempotency_key="async-agent-group-create-v1",
        )
        assert create_client._request.await_args_list[0].kwargs["headers"] == {
            "Idempotency-Key": "async-resource-set-create-v1"
        }
        assert create_client._request.await_args_list[1].kwargs["headers"] == {
            "Idempotency-Key": "async-agent-group-create-v1"
        }

        for operation in (
            lambda key: async_resource_sets.create(
                name="Mail operations",
                idempotency_key=key,
            ),
            lambda key: async_groups.create(
                name="Mail agents",
                idempotency_key=key,
            ),
        ):
            create_client._request.reset_mock()
            with pytest.raises(ValueError, match="idempotency_key"):
                await operation("zero\u200bwidth")
            create_client._request.assert_not_awaited()

        with pytest.raises(TypeError, match="idempotency_key"):
            cast(Any, async_resource_sets.create)(name="Mail operations")
        with pytest.raises(TypeError, match="idempotency_key"):
            cast(Any, async_groups.create)(name="Mail agents")

        run_client._request.reset_mock()
        with pytest.raises(NotImplementedError, match="caller-owned replay contract"):
            await compilations.run(7)
        run_client._request.assert_not_awaited()

    asyncio.run(scenario())


def test_async_tools_and_resource_set_members_follow_compact_cursor_pages():
    async def scenario():
        first_member = _member_payload(
            public_id="00000000-0000-4000-8000-000000000001",
            resource_type="user_mcp_tool",
            resource_id=TOOL_IDS[0],
            alias="sales",
            position=0,
        )
        second_member = _member_payload(
            public_id="00000000-0000-4000-8000-000000000002",
            resource_type="user_mcp_tool",
            resource_id=TOOL_IDS[1],
            alias="support",
            position=1,
        )
        client = MagicMock()
        client._request = AsyncMock(
            side_effect=[
                {
                    "next_cursor": "tools-next",
                    "previous_cursor": None,
                    "results": [_tool_payload(1, "sales")],
                },
                {
                    "next_cursor": None,
                    "previous_cursor": "tools-previous",
                    "results": [_tool_payload(2, "support")],
                },
                {
                    "next_cursor": "members-next",
                    "previous_cursor": None,
                    "results": [first_member],
                },
                {
                    "next_cursor": None,
                    "previous_cursor": "members-previous",
                    "results": [second_member],
                },
            ]
        )

        tools = await AsyncTools(client).list(
            page_size=1,
            mcp_tool="gmail",
        )
        members = await AsyncMcpResourceSets(client).list_members(
            OWNER_SET_ID,
            page_size=1,
        )

        assert [tool.public_id for tool in tools] == list(TOOL_IDS[:2])
        assert [member.resource_id for member in members] == list(TOOL_IDS[:2])
        assert client._request.await_args_list[0].kwargs["params"] == {
            "mcp_tool": "gmail",
            "page_size": 1,
        }
        assert client._request.await_args_list[1].kwargs["params"] == {
            "mcp_tool": "gmail",
            "page_size": 1,
            "cursor": "tools-next",
        }
        assert client._request.await_args_list[2].args == (
            "GET",
            f"/api/v1/agents/mcp-resource-sets/{OWNER_SET_ID}/members/",
        )
        assert client._request.await_args_list[3].kwargs["params"] == {
            "page_size": 1,
            "cursor": "members-next",
        }

    asyncio.run(scenario())


def test_async_scoped_to_legacy_uses_cleanup_then_separate_mode_patch():
    async def scenario():
        group_without_agent = {**_group_payload(), "agent_ids": []}
        scoped_without_direct_grants = {
            "uuid": AGENT_ID,
            "name": "Mail analyst",
            "user_prompt": "Compare the mailboxes",
            "mcp_resource_set_ids": [],
            "mcp_access_mode": "scoped",
            "created_at": NOW,
            "updated_at": NOW,
        }
        client = MagicMock()
        client._request = AsyncMock(
            side_effect=[
                {
                    "next_cursor": None,
                    "previous_cursor": None,
                    "results": [_group_payload()],
                },
                group_without_agent,
                scoped_without_direct_grants,
                {**scoped_without_direct_grants, "mcp_access_mode": "legacy"},
            ]
        )
        groups = AsyncAgentGroups(client)
        agents = AsyncAgents(client)

        for group in await groups.list():
            if AGENT_ID in group.agent_ids:
                await groups.replace_assignments(
                    group.id,
                    agent_ids=[item for item in group.agent_ids if item != AGENT_ID],
                    resource_set_ids=group.resource_set_ids,
                )
        cleared = await agents.update(AGENT_ID, mcp_resource_set_ids=[])
        legacy = await agents.update(
            AGENT_ID,
            mcp_access_mode=McpAccessMode.LEGACY,
        )

        assert cleared.mcp_access_mode is McpAccessMode.SCOPED
        assert legacy.mcp_access_mode is McpAccessMode.LEGACY
        assert client._request.await_args_list[1].kwargs["json"] == {
            "agent_ids": [],
            "resource_set_ids": [OWNER_SET_ID],
        }
        assert client._request.await_args_list[2].kwargs["json"] == {
            "mcp_resource_set_ids": []
        }
        assert client._request.await_args_list[3].kwargs["json"] == {
            "mcp_access_mode": "legacy"
        }

    asyncio.run(scenario())


def test_async_access_mode_metadata_cas_and_slug_parity():
    async def scenario():
        client = MagicMock()
        client._request = AsyncMock(
            side_effect=[
                {
                    "uuid": AGENT_ID,
                    "name": "Mail analyst",
                    "user_prompt": "Compare the mailboxes",
                    "mcp_access_mode": "scoped",
                    "mcp_resource_set_ids": [OWNER_SET_ID],
                    "created_at": NOW,
                    "updated_at": NOW,
                },
                {
                    "uuid": AGENT_ID,
                    "name": "Mail analyst",
                    "user_prompt": "Compare the mailboxes",
                    "created_at": NOW,
                    "updated_at": NOW,
                },
                _resource_set_payload(revision=2),
                _resource_set_payload(revision=3),
                _tool_payload(1, "sales_mail"),
            ]
        )
        agents = AsyncAgents(client)
        resource_sets = AsyncMcpResourceSets(client)
        tools = AsyncTools(client)

        agent = await agents.create(
            name="Mail analyst",
            goal="Compare the mailboxes",
            mcp_resource_set_ids=[OWNER_SET_ID],
            mcp_access_mode=McpAccessMode.SCOPED,
        )
        legacy = await agents.update(AGENT_ID, name="Mail analyst")
        await resource_sets.update(
            OWNER_SET_ID,
            expected_revision=1,
            description="Updated",
        )
        await resource_sets.replace(
            OWNER_SET_ID,
            expected_revision=2,
            name="Mail operations",
        )
        await tools.create(mcp_tool="gmail", alias="sales_mail")

        assert agent.mcp_access_mode is McpAccessMode.SCOPED
        assert legacy.mcp_access_mode is McpAccessMode.LEGACY
        assert (
            client._request.await_args_list[0].kwargs["json"]["mcp_access_mode"]
            == "scoped"
        )
        assert client._request.await_args_list[1].kwargs["json"] == {
            "name": "Mail analyst"
        }
        assert client._request.await_args_list[2].kwargs["json"] == {
            "expected_revision": 1,
            "description": "Updated",
        }
        assert client._request.await_args_list[3].args[0] == "PUT"
        assert (
            client._request.await_args_list[3].kwargs["json"]["expected_revision"] == 2
        )

        client._request.reset_mock()
        with pytest.raises(ValueError, match="alias"):
            await tools.create(mcp_tool="gmail", alias="sales mail")
        client._request.assert_not_awaited()

    asyncio.run(scenario())


def test_readme_is_personal_first_and_separates_rest_from_agents_mcp():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    personal = readme.index("## Personal connection first")
    advanced = readme.index("## Advanced connection access")
    embedded = readme.index("## Embedded customer agents")
    assert personal < advanced < embedded
    assert "implicit personal space" in readme[personal:advanced]
    assert "resource set" not in readme[personal:advanced].lower()
    assert "mcp_access_mode=McpAccessMode.LEGACY" in readme[personal:advanced]
    assert "gmail.redirect_url" in readme[personal:advanced]
    assert "gmail.is_configured" in readme[personal:advanced]
    assert "client.tools.get(gmail.id)" in readme[personal:advanced]
    assert "expected_revision=resource_set.revision" in readme[advanced:embedded]
    assert "management_mode=McpResourceSetManagementMode.CUSTOMER" in readme
    assert "client.mcp_resource_sets.list()" in readme[advanced:embedded]
    cleanup = readme.index("client.agents.update(agent.id, mcp_resource_set_ids=[])")
    mode_patch = readme.index(
        "client.agents.update(agent.id, mcp_access_mode=McpAccessMode.LEGACY)"
    )
    assert advanced < cleanup < mode_patch < embedded
    assert "If any cleanup request fails, do not send the final mode PATCH" in readme
    assert "Agents REST API" in readme
    assert "Agents MCP gateway" in readme
    assert "access()` is read-only" in readme
    assert "create_connection_link()" in readme
    assert "principal = access.principal_for_external_user(customer_id)" in readme
    assert "principal_id=principal.id" in readme
    assert "resource_set_id=customer_mapping.id" in readme
    assert "resource_set_revision=customer_mapping.revision" in readme
    assert "ready_connection_ids_for_slot" in readme
    assert "principal_id=principal_id" not in readme
    assert "Deprecated `client.compilations.run()`" in readme
    assert "`tools.call()`" in readme
    assert "never generates one\nfor the caller" in readme
