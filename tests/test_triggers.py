import asyncio
import inspect
import json

import httpx
import pytest
from pydantic import ValidationError

from flymyai.agents import (
    AsyncAgentClient,
    AsyncTriggersV1,
    FlyMyAIAgentError,
    TriggerEventV1,
    TriggersV1,
    TriggerV1,
)
from tests.test_mcp_teams import client_with, response

AGENT = "22222222-2222-4222-8222-222222222222"
TRIGGER = "11111111-1111-4111-8111-111111111111"
ROOT = "/api/v1/agents/triggers/"
ROW = {
    "schema": "flymy.trigger.v1",
    "uuid": TRIGGER,
    "agent_task": AGENT,
    "kind": "webhook",
    "status": "active",
    "config": {},
    "rate_limit_per_hour": 30,
    "signature_required": True,
    "webhook_url": "https://example.test/api/v1/hooks/w/private-route/",
    "created_at": "2026-10-13T12:00:00Z",
    "future_field": "kept",
}
EVENT = {
    "schema": "flymy.trigger-event.v1",
    "uuid": TRIGGER,
    "provider": "webhook",
    "status": "dispatched",
    "detail": "",
    "execution": 12,
    "execution_public_id": "aaa-aaaa-aaa",
    "received_at": "2026-10-13T12:00:00Z",
}


def handler(seen):
    def answer(request):
        seen.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.url.path.endswith("/events/"):
            return response([EVENT])
        if request.method == "GET" and request.url.path == ROOT:
            return response([ROW])
        return response(ROW)

    return answer


def check_requests(seen):
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", ROOT),
        ("GET", f"{ROOT}{TRIGGER}/"),
        ("POST", ROOT),
        ("PATCH", f"{ROOT}{TRIGGER}/"),
        ("POST", f"{ROOT}{TRIGGER}/pause/"),
        ("POST", f"{ROOT}{TRIGGER}/activate/"),
        ("GET", f"{ROOT}{TRIGGER}/events/"),
        ("DELETE", f"{ROOT}{TRIGGER}/"),
    ]
    assert seen[0].url.params["task"] == AGENT
    assert [r.headers["Idempotency-Key"] for r in seen if r.method != "GET"] == [
        "create",
        "update",
        "pause",
        "activate",
        "delete",
    ]
    assert json.loads(seen[2].content) == {
        "agent_task": AGENT,
        "kind": "webhook",
        "secret": "test-only",
    }
    assert json.loads(seen[3].content) == {"config": {}}
    assert all(r.headers["X-API-KEY"] == "personal-key" for r in seen)


def test_sync_complete_versioned_resource():
    seen = []
    with client_with(handler(seen)) as client:
        resource = client.triggers_v1
        assert str(resource.list(agent_id=AGENT)[0].uuid) == TRIGGER
        assert resource.get(TRIGGER).model_extra["future_field"] == "kept"
        made = resource.create(AGENT, secret="test-only", idempotency_key="create")
        assert "private-route" not in repr(made)
        resource.update(TRIGGER, config={}, idempotency_key="update")
        resource.pause(TRIGGER, idempotency_key="pause")
        resource.activate(TRIGGER, idempotency_key="activate")
        assert resource.events(TRIGGER)[0].execution_public_id == "aaa-aaaa-aaa"
        assert resource.delete(TRIGGER, idempotency_key="delete") is None
    check_requests(seen)


def test_async_complete_versioned_resource():
    seen = []

    async def run():
        async with AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        ) as client:
            await client._http.aclose()
            client._http = httpx.AsyncClient(
                base_url="https://example.test",
                headers={"X-API-KEY": "personal-key"},
                transport=httpx.MockTransport(handler(seen)),
            )
            resource = client.triggers_v1
            await resource.list(agent_id=AGENT)
            await resource.get(TRIGGER)
            await resource.create(AGENT, secret="test-only", idempotency_key="create")
            await resource.update(TRIGGER, config={}, idempotency_key="update")
            await resource.pause(TRIGGER, idempotency_key="pause")
            await resource.activate(TRIGGER, idempotency_key="activate")
            assert (await resource.events(TRIGGER))[
                0
            ].execution_public_id == "aaa-aaaa-aaa"
            await resource.delete(TRIGGER, idempotency_key="delete")

    asyncio.run(run())
    check_requests(seen)


def test_wrong_contract_and_unsafe_identity_fail_without_secret_disclosure():
    with pytest.raises(ValidationError) as error:
        TriggerV1.model_validate({**ROW, "schema": "flymy.trigger.v2"})
    assert "private-route" not in str(error.value)
    seen = []
    with client_with(handler(seen)) as client:
        with pytest.raises(ValueError):
            client.triggers_v1.get("../../users")
        with pytest.raises(ValueError):
            client.triggers_v1.create(AGENT, secret="test-only", idempotency_key="")
    assert seen == []


def test_lost_response_does_not_automatically_repeat_write():
    seen = []

    def timeout(request):
        seen.append(request)
        raise httpx.ReadTimeout("test timeout", request=request)

    with client_with(timeout) as client:
        with pytest.raises(Exception):
            client.triggers_v1.create(
                AGENT, secret="test-only", idempotency_key="stable"
            )
    assert len(seen) == 1
    assert seen[0].headers["Idempotency-Key"] == "stable"


@pytest.mark.parametrize("status", [200, 202])
def test_delete_requires_204_for_sync_and_async(status):
    seen = []

    def ambiguous(request):
        seen.append(request)
        return httpx.Response(status, json={"private": "must-not-be-disclosed"})

    with client_with(ambiguous) as client:
        with pytest.raises(FlyMyAIAgentError, match="outcome unknown") as caught:
            client.triggers_v1.delete(TRIGGER, idempotency_key="same-delete")
        assert caught.value.response_body is None
        assert "must-not-be-disclosed" not in str(caught.value)

    async def run():
        async with AsyncAgentClient(
            api_key="test", base_url="https://example.test"
        ) as client:
            await client._http.aclose()
            client._http = httpx.AsyncClient(
                base_url="https://example.test",
                transport=httpx.MockTransport(ambiguous),
            )
            with pytest.raises(FlyMyAIAgentError, match="outcome unknown"):
                await client.triggers_v1.delete(TRIGGER, idempotency_key="same-delete")

    asyncio.run(run())
    assert len(seen) == 2
    assert all(r.headers["Idempotency-Key"] == "same-delete" for r in seen)


# flymy.trigger.v1, released 2026-10-16 in flymyai 1.2.0rc15. Callers keep these
# names and arguments when a v2 namespace is added: within v1 only optional
# arguments and fields may be added (backend ADDING_A_NEW_TRIGGER.md).
V1_METHODS = {
    "list": {"agent_id": False},
    "get": {"trigger_id": True},
    "create": {
        "agent_id": True,
        "idempotency_key": True,
        "secret": True,
        "rate_limit_per_hour": False,
        "config": False,
    },
    "update": {
        "trigger_id": True,
        "idempotency_key": True,
        "secret": False,
        "rate_limit_per_hour": False,
        "config": False,
    },
    "pause": {"trigger_id": True, "idempotency_key": True},
    "activate": {"trigger_id": True, "idempotency_key": True},
    "delete": {"trigger_id": True, "idempotency_key": True},
    "events": {"trigger_id": True},
}
V1_POSITIONAL = {
    "list": [],
    "get": ["trigger_id"],
    "create": ["agent_id"],
    "update": ["trigger_id"],
    "pause": ["trigger_id"],
    "activate": ["trigger_id"],
    "delete": ["trigger_id"],
    "events": ["trigger_id"],
}
# Every field of the answers the backend sends, as production returned them.
V1_RECORD = {
    "schema": "flymy.trigger.v1",
    "uuid": TRIGGER,
    "agent_task": AGENT,
    "compilation": None,
    "kind": "webhook",
    "config": {"filters": {"contains": "order"}},
    "status": "active",
    "rate_limit_per_hour": 30,
    "webhook_url": "https://example.test/api/v1/hooks/w/private-route/",
    "signature_required": True,
    "event_count": 0,
    "dispatch_count": 0,
    "last_event_at": None,
    "last_dispatch_at": None,
    "created_at": "2026-10-16T12:54:03.123456Z",
}
V1_EVENT = {
    "schema": "flymy.trigger-event.v1",
    "uuid": TRIGGER,
    "provider": "webhook",
    "status": "filtered",
    "detail": "filtered by config",
    "execution": None,
    "execution_public_id": None,
    "received_at": "2026-10-16T12:56:48.123456Z",
    "processed_at": None,
}


@pytest.mark.parametrize("resource", [TriggersV1, AsyncTriggersV1])
def test_the_v1_methods_and_arguments_stay(resource):
    for name, pinned in V1_METHODS.items():
        method = getattr(resource, name)
        assert inspect.iscoroutinefunction(method) == (resource is AsyncTriggersV1)
        params = dict(list(inspect.signature(method).parameters.items())[1:])
        for argument, required in pinned.items():
            assert argument in params, f"{name} lost {argument}"
            assert (params[argument].default is inspect.Parameter.empty) == required
        for argument, param in params.items():
            if argument not in pinned:
                assert param.default is not inspect.Parameter.empty, argument
        positional = [
            argument
            for argument, param in params.items()
            if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert positional == V1_POSITIONAL[name], name


def test_both_clients_expose_the_v1_namespace():
    with client_with(handler([])) as client:
        assert isinstance(client.triggers_v1, TriggersV1)

    async def run():
        async with AsyncAgentClient(
            api_key="test", base_url="https://example.test"
        ) as client:
            assert isinstance(client.triggers_v1, AsyncTriggersV1)

    asyncio.run(run())


def test_the_v1_answer_models_know_every_field_and_require_no_more():
    def aliases(model):
        return {field.alias or name for name, field in model.model_fields.items()}

    def required(model):
        return {
            field.alias or name
            for name, field in model.model_fields.items()
            if field.is_required()
        }

    assert aliases(TriggerV1) == set(V1_RECORD)
    assert aliases(TriggerEventV1) == set(V1_EVENT)
    assert required(TriggerV1) == {
        "schema",
        "uuid",
        "agent_task",
        "kind",
        "status",
        "config",
        "rate_limit_per_hour",
        "signature_required",
        "created_at",
    }
    assert required(TriggerEventV1) == {
        "schema",
        "uuid",
        "provider",
        "status",
        "detail",
        "received_at",
    }
    trigger = TriggerV1.model_validate(V1_RECORD)
    assert trigger.compilation is None and trigger.last_event_at is None
    assert (
        trigger.model_dump(mode="json", by_alias=True)["schema"] == "flymy.trigger.v1"
    )
    event = TriggerEventV1.model_validate(V1_EVENT)
    assert event.execution is None and event.execution_public_id is None
    with pytest.raises(ValidationError):
        TriggerEventV1.model_validate({**V1_EVENT, "schema": "flymy.trigger.v1"})
