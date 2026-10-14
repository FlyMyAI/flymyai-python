import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from flymyai.agents import AsyncAgentClient, TriggerV1
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
