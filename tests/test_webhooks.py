import asyncio
import base64
import hashlib
import hmac
import json

import httpx
import pytest

from flymyai import (
    FlyMyAIAgentError,
    InvalidWebhookError,
    WebhookLimitError,
    WebhookSignatureError,
    verify_webhook,
)
from flymyai.agents import (
    AsyncWebhooks,
    Webhook,
    WebhookDelivery,
    WebhookEvent,
    WebhookList,
    Webhooks,
    WebhookWithSecret,
)
from flymyai.agents._client import AsyncAgentClient
from tests.test_mcp_teams import client_with, response

ROOT = "/api/v1/agents/webhooks/"
WEBHOOK_ID = "1b0d6c6e-5d0a-4a43-9d1e-6b8e1c9f2a10"
WEBHOOK = {
    "id": WEBHOOK_ID,
    "url": "https://example.com/flymyai/events",
    "events": ["connection.expired"],
    "description": "Reconnect alerts",
    "is_active": True,
    "created_at": "2026-11-17T02:00:00+00:00",
    "last_delivery_at": None,
    "last_delivery_status": None,
    "consecutive_failures": 0,
}
SECRET = "whsec_" + base64.b64encode(b"k" * 32).decode()
DELIVERY = {
    "id": "6f1b2c3d-0000-4000-8000-000000000001",
    "event_type": "webhook.test",
    "status": "pending",
    "attempts": 0,
    "last_status_code": None,
    "last_error": None,
    "created_at": "2026-11-17T02:01:00+00:00",
    "delivered_at": None,
}


def _answer(**extra):
    return {"schema": "flymy.webhooks.v1", "webhook": WEBHOOK, **extra}


def test_create_list_update_test_rotate_and_delete_over_rest():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((
            request.method,
            request.url.path,
            body,
            request.headers.get("Idempotency-Key"),
        ))
        path, method = request.url.path, request.method
        if path == ROOT and method == "POST":
            return response(_answer(secret=SECRET), 201)
        if path == ROOT:
            return response({
                "schema": "flymy.webhooks.v1",
                "webhooks": [WEBHOOK],
                "event_types": {"connection.expired": "A connection died."},
            })
        if path.endswith("/test/"):
            return response({"schema": "flymy.webhooks.v1", "delivery": DELIVERY}, 202)
        if path.endswith("/rotate-secret/"):
            return response(_answer(secret=SECRET), 200)
        if path.endswith("/deliveries/"):
            return response({"schema": "flymy.webhooks.v1", "deliveries": [DELIVERY]})
        if method == "DELETE":
            return httpx.Response(204)
        if method == "PATCH":
            return response(_answer(webhook={**WEBHOOK, "is_active": False}))
        return response(_answer())

    client = client_with(handler)
    created = client.webhooks.create(
        WEBHOOK["url"],
        events=["connection.expired"],
        description="Reconnect alerts",
        idempotency_key="webhook-1",
    )
    listed = client.webhooks.list()
    got = client.webhooks.get(WEBHOOK_ID)
    paused = client.webhooks.update(WEBHOOK_ID, is_active=False, idempotency_key="p-1")
    queued = client.webhooks.send_test(WEBHOOK_ID)
    rotated = client.webhooks.rotate_secret(WEBHOOK_ID, idempotency_key="r-1")
    deliveries = client.webhooks.list_deliveries(WEBHOOK_ID)
    assert client.webhooks.delete(WEBHOOK_ID) is None

    assert isinstance(created, WebhookWithSecret) and created.secret == SECRET
    assert created.webhook.events == ["connection.expired"]
    assert isinstance(listed, WebhookList) and listed.webhooks[0].id == WEBHOOK_ID
    assert "connection.expired" in listed.event_types
    assert isinstance(got, Webhook) and got.url == WEBHOOK["url"]
    assert paused.is_active is False
    assert isinstance(queued, WebhookDelivery) and queued.event_type == "webhook.test"
    assert rotated.secret == SECRET
    assert [d.status for d in deliveries] == ["pending"]
    item = f"{ROOT}{WEBHOOK_ID}/"
    assert seen == [
        (
            "POST",
            ROOT,
            {
                "url": WEBHOOK["url"],
                "events": ["connection.expired"],
                "description": "Reconnect alerts",
            },
            "webhook-1",
        ),
        ("GET", ROOT, None, None),
        ("GET", item, None, None),
        ("PATCH", item, {"is_active": False}, "p-1"),
        ("POST", f"{item}test/", None, None),
        ("POST", f"{item}rotate-secret/", None, "r-1"),
        ("GET", f"{item}deliveries/", None, None),
        ("DELETE", item, None, None),
    ]


def test_a_replayed_create_has_no_secret():
    def handler(request):
        return response(_answer(secret=None, detail="The secret was shown once."), 201)

    created = client_with(handler).webhooks.create(
        WEBHOOK["url"], events=["connection.expired"], idempotency_key="webhook-1"
    )
    assert created.secret is None and created.webhook.id == WEBHOOK_ID


@pytest.mark.parametrize(
    "call",
    [
        lambda w: w.create("http://example.com/x", events=["connection.expired"]),
        lambda w: w.create("https://example.com/x", events=[]),
        lambda w: w.create("https://example.com/x", events="connection.expired"),
        lambda w: w.create(
            "https://example.com/x",
            events=["connection.expired"],
            description="x" * 201,
        ),
        lambda w: w.update(WEBHOOK_ID),
        lambda w: w.update(WEBHOOK_ID, is_active="no"),
        lambda w: w.get("../tools"),
        lambda w: w.delete(WEBHOOK_ID, idempotency_key=" padded "),
    ],
)
def test_a_bad_argument_is_refused_before_any_request(call):
    def handler(request):
        raise AssertionError("no request may be sent")

    with pytest.raises(ValueError):
        call(client_with(handler).webhooks)


def test_refusals_are_typed_by_their_stable_codes():
    answers = [
        (400, {"code": "invalid_webhook", "field": "url", "detail": "Not public."}),
        (409, {"code": "webhook_limit", "detail": "At most 10 webhooks."}),
    ]

    def handler(request):
        status, body = answers.pop(0)
        return response(body, status)

    webhooks = client_with(handler).webhooks
    with pytest.raises(InvalidWebhookError) as invalid:
        webhooks.create("https://10.0.0.1/x", events=["connection.expired"])
    with pytest.raises(WebhookLimitError):
        webhooks.create("https://example.com/x", events=["connection.expired"])
    assert invalid.value.field == "url" and invalid.value.status_code == 400


def test_an_unexpected_write_status_says_the_outcome_is_unknown():
    def handler(request):
        return response(_answer(secret=SECRET), 200)

    with pytest.raises(FlyMyAIAgentError, match="outcome unknown"):
        client_with(handler).webhooks.create(
            WEBHOOK["url"], events=["connection.expired"], idempotency_key="k-1"
        )


def _signed(body: bytes, *, secret=SECRET, message_id="msg_1", timestamp=1_800_000_000):
    key = base64.b64decode(secret[len("whsec_") :])
    digest = hmac.new(
        key, f"{message_id}.{timestamp}.".encode() + body, hashlib.sha256
    ).digest()
    return {
        "Webhook-Id": message_id,
        "Webhook-Timestamp": str(timestamp),
        "Webhook-Signature": "v1," + base64.b64encode(digest).decode(),
    }


EVENT = json.dumps(
    {
        "id": "msg_1",
        "type": "connection.expired",
        "created_at": "2026-11-17T02:00:00+00:00",
        "schema": "flymy.webhooks.v1",
        "data": {"connection": {"toolkit": "linear"}, "reconnect_url": "https://a"},
    },
    separators=(",", ":"),
).encode()


def test_a_genuine_delivery_verifies_and_returns_its_event():
    event = verify_webhook(EVENT, _signed(EVENT), SECRET, now=1_800_000_100)

    assert isinstance(event, WebhookEvent)
    assert event.type == "connection.expired"
    assert event.data["connection"]["toolkit"] == "linear"
    # The same check through the client, and with a rotated-in second signature.
    headers = _signed(EVENT)
    headers["Webhook-Signature"] = "v1,AAAA " + headers["Webhook-Signature"]
    assert Webhooks.verify(EVENT, headers, SECRET, now=1_800_000_000).id == "msg_1"
    assert AsyncWebhooks.verify(EVENT, headers, SECRET, now=1_800_000_000).id == "msg_1"


@pytest.mark.parametrize(
    "body, headers, now",
    [
        (EVENT + b" ", _signed(EVENT), 1_800_000_000),  # the body was changed
        (EVENT, _signed(EVENT, message_id="msg_2"), None),  # another id, same signature
        (
            EVENT,
            _signed(EVENT, secret="whsec_" + base64.b64encode(b"x" * 32).decode()),
            1_800_000_000,
        ),  # another secret
        (EVENT, _signed(EVENT), 1_800_000_301),  # older than five minutes
        (EVENT, {"webhook-id": "msg_1"}, 1_800_000_000),  # headers missing
    ],
)
def test_a_forged_stale_or_partial_delivery_is_refused(body, headers, now):
    if now is None:
        headers = {**headers, "Webhook-Id": "msg_1"}
        now = 1_800_000_000
    with pytest.raises(WebhookSignatureError):
        verify_webhook(body, headers, SECRET, now=now)


def test_the_async_client_has_the_same_webhooks():
    def handler(request):
        if request.url.path.endswith("/test/"):
            return httpx.Response(
                202,
                stream=httpx.ByteStream(
                    json.dumps(
                        {"schema": "flymy.webhooks.v1", "delivery": DELIVERY}
                    ).encode()
                ),
            )
        return httpx.Response(
            201, stream=httpx.ByteStream(json.dumps(_answer(secret=SECRET)).encode())
        )

    async def go():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            created = await client.webhooks.create(
                WEBHOOK["url"], events=["connection.expired"]
            )
            queued = await client.webhooks.send_test(created.webhook.id)
        finally:
            await client.close()
        return created, queued

    created, queued = asyncio.run(go())
    assert created.secret == SECRET and queued.event_type == "webhook.test"


# The v1 methods stay (a frozen program keeps the calls it was written with): one may be
# added, none renamed or removed.
V1_METHODS = {
    "list",
    "create",
    "get",
    "update",
    "delete",
    "send_test",
    "rotate_secret",
    "list_deliveries",
    "verify",
}


@pytest.mark.parametrize("resource", [Webhooks, AsyncWebhooks])
def test_the_v1_methods_stay(resource):
    assert V1_METHODS <= {name for name in dir(resource) if not name.startswith("_")}
