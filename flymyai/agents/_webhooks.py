"""Webhooks (``flymy.webhooks.v1``): signed platform events to your own endpoint.

``client.webhooks`` subscribes a public ``https://`` endpoint to event types, first
``connection.expired`` (a connected account stopped accepting FlyMyAI's access and must
be reconnected). Each event is one POST of a JSON envelope, ``{"id", "type",
"created_at", "schema", "data"}``, signed the Standard Webhooks way. The signing secret
(``whsec_...``) is in the answer of ``create`` and ``rotate_secret`` only: store it where
your receiver reads it. In the receiver, :func:`verify_webhook` checks a delivery and
returns its event. Unknown answer fields are kept on the models (``extra="allow"``).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from flymyai.agents._artifacts import _key

_ROOT = "/api/v1/agents/webhooks/"
_URL_MAX_CHARS = 2048
_DESCRIPTION_MAX_CHARS = 200
# Standard Webhooks: refuse a delivery signed more than five minutes away from now.
SIGNATURE_TOLERANCE_SECONDS = 300


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="allow", hide_input_in_errors=True, populate_by_name=True
    )


class Webhook(_Model):
    """A subscription; never its secret."""

    id: str
    url: str
    events: List[str]
    description: str = ""
    is_active: bool = True
    created_at: Optional[str] = None
    last_delivery_at: Optional[str] = None
    last_delivery_status: Optional[str] = None
    consecutive_failures: int = 0


class WebhookList(_Model):
    """Your webhooks and every event type you can subscribe to (type -> description)."""

    schema_version: str = Field(default="flymy.webhooks.v1", alias="schema")
    webhooks: List[Webhook]
    event_types: Dict[str, str] = Field(default_factory=dict)


class WebhookWithSecret(_Model):
    """The answer of ``create`` and ``rotate_secret``. ``secret`` is ``None`` when the
    answer is a replay of an earlier request with the same idempotency key: the secret
    is shown once, so rotate it if it was lost."""

    schema_version: str = Field(default="flymy.webhooks.v1", alias="schema")
    webhook: Webhook
    secret: Optional[str] = None
    detail: Optional[str] = None


class WebhookDelivery(_Model):
    """One event sent to a webhook: ``pending`` (being sent or retried),
    ``delivered`` or ``failed``."""

    id: str
    event_type: str
    status: str
    attempts: int = 0
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None
    created_at: Optional[str] = None
    delivered_at: Optional[str] = None


class WebhookEvent(_Model):
    """A verified event, as :func:`verify_webhook` returns it. ``id`` is the
    ``webhook-id`` header: the same on every retry, so use it to skip duplicates."""

    id: str
    type: str
    created_at: Optional[str] = None
    schema_version: str = Field(default="flymy.webhooks.v1", alias="schema")
    data: Dict[str, Any] = Field(default_factory=dict)


class WebhookSignatureError(ValueError):
    """A request is not a genuine, fresh FlyMyAI delivery: answer it with 400."""


def _id(webhook_id: Union[str, UUID]) -> str:
    try:
        return str(UUID(str(webhook_id)))
    except ValueError:
        raise ValueError("webhook_id must be a webhook's id (a UUID).") from None


def _url(url: Any) -> str:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("url must be an https:// URL.")
    if len(url) > _URL_MAX_CHARS:
        raise ValueError(f"url must be at most {_URL_MAX_CHARS} characters.")
    return url


def _events(events: Any) -> List[str]:
    if isinstance(events, str) or not isinstance(events, Sequence) or not events:
        raise ValueError("events must be a non-empty list of event types.")
    if not all(isinstance(event, str) and event.strip() for event in events):
        raise ValueError("events must be a non-empty list of event types.")
    return list(events)


def _description(description: Any) -> str:
    if not isinstance(description, str):
        raise ValueError("description must be a string.")
    if len(description) > _DESCRIPTION_MAX_CHARS:
        raise ValueError(
            f"description must be at most {_DESCRIPTION_MAX_CHARS} characters."
        )
    return description


def create_body(
    url: str, events: Sequence[str], description: Optional[str] = None
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"url": _url(url), "events": _events(events)}
    if description is not None:
        body["description"] = _description(description)
    return body


def update_body(
    *,
    url: Optional[str] = None,
    events: Optional[Sequence[str]] = None,
    description: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if url is not None:
        body["url"] = _url(url)
    if events is not None:
        body["events"] = _events(events)
    if description is not None:
        body["description"] = _description(description)
    if is_active is not None:
        if not isinstance(is_active, bool):
            raise ValueError("is_active must be True or False.")
        body["is_active"] = is_active
    if not body:
        raise ValueError("Give at least one of url, events, description, is_active.")
    return body


def _headers(idempotency_key: Optional[str]) -> Dict[str, str]:
    return {} if idempotency_key is None else _key(idempotency_key)


def _signing_key(secret: str) -> bytes:
    if not isinstance(secret, str) or not secret:
        raise ValueError("secret must be the webhook's whsec_ secret.")
    encoded = secret[len("whsec_") :] if secret.startswith("whsec_") else secret
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("secret must be the webhook's whsec_ secret.") from None


def verify_webhook(
    payload: Union[bytes, str],
    headers: Mapping[str, str],
    secret: str,
    *,
    tolerance_seconds: int = SIGNATURE_TOLERANCE_SECONDS,
    now: Optional[float] = None,
) -> WebhookEvent:
    """Check one delivery and return its event.

    ``payload`` is the raw request body exactly as received (verify before parsing
    it), ``headers`` the request's headers (any mapping; names are matched without
    case) and ``secret`` the webhook's ``whsec_`` secret. Raises
    :class:`WebhookSignatureError` when a header is missing, the signature does not
    match or the timestamp is more than ``tolerance_seconds`` away from now."""
    key = _signing_key(secret)
    body = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    named = {str(name).lower(): value for name, value in headers.items()}
    message_id = named.get("webhook-id")
    timestamp = named.get("webhook-timestamp")
    signatures = named.get("webhook-signature")
    if not message_id or not timestamp or not signatures:
        raise WebhookSignatureError(
            "A webhook-id, -timestamp or -signature is missing."
        )
    if not (timestamp.isascii() and timestamp.isdigit()) or len(timestamp) > 12:
        raise WebhookSignatureError("webhook-timestamp is not a Unix time.")
    current = time.time() if now is None else now
    if abs(current - int(timestamp)) > tolerance_seconds:
        raise WebhookSignatureError("webhook-timestamp is too far from now.")
    signed = f"{message_id}.{timestamp}.".encode("utf-8") + body
    expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest())
    for candidate in signatures.split():
        version, _, value = candidate.partition(",")
        if version == "v1" and hmac.compare_digest(value.encode("utf-8"), expected):
            break
    else:
        raise WebhookSignatureError("No webhook-signature matches this secret.")
    try:
        event = json.loads(body)
    except ValueError:
        raise WebhookSignatureError("The signed body is not JSON.") from None
    if not isinstance(event, dict):
        raise WebhookSignatureError("The signed body is not an event.")
    return WebhookEvent.model_validate(event)


class Webhooks:
    """``client.webhooks``: send platform events to your own endpoint.

    Writes take an optional ``idempotency_key``: with one, a retry of the same request
    changes nothing twice and answers the same (a replayed ``create`` or
    ``rotate_secret`` has ``secret=None``)."""

    verify = staticmethod(verify_webhook)

    def __init__(self, client: Any) -> None:
        self._c = client

    def list(self) -> WebhookList:
        """Your webhooks (at most 10) and the event types you can subscribe to."""
        return WebhookList.model_validate(self._c._request("GET", _ROOT))

    def create(
        self,
        url: str,
        *,
        events: Sequence[str],
        description: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> WebhookWithSecret:
        """Subscribe the public ``https://`` ``url`` to ``events``. The answer carries
        the signing secret, once. Raises :class:`~flymyai.agents.InvalidWebhookError`
        for a refused URL, event or field and
        :class:`~flymyai.agents.WebhookLimitError` past 10 webhooks."""
        return WebhookWithSecret.model_validate(
            self._c._request(
                "POST",
                _ROOT,
                json=create_body(url, events, description),
                headers=_headers(idempotency_key),
                expected_status=201,
            )
        )

    def get(self, webhook_id: Union[str, UUID]) -> Webhook:
        answer = self._c._request("GET", f"{_ROOT}{_id(webhook_id)}/")
        return Webhook.model_validate(answer["webhook"])

    def update(
        self,
        webhook_id: Union[str, UUID],
        *,
        url: Optional[str] = None,
        events: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        idempotency_key: Optional[str] = None,
    ) -> Webhook:
        """Change the URL, the events or the description, or pause
        (``is_active=False``) and resume it."""
        answer = self._c._request(
            "PATCH",
            f"{_ROOT}{_id(webhook_id)}/",
            json=update_body(
                url=url, events=events, description=description, is_active=is_active
            ),
            headers=_headers(idempotency_key),
            expected_status=200,
        )
        return Webhook.model_validate(answer["webhook"])

    def delete(
        self, webhook_id: Union[str, UUID], *, idempotency_key: Optional[str] = None
    ) -> None:
        self._c._request(
            "DELETE",
            f"{_ROOT}{_id(webhook_id)}/",
            headers=_headers(idempotency_key),
            expected_status=204,
        )

    def send_test(
        self, webhook_id: Union[str, UUID], *, idempotency_key: Optional[str] = None
    ) -> WebhookDelivery:
        """Queue one ``webhook.test`` event; follow it in :meth:`list_deliveries`."""
        answer = self._c._request(
            "POST",
            f"{_ROOT}{_id(webhook_id)}/test/",
            headers=_headers(idempotency_key),
            expected_status=202,
        )
        return WebhookDelivery.model_validate(answer["delivery"])

    def rotate_secret(
        self, webhook_id: Union[str, UUID], *, idempotency_key: Optional[str] = None
    ) -> WebhookWithSecret:
        """A new signing secret, at once: the old one stops verifying."""
        return WebhookWithSecret.model_validate(
            self._c._request(
                "POST",
                f"{_ROOT}{_id(webhook_id)}/rotate-secret/",
                headers=_headers(idempotency_key),
                expected_status=200,
            )
        )

    def list_deliveries(self, webhook_id: Union[str, UUID]) -> List[WebhookDelivery]:
        """The 50 newest deliveries of the last 30 days, newest first."""
        answer = self._c._request("GET", f"{_ROOT}{_id(webhook_id)}/deliveries/")
        return [WebhookDelivery.model_validate(d) for d in answer["deliveries"]]


class AsyncWebhooks:
    """``client.webhooks`` of :class:`AsyncAgentClient`."""

    verify = staticmethod(verify_webhook)

    def __init__(self, client: Any) -> None:
        self._c = client

    async def list(self) -> WebhookList:
        return WebhookList.model_validate(await self._c._request("GET", _ROOT))

    async def create(
        self,
        url: str,
        *,
        events: Sequence[str],
        description: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> WebhookWithSecret:
        return WebhookWithSecret.model_validate(
            await self._c._request(
                "POST",
                _ROOT,
                json=create_body(url, events, description),
                headers=_headers(idempotency_key),
                expected_status=201,
            )
        )

    async def get(self, webhook_id: Union[str, UUID]) -> Webhook:
        answer = await self._c._request("GET", f"{_ROOT}{_id(webhook_id)}/")
        return Webhook.model_validate(answer["webhook"])

    async def update(
        self,
        webhook_id: Union[str, UUID],
        *,
        url: Optional[str] = None,
        events: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        idempotency_key: Optional[str] = None,
    ) -> Webhook:
        answer = await self._c._request(
            "PATCH",
            f"{_ROOT}{_id(webhook_id)}/",
            json=update_body(
                url=url, events=events, description=description, is_active=is_active
            ),
            headers=_headers(idempotency_key),
            expected_status=200,
        )
        return Webhook.model_validate(answer["webhook"])

    async def delete(
        self, webhook_id: Union[str, UUID], *, idempotency_key: Optional[str] = None
    ) -> None:
        await self._c._request(
            "DELETE",
            f"{_ROOT}{_id(webhook_id)}/",
            headers=_headers(idempotency_key),
            expected_status=204,
        )

    async def send_test(
        self, webhook_id: Union[str, UUID], *, idempotency_key: Optional[str] = None
    ) -> WebhookDelivery:
        answer = await self._c._request(
            "POST",
            f"{_ROOT}{_id(webhook_id)}/test/",
            headers=_headers(idempotency_key),
            expected_status=202,
        )
        return WebhookDelivery.model_validate(answer["delivery"])

    async def rotate_secret(
        self, webhook_id: Union[str, UUID], *, idempotency_key: Optional[str] = None
    ) -> WebhookWithSecret:
        return WebhookWithSecret.model_validate(
            await self._c._request(
                "POST",
                f"{_ROOT}{_id(webhook_id)}/rotate-secret/",
                headers=_headers(idempotency_key),
                expected_status=200,
            )
        )

    async def list_deliveries(
        self, webhook_id: Union[str, UUID]
    ) -> List[WebhookDelivery]:
        answer = await self._c._request("GET", f"{_ROOT}{_id(webhook_id)}/deliveries/")
        return [WebhookDelivery.model_validate(d) for d in answer["deliveries"]]
