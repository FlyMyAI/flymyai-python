"""Versioned webhook management, ``client.triggers_v1``.

Only generic webhooks are supported. A delivery admits the owner's Live frozen
version; acceptance is not successful agent completion. Keep these methods when
a future API version is added. Unknown response fields are preserved.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from flymyai.agents._resources import _idempotency_headers

_ROOT = "/api/v1/agents/triggers/"


class TriggerV1(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    schema_: Literal["flymy.trigger.v1"] = Field(alias="schema")
    uuid: UUID
    agent_task: UUID
    kind: str
    status: str
    config: Dict[str, Any]
    rate_limit_per_hour: int
    webhook_url: Optional[str] = Field(default=None, repr=False)
    signature_required: bool
    compilation: Optional[int] = None
    event_count: int = 0
    dispatch_count: int = 0
    last_event_at: Optional[datetime] = None
    last_dispatch_at: Optional[datetime] = None
    created_at: datetime


class TriggerEventV1(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)

    schema_: Literal["flymy.trigger-event.v1"] = Field(alias="schema")
    uuid: UUID
    provider: str
    status: str
    detail: str
    execution: Optional[int] = None
    execution_public_id: Optional[str] = None
    received_at: datetime
    processed_at: Optional[datetime] = None


def _path(trigger_id: Any) -> str:
    return f"{_ROOT}{UUID(str(trigger_id))}/"


def _edit(
    secret: Optional[str],
    rate_limit_per_hour: Optional[int],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        key: value
        for key, value in {
            "secret": secret,
            "rate_limit_per_hour": rate_limit_per_hour,
            "config": config,
        }.items()
        if value is not None
    }


class TriggersV1:
    def __init__(self, client: Any) -> None:
        self._c = client

    def list(self, *, agent_id: Any = None) -> List[TriggerV1]:
        """Your triggers, optionally restricted to an owned agent UUID."""
        params = {"task": str(UUID(str(agent_id)))} if agent_id is not None else {}
        return [
            TriggerV1.model_validate(row)
            for row in self._c._request("GET", _ROOT, params=params)
        ]

    def get(self, trigger_id: Any) -> TriggerV1:
        return TriggerV1.model_validate(self._c._request("GET", _path(trigger_id)))

    def create(
        self,
        agent_id: Any,
        *,
        idempotency_key: str,
        secret: str,
        rate_limit_per_hour: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TriggerV1:
        """Create a webhook. Reuse the key only for this same logical request.

        The required signing secret is write-only. Treat the returned URL as
        private. The agent needs a Live frozen version before an event can run.
        """
        return TriggerV1.model_validate(
            self._c._request(
                "POST",
                _ROOT,
                json={
                    "agent_task": str(UUID(str(agent_id))),
                    "kind": "webhook",
                    **_edit(secret, rate_limit_per_hour, config),
                },
                headers=_idempotency_headers(idempotency_key),
            )
        )

    def update(
        self,
        trigger_id: Any,
        *,
        idempotency_key: str,
        secret: Optional[str] = None,
        rate_limit_per_hour: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TriggerV1:
        """Edit settings. Config replaces the previous object; None leaves it alone.

        None leaves the secret alone. Empty secrets are rejected.
        """
        return TriggerV1.model_validate(
            self._c._request(
                "PATCH",
                _path(trigger_id),
                json=_edit(secret, rate_limit_per_hour, config),
                headers=_idempotency_headers(idempotency_key),
            )
        )

    def pause(self, trigger_id: Any, *, idempotency_key: str) -> TriggerV1:
        """Pause new events; they are acknowledged and discarded while paused."""
        return TriggerV1.model_validate(
            self._c._request(
                "POST",
                f"{_path(trigger_id)}pause/",
                json={},
                headers=_idempotency_headers(idempotency_key),
            )
        )

    def activate(self, trigger_id: Any, *, idempotency_key: str) -> TriggerV1:
        """Resume new events without resetting the current hourly rate counter."""
        return TriggerV1.model_validate(
            self._c._request(
                "POST",
                f"{_path(trigger_id)}activate/",
                json={},
                headers=_idempotency_headers(idempotency_key),
            )
        )

    def delete(self, trigger_id: Any, *, idempotency_key: str) -> None:
        """Delete the binding and private receipts; admitted runs continue."""
        self._c._request(
            "DELETE",
            _path(trigger_id),
            headers=_idempotency_headers(idempotency_key),
            expected_status=204,
        )

    def events(self, trigger_id: Any) -> List[TriggerEventV1]:
        """Latest 50 receipts, newest first, without raw private payloads."""
        return [
            TriggerEventV1.model_validate(row)
            for row in self._c._request("GET", f"{_path(trigger_id)}events/")
        ]


class AsyncTriggersV1:
    def __init__(self, client: Any) -> None:
        self._c = client

    async def list(self, *, agent_id: Any = None) -> List[TriggerV1]:
        params = {"task": str(UUID(str(agent_id)))} if agent_id is not None else {}
        return [
            TriggerV1.model_validate(row)
            for row in await self._c._request("GET", _ROOT, params=params)
        ]

    async def get(self, trigger_id: Any) -> TriggerV1:
        return TriggerV1.model_validate(
            await self._c._request("GET", _path(trigger_id))
        )

    async def create(
        self,
        agent_id: Any,
        *,
        idempotency_key: str,
        secret: str,
        rate_limit_per_hour: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TriggerV1:
        return TriggerV1.model_validate(
            await self._c._request(
                "POST",
                _ROOT,
                json={
                    "agent_task": str(UUID(str(agent_id))),
                    "kind": "webhook",
                    **_edit(secret, rate_limit_per_hour, config),
                },
                headers=_idempotency_headers(idempotency_key),
            )
        )

    async def update(
        self,
        trigger_id: Any,
        *,
        idempotency_key: str,
        secret: Optional[str] = None,
        rate_limit_per_hour: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TriggerV1:
        return TriggerV1.model_validate(
            await self._c._request(
                "PATCH",
                _path(trigger_id),
                json=_edit(secret, rate_limit_per_hour, config),
                headers=_idempotency_headers(idempotency_key),
            )
        )

    async def pause(self, trigger_id: Any, *, idempotency_key: str) -> TriggerV1:
        return TriggerV1.model_validate(
            await self._c._request(
                "POST",
                f"{_path(trigger_id)}pause/",
                json={},
                headers=_idempotency_headers(idempotency_key),
            )
        )

    async def activate(self, trigger_id: Any, *, idempotency_key: str) -> TriggerV1:
        return TriggerV1.model_validate(
            await self._c._request(
                "POST",
                f"{_path(trigger_id)}activate/",
                json={},
                headers=_idempotency_headers(idempotency_key),
            )
        )

    async def delete(self, trigger_id: Any, *, idempotency_key: str) -> None:
        await self._c._request(
            "DELETE",
            _path(trigger_id),
            headers=_idempotency_headers(idempotency_key),
            expected_status=204,
        )

    async def events(self, trigger_id: Any) -> List[TriggerEventV1]:
        return [
            TriggerEventV1.model_validate(row)
            for row in await self._c._request("GET", f"{_path(trigger_id)}events/")
        ]
