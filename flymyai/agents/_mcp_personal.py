"""Exact-connection email invitations; the shared bounded transport never retries."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from ._mcp_sharing import (
    AsyncMcpTeams,
    McpTeamDevice,
    McpTeamPage,
    McpTeams,
    _TeamModel,
)

_ROOT = "/api/v1/mcp-shares/"
_INVITES = "/api/v1/mcp-share-invitations/"


class McpSharePerson(_TeamModel):
    id: int
    username: str


class McpShare(_TeamModel):
    id: UUID
    connection_id: UUID
    workspace_id: UUID
    label: str
    source_id: UUID
    source_kind: str
    owner: McpSharePerson
    recipient: McpSharePerson
    state: str
    expires_at: datetime
    payer: str
    billing_mode: str = "recipient_pays"
    provider_billing: str = "credential_owner"


class McpPersonalInvitation(_TeamModel):
    id: UUID
    email: str
    state: str
    label: str
    owner: McpSharePerson
    recipient: Optional[McpSharePerson] = None
    grant_id: Optional[UUID] = None
    access_days: int
    expires_at: datetime
    access_expires_at: Optional[datetime] = None


def _path(identifier: Any, suffix: str = "", *, invitation: bool = False) -> str:
    return (
        (_INVITES if invitation else _ROOT) + str(UUID(str(identifier))) + "/" + suffix
    )


def _scope(scope: str) -> Dict[str, str]:
    if scope not in ("sent", "received"):
        raise ValueError("scope must be sent or received")
    return {"scope": scope}


class McpShares:
    """A verified personal API key, explicit payer consent and caller-driven pages."""

    def __init__(self, client: Any) -> None:
        self._transport = McpTeams(client)

    def list(
        self, *, scope: str = "received", cursor: Optional[str] = None
    ) -> McpTeamPage[McpShare]:
        return McpTeamPage[McpShare].model_validate(
            self._transport._request("GET", _ROOT, query=_scope(scope), cursor=cursor)
        )

    def invitations(
        self, *, scope: str = "received", cursor: Optional[str] = None
    ) -> McpTeamPage[McpPersonalInvitation]:
        return McpTeamPage[McpPersonalInvitation].model_validate(
            self._transport._request(
                "GET", _INVITES, query=_scope(scope), cursor=cursor
            )
        )

    def invite(
        self,
        source_id: Any,
        email: str,
        *,
        accept_sharing: bool,
        idempotency_key: str,
        access_days: int = 7,
    ) -> McpPersonalInvitation:
        return McpPersonalInvitation.model_validate(
            self._transport._request(
                "POST",
                _INVITES,
                {
                    "source_id": str(UUID(str(source_id))),
                    "email": email,
                    "accept_sharing": accept_sharing,
                    "access_days": access_days,
                },
                key=idempotency_key,
            )
        )

    def accept(self, invitation_id: Any, *, accept_billing: bool) -> McpShare:
        return McpShare.model_validate(
            self._transport._request(
                "POST", _path(invitation_id, "accept/", invitation=True), {"accept_billing": accept_billing}
            )
        )

    def cancel(self, invitation_id: Any) -> McpPersonalInvitation:
        return McpPersonalInvitation.model_validate(
            self._transport._request("DELETE", _path(invitation_id, invitation=True))
        )

    def revoke(self, share_id: Any) -> McpShare:
        return McpShare.model_validate(
            self._transport._request("DELETE", _path(share_id))
        )

    def tools(self, share_id: Any, *, cursor: Optional[str] = None) -> Dict[str, Any]:
        return self._transport._request("GET", _path(share_id, "tools/"), cursor=cursor)

    def call(
        self,
        share_id: Any,
        name: str,
        arguments: Dict[str, Any],
        *,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        return self._transport._request(
            "POST",
            _path(share_id, "call/"),
            {"name": name, "arguments": arguments},
            key=idempotency_key,
        )

    def devices(
        self, share_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamDevice]:
        return McpTeamPage[McpTeamDevice].model_validate(
            self._transport._request("GET", _path(share_id, "devices/"), cursor=cursor)
        )

    def create_device(
        self, share_id: Any, label: str, *, idempotency_key: str, access_days: int = 7
    ) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            self._transport._request(
                "POST",
                _path(share_id, "devices/"),
                {"label": label, "access_days": access_days},
                key=idempotency_key,
            )
        )

    def revoke_device(self, share_id: Any, device_id: Any) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            self._transport._request(
                "DELETE", _path(share_id, f"devices/{UUID(str(device_id))}/")
            )
        )

    def activity(
        self, share_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            self._transport._request("GET", _path(share_id, "activity/"), cursor=cursor)
        )


class AsyncMcpShares:
    """Async counterpart with identical email, grant and resource boundaries."""

    def __init__(self, client: Any) -> None:
        self._transport = AsyncMcpTeams(client)

    async def list(
        self, *, scope: str = "received", cursor: Optional[str] = None
    ) -> McpTeamPage[McpShare]:
        return McpTeamPage[McpShare].model_validate(
            await self._transport._request(
                "GET", _ROOT, query=_scope(scope), cursor=cursor
            )
        )

    async def invitations(
        self, *, scope: str = "received", cursor: Optional[str] = None
    ) -> McpTeamPage[McpPersonalInvitation]:
        return McpTeamPage[McpPersonalInvitation].model_validate(
            await self._transport._request(
                "GET", _INVITES, query=_scope(scope), cursor=cursor
            )
        )

    async def invite(
        self,
        source_id: Any,
        email: str,
        *,
        accept_sharing: bool,
        idempotency_key: str,
        access_days: int = 7,
    ) -> McpPersonalInvitation:
        return McpPersonalInvitation.model_validate(
            await self._transport._request(
                "POST",
                _INVITES,
                {
                    "source_id": str(UUID(str(source_id))),
                    "email": email,
                    "accept_sharing": accept_sharing,
                    "access_days": access_days,
                },
                key=idempotency_key,
            )
        )

    async def accept(self, invitation_id: Any, *, accept_billing: bool) -> McpShare:
        return McpShare.model_validate(
            await self._transport._request(
                "POST", _path(invitation_id, "accept/", invitation=True), {"accept_billing": accept_billing}
            )
        )

    async def cancel(self, invitation_id: Any) -> McpPersonalInvitation:
        return McpPersonalInvitation.model_validate(
            await self._transport._request(
                "DELETE", _path(invitation_id, invitation=True)
            )
        )

    async def revoke(self, share_id: Any) -> McpShare:
        return McpShare.model_validate(
            await self._transport._request("DELETE", _path(share_id))
        )

    async def tools(
        self, share_id: Any, *, cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._transport._request(
            "GET", _path(share_id, "tools/"), cursor=cursor
        )

    async def call(
        self,
        share_id: Any,
        name: str,
        arguments: Dict[str, Any],
        *,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        return await self._transport._request(
            "POST",
            _path(share_id, "call/"),
            {"name": name, "arguments": arguments},
            key=idempotency_key,
        )

    async def devices(
        self, share_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamDevice]:
        return McpTeamPage[McpTeamDevice].model_validate(
            await self._transport._request(
                "GET", _path(share_id, "devices/"), cursor=cursor
            )
        )

    async def create_device(
        self, share_id: Any, label: str, *, idempotency_key: str, access_days: int = 7
    ) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            await self._transport._request(
                "POST",
                _path(share_id, "devices/"),
                {"label": label, "access_days": access_days},
                key=idempotency_key,
            )
        )

    async def revoke_device(self, share_id: Any, device_id: Any) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            await self._transport._request(
                "DELETE", _path(share_id, f"devices/{UUID(str(device_id))}/")
            )
        )

    async def activity(
        self, share_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            await self._transport._request(
                "GET", _path(share_id, "activity/"), cursor=cursor
            )
        )
