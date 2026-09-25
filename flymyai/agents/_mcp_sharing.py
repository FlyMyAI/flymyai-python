"""Optional team management. Bounded pages, no automatic retries or secret cache."""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

T = TypeVar("T")
_ROOT = "/api/v1/mcp-teams/"
_MAX_RESPONSE = 256 * 1024


class _TeamModel(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)


class McpTeamPage(_TeamModel, Generic[T]):
    items: List[T] = Field(max_length=25)
    next_cursor: Optional[str] = Field(default=None, max_length=64)


class McpTeam(_TeamModel):
    id: UUID
    username: str
    owner: Optional[int]
    payer: str
    role: str
    member_id: UUID
    active: bool
    revision: int
    expires_at: datetime
    call_limit: int
    spend_limit: Decimal
    calls: int
    spent: Decimal
    reserved: Decimal
    team_user_id: int
    pending_transfer: Optional[Dict[str, Any]] = None


class McpTeamConnection(_TeamModel):
    id: UUID
    label: str
    owner_id: Optional[int]
    source_kind: str
    source_id: str
    shared: bool
    active: bool
    actions: List[str] = Field(max_length=16)
    revision: int
    available: bool
    unavailable_reason: Optional[str] = None
    owner_name: Optional[str] = None
    calls: int = 0
    users: int = 0
    last_used_at: Optional[datetime] = None


class McpTeamDevice(_TeamModel):
    id: UUID
    member_id: UUID
    label: str
    prefix: str
    expires_at: datetime
    revoked_at: Optional[datetime] = None
    token: Optional[SecretStr] = Field(default=None, repr=False)


class McpTeamInvite(_TeamModel):
    id: UUID
    role: str
    state: str
    claimant: Optional[str] = None
    expires_at: datetime
    call_limit: int
    spend_limit: Decimal
    email: str = ""
    access_days: int = 7
    token: Optional[SecretStr] = Field(default=None, repr=False)


class McpTeamError(Exception):
    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(f"MCP team request failed ({status_code}): {code}")


def _team_path(team_id: Any, suffix: str = "") -> str:
    return _ROOT + str(UUID(str(team_id))) + "/" + suffix


def _page_query(cursor: Optional[str]) -> Dict[str, str]:
    if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 64):
        raise ValueError("cursor must contain at most 64 characters")
    return {"cursor": cursor} if cursor else {}


def _options(
    data: Any,
    key: Optional[str],
    cursor: Optional[str],
    query: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Content-Type": "application/json",
    }
    if key is not None:
        if (
            not isinstance(key, str)
            or not 8 <= len(key) <= 128
            or any(ord(c) < 33 or ord(c) > 126 for c in key)
        ):
            raise ValueError(
                "idempotency_key must be 8-128 printable ASCII characters without"
                " spaces"
            )
        headers["Idempotency-Key"] = key
    raw = json.dumps(data, separators=(",", ":")).encode() if data is not None else None
    if raw and len(raw) > 64 * 1024:
        raise ValueError("team request exceeds 64 KiB")
    return {
        "headers": headers,
        "content": raw,
        "params": {**(query or {}), **_page_query(cursor)},
        "follow_redirects": False,
        "timeout": 35,
    }


def _response_headers(response: Any) -> None:
    if response.headers.get("content-encoding", "identity") != "identity":
        raise McpTeamError(502, "unexpected_compression")
    length = response.headers.get("content-length")
    if length and (
        not length.isascii() or not length.isdigit() or int(length) > _MAX_RESPONSE
    ):
        raise McpTeamError(502, "response_too_large")


def _decode(response: Any, raw: bytearray) -> Any:
    try:
        payload = json.loads(raw) if raw else {}
    except (ValueError, RecursionError):
        raise McpTeamError(502, "invalid_response") from None
    if not response.is_success:
        code = payload.get("code") if isinstance(payload, dict) else None
        if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,64}", code):
            code = "request_rejected"
        raise McpTeamError(response.status_code, code)
    return payload


class McpTeams:
    """Use a verified personal API key. Device tokens are for MCP runtime only."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _request(
        self,
        method: str,
        path: str,
        data: Any = None,
        *,
        key: Optional[str] = None,
        cursor: Optional[str] = None,
        query: Optional[Dict[str, str]] = None,
    ) -> Any:
        with self._client._http.stream(
            method, path, **_options(data, key, cursor, query)
        ) as response:
            _response_headers(response)
            raw = bytearray()
            for chunk in response.iter_raw():
                if len(chunk) > _MAX_RESPONSE - len(raw):
                    raise McpTeamError(502, "response_too_large")
                raw.extend(chunk)
            return _decode(response, raw)

    def list(self, *, cursor: Optional[str] = None) -> McpTeamPage[McpTeam]:
        return McpTeamPage[McpTeam].model_validate(
            self._request("GET", _ROOT, cursor=cursor)
        )

    def get(self, team_id: Any) -> McpTeam:
        return McpTeam.model_validate(self._request("GET", _team_path(team_id)))

    def enable(
        self, username: str, *, create: bool = False, accept_billing: bool = False
    ) -> McpTeam:
        return McpTeam.model_validate(
            self._request(
                "POST",
                _ROOT,
                {
                    "username": username,
                    "create": create,
                    "accept_billing": accept_billing,
                },
            )
        )

    def update(self, team_id: Any, *, revision: int, **changes: Any) -> McpTeam:
        return McpTeam.model_validate(
            self._request(
                "PATCH", _team_path(team_id), {**changes, "revision": revision}
            )
        )

    def connections(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamConnection]:
        return McpTeamPage[McpTeamConnection].model_validate(
            self._request("GET", _team_path(team_id, "connections/"), cursor=cursor)
        )

    def add_connection(
        self,
        team_id: Any,
        *,
        source_kind: str,
        source_id: Any,
        label: str,
        actions: List[str],
        shared: bool = False,
        consent: bool = False,
    ) -> McpTeamConnection:
        return McpTeamConnection.model_validate(
            self._request(
                "POST",
                _team_path(team_id, "connections/"),
                {
                    "source_kind": source_kind,
                    "source_id": str(UUID(str(source_id))),
                    "label": label,
                    "actions": actions,
                    "shared": shared,
                    "consent": consent,
                },
            )
        )

    def share_connection(
        self,
        team_id: Any,
        connection_id: Any,
        *,
        revision: int,
        actions: List[str],
        shared: bool,
        consent: bool,
    ) -> McpTeamConnection:
        return McpTeamConnection.model_validate(
            self._request(
                "PATCH",
                _team_path(team_id, f"connections/{UUID(str(connection_id))}/"),
                {
                    "revision": revision,
                    "actions": actions,
                    "shared": shared,
                    "consent": consent,
                },
            )
        )

    def revoke_connection(self, team_id: Any, connection_id: Any) -> McpTeamConnection:
        return McpTeamConnection.model_validate(
            self._request(
                "DELETE",
                _team_path(team_id, f"connections/{UUID(str(connection_id))}/"),
            )
        )

    def invite(
        self,
        team_id: Any,
        *,
        idempotency_key: str,
        email: str = "",
        role: str = "member",
        access_days: int = 7,
        call_limit: int = 500,
        spend_limit: str = "5",
    ) -> McpTeamInvite:
        return McpTeamInvite.model_validate(
            self._request(
                "POST",
                _team_path(team_id, "invites/"),
                {
                    "email": email,
                    "role": role,
                    "access_days": access_days,
                    "call_limit": call_limit,
                    "spend_limit": spend_limit,
                },
                key=idempotency_key,
            )
        )

    def claim_invite(
        self, token: SecretStr, *, accept_team_scope: bool
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/mcp-team-invitations/claim/",
            {"token": token.get_secret_value(), "accept_team_scope": accept_team_scope},
        )

    def decide_invite(
        self, team_id: Any, invite_id: Any, *, approve: bool
    ) -> McpTeamInvite:
        return McpTeamInvite.model_validate(
            self._request(
                "POST",
                _team_path(team_id, f"invites/{UUID(str(invite_id))}/decision/"),
                {"approve": approve},
            )
        )

    def devices(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamDevice]:
        return McpTeamPage[McpTeamDevice].model_validate(
            self._request("GET", _team_path(team_id, "devices/"), cursor=cursor)
        )

    def create_device(
        self, team_id: Any, *, label: str, idempotency_key: str, access_days: int = 7
    ) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            self._request(
                "POST",
                _team_path(team_id, "devices/"),
                {"label": label, "access_days": access_days},
                key=idempotency_key,
            )
        )

    def revoke_device(self, team_id: Any, device_id: Any) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            self._request(
                "DELETE", _team_path(team_id, f"devices/{UUID(str(device_id))}/")
            )
        )

    def activity(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            self._request("GET", _team_path(team_id, "activity/"), cursor=cursor)
        )

    def capabilities(self) -> Dict[str, Any]:
        return self._request("GET", _ROOT + "capabilities/")

    def eligible(self, *, cursor: Optional[str] = None) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            self._request("GET", _ROOT + "eligible/", cursor=cursor)
        )

    def sources(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            self._request("GET", _team_path(team_id, "sources/"), cursor=cursor)
        )

    def members(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            self._request("GET", _team_path(team_id, "members/"), cursor=cursor)
        )

    def update_member(
        self, team_id: Any, member_id: Any, *, epoch: int, **changes: Any
    ) -> Dict[str, Any]:
        return self._request(
            "PATCH",
            _team_path(team_id, f"members/{UUID(str(member_id))}/"),
            {**changes, "epoch": epoch},
        )

    def remove_member(
        self, team_id: Any, member_id: Any, *, accept_team_scope: bool
    ) -> Dict[str, Any]:
        return self._request(
            "DELETE",
            _team_path(team_id, f"members/{UUID(str(member_id))}/"),
            {"accept_team_scope": accept_team_scope},
        )

    def invitations(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamInvite]:
        return McpTeamPage[McpTeamInvite].model_validate(
            self._request("GET", _team_path(team_id, "invites/"), cursor=cursor)
        )

    def preview_invite(self, token: SecretStr) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/mcp-team-invitations/preview/",
            {"token": token.get_secret_value()},
        )

    def usage(
        self, team_id: Any, *, group: str = "actor", cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        if group not in ("actor", "action"):
            raise ValueError("group must be actor or action")
        return McpTeamPage[Dict[str, Any]].model_validate(
            self._request(
                "GET",
                _team_path(team_id, "usage/"),
                cursor=cursor,
                query={"group": group},
            )
        )

    def propose_transfer(self, team_id: Any, member_id: Any) -> Dict[str, Any]:
        return self._request(
            "POST",
            _team_path(team_id, "transfer/"),
            {"recipient_id": str(UUID(str(member_id)))},
        )

    def accept_transfer(self, team_id: Any, *, accept_billing: bool) -> McpTeam:
        return McpTeam.model_validate(
            self._request(
                "POST",
                _team_path(team_id, "transfer/accept/"),
                {"accept_billing": accept_billing},
            )
        )

    def cancel_transfer(self, team_id: Any) -> Dict[str, Any]:
        return self._request("DELETE", _team_path(team_id, "transfer/"))


class AsyncMcpTeams:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def _request(
        self,
        method: str,
        path: str,
        data: Any = None,
        *,
        key: Optional[str] = None,
        cursor: Optional[str] = None,
        query: Optional[Dict[str, str]] = None,
    ) -> Any:
        async with self._client._http.stream(
            method, path, **_options(data, key, cursor, query)
        ) as response:
            _response_headers(response)
            raw = bytearray()
            async for chunk in response.aiter_raw():
                if len(chunk) > _MAX_RESPONSE - len(raw):
                    raise McpTeamError(502, "response_too_large")
                raw.extend(chunk)
            return _decode(response, raw)

    async def list(self, *, cursor: Optional[str] = None) -> McpTeamPage[McpTeam]:
        return McpTeamPage[McpTeam].model_validate(
            await self._request("GET", _ROOT, cursor=cursor)
        )

    async def get(self, team_id: Any) -> McpTeam:
        return McpTeam.model_validate(await self._request("GET", _team_path(team_id)))

    async def enable(
        self, username: str, *, create: bool = False, accept_billing: bool = False
    ) -> McpTeam:
        return McpTeam.model_validate(
            await self._request(
                "POST",
                _ROOT,
                {
                    "username": username,
                    "create": create,
                    "accept_billing": accept_billing,
                },
            )
        )

    async def update(self, team_id: Any, *, revision: int, **changes: Any) -> McpTeam:
        return McpTeam.model_validate(
            await self._request(
                "PATCH", _team_path(team_id), {**changes, "revision": revision}
            )
        )

    async def connections(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamConnection]:
        return McpTeamPage[McpTeamConnection].model_validate(
            await self._request(
                "GET", _team_path(team_id, "connections/"), cursor=cursor
            )
        )

    async def add_connection(
        self,
        team_id: Any,
        *,
        source_kind: str,
        source_id: Any,
        label: str,
        actions: List[str],
        shared: bool = False,
        consent: bool = False,
    ) -> McpTeamConnection:
        return McpTeamConnection.model_validate(
            await self._request(
                "POST",
                _team_path(team_id, "connections/"),
                {
                    "source_kind": source_kind,
                    "source_id": str(UUID(str(source_id))),
                    "label": label,
                    "actions": actions,
                    "shared": shared,
                    "consent": consent,
                },
            )
        )

    async def share_connection(
        self,
        team_id: Any,
        connection_id: Any,
        *,
        revision: int,
        actions: List[str],
        shared: bool,
        consent: bool,
    ) -> McpTeamConnection:
        return McpTeamConnection.model_validate(
            await self._request(
                "PATCH",
                _team_path(team_id, f"connections/{UUID(str(connection_id))}/"),
                {
                    "revision": revision,
                    "actions": actions,
                    "shared": shared,
                    "consent": consent,
                },
            )
        )

    async def revoke_connection(
        self, team_id: Any, connection_id: Any
    ) -> McpTeamConnection:
        return McpTeamConnection.model_validate(
            await self._request(
                "DELETE",
                _team_path(team_id, f"connections/{UUID(str(connection_id))}/"),
            )
        )

    async def invite(
        self,
        team_id: Any,
        *,
        idempotency_key: str,
        email: str = "",
        role: str = "member",
        access_days: int = 7,
        call_limit: int = 500,
        spend_limit: str = "5",
    ) -> McpTeamInvite:
        return McpTeamInvite.model_validate(
            await self._request(
                "POST",
                _team_path(team_id, "invites/"),
                {
                    "email": email,
                    "role": role,
                    "access_days": access_days,
                    "call_limit": call_limit,
                    "spend_limit": spend_limit,
                },
                key=idempotency_key,
            )
        )

    async def claim_invite(
        self, token: SecretStr, *, accept_team_scope: bool
    ) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/mcp-team-invitations/claim/",
            {"token": token.get_secret_value(), "accept_team_scope": accept_team_scope},
        )

    async def decide_invite(
        self, team_id: Any, invite_id: Any, *, approve: bool
    ) -> McpTeamInvite:
        return McpTeamInvite.model_validate(
            await self._request(
                "POST",
                _team_path(team_id, f"invites/{UUID(str(invite_id))}/decision/"),
                {"approve": approve},
            )
        )

    async def devices(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamDevice]:
        return McpTeamPage[McpTeamDevice].model_validate(
            await self._request("GET", _team_path(team_id, "devices/"), cursor=cursor)
        )

    async def create_device(
        self, team_id: Any, *, label: str, idempotency_key: str, access_days: int = 7
    ) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            await self._request(
                "POST",
                _team_path(team_id, "devices/"),
                {"label": label, "access_days": access_days},
                key=idempotency_key,
            )
        )

    async def revoke_device(self, team_id: Any, device_id: Any) -> McpTeamDevice:
        return McpTeamDevice.model_validate(
            await self._request(
                "DELETE", _team_path(team_id, f"devices/{UUID(str(device_id))}/")
            )
        )

    async def activity(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            await self._request("GET", _team_path(team_id, "activity/"), cursor=cursor)
        )

    async def capabilities(self) -> Dict[str, Any]:
        return await self._request("GET", _ROOT + "capabilities/")

    async def eligible(
        self, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            await self._request("GET", _ROOT + "eligible/", cursor=cursor)
        )

    async def sources(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            await self._request("GET", _team_path(team_id, "sources/"), cursor=cursor)
        )

    async def members(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        return McpTeamPage[Dict[str, Any]].model_validate(
            await self._request("GET", _team_path(team_id, "members/"), cursor=cursor)
        )

    async def update_member(
        self, team_id: Any, member_id: Any, *, epoch: int, **changes: Any
    ) -> Dict[str, Any]:
        return await self._request(
            "PATCH",
            _team_path(team_id, f"members/{UUID(str(member_id))}/"),
            {**changes, "epoch": epoch},
        )

    async def remove_member(
        self, team_id: Any, member_id: Any, *, accept_team_scope: bool
    ) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            _team_path(team_id, f"members/{UUID(str(member_id))}/"),
            {"accept_team_scope": accept_team_scope},
        )

    async def invitations(
        self, team_id: Any, *, cursor: Optional[str] = None
    ) -> McpTeamPage[McpTeamInvite]:
        return McpTeamPage[McpTeamInvite].model_validate(
            await self._request("GET", _team_path(team_id, "invites/"), cursor=cursor)
        )

    async def preview_invite(self, token: SecretStr) -> Dict[str, Any]:
        return await self._request(
            "POST",
            "/api/v1/mcp-team-invitations/preview/",
            {"token": token.get_secret_value()},
        )

    async def usage(
        self, team_id: Any, *, group: str = "actor", cursor: Optional[str] = None
    ) -> McpTeamPage[Dict[str, Any]]:
        if group not in ("actor", "action"):
            raise ValueError("group must be actor or action")
        return McpTeamPage[Dict[str, Any]].model_validate(
            await self._request(
                "GET",
                _team_path(team_id, "usage/"),
                cursor=cursor,
                query={"group": group},
            )
        )

    async def propose_transfer(self, team_id: Any, member_id: Any) -> Dict[str, Any]:
        return await self._request(
            "POST",
            _team_path(team_id, "transfer/"),
            {"recipient_id": str(UUID(str(member_id)))},
        )

    async def accept_transfer(self, team_id: Any, *, accept_billing: bool) -> McpTeam:
        return McpTeam.model_validate(
            await self._request(
                "POST",
                _team_path(team_id, "transfer/accept/"),
                {"accept_billing": accept_billing},
            )
        )

    async def cancel_transfer(self, team_id: Any) -> Dict[str, Any]:
        return await self._request("DELETE", _team_path(team_id, "transfer/"))
