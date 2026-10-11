"""Bounded owner GPU diagnostics. Available on the UAT candidate endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

MAX_RESPONSE_BYTES = 64 * 1024


class ComputeLogTail(BaseModel):
    """Recent logs; optional source=system identifies RunPod image/startup progress."""

    model_config = ConfigDict(extra="allow")
    lines: List[str] = Field(default_factory=list)
    bytes: int = 0
    truncated: bool = False
    retention: str = "provider_tail"


class ComputeLogs(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = Field(alias="schema")
    server_id: str
    status: str
    availability: str
    ready_code: Optional[int] = None
    health: Optional[Dict[str, Any]] = None
    logs: ComputeLogTail = Field(default_factory=ComputeLogTail)
    error: Optional[Dict[str, str]] = None


def _args(server_id: str, lines: int, max_bytes: int) -> tuple[str, Dict[str, int]]:
    server = str(UUID(str(server_id)))
    if type(lines) is not int or not 1 <= lines <= 200:
        raise ValueError("lines must be an integer from 1 to 200")
    if type(max_bytes) is not int or not 256 <= max_bytes <= 16384:
        raise ValueError("max_bytes must be an integer from 256 to 16384")
    return f"/api/v1/agents/compute/servers/{server}/logs/", {
        "lines": lines,
        "max_bytes": max_bytes,
    }


def _append(raw: bytearray, chunk: bytes) -> None:
    if len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
        from flymyai.agents._client import FlyMyAIAgentError

        raise FlyMyAIAgentError(
            "GPU diagnostic response exceeds 64 KiB", status_code=502
        )
    raw.extend(chunk)


def _identity(response: httpx.Response) -> None:
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        from flymyai.agents._client import FlyMyAIAgentError

        raise FlyMyAIAgentError(
            "Compressed GPU diagnostic response refused", status_code=502
        )


def _parse(response: httpx.Response, raw: bytearray) -> ComputeLogs:
    from flymyai.agents._client import _raise_for_status

    _raise_for_status(
        httpx.Response(
            response.status_code, content=bytes(raw), request=response.request
        )
    )
    return ComputeLogs.model_validate_json(raw)


class Compute:
    def __init__(self, client: SyncAgentClient) -> None:
        self._client = client

    def logs(
        self, server_id: str, *, lines: int = 100, max_bytes: int = 8192
    ) -> ComputeLogs:
        """Read owner-only health/logs without waking or extending a GPU allocation."""
        path, params = _args(server_id, lines, max_bytes)
        raw = bytearray()
        with self._client._http.stream(
            "GET", path, params=params, headers={"Accept-Encoding": "identity"}
        ) as response:
            _identity(response)
            for chunk in response.iter_bytes(chunk_size=4096):
                _append(raw, chunk)
            return _parse(response, raw)


class AsyncCompute:
    def __init__(self, client: AsyncAgentClient) -> None:
        self._client = client

    async def logs(
        self, server_id: str, *, lines: int = 100, max_bytes: int = 8192
    ) -> ComputeLogs:
        """Read owner-only health/logs without waking or extending a GPU allocation."""
        path, params = _args(server_id, lines, max_bytes)
        raw = bytearray()
        async with self._client._http.stream(
            "GET", path, params=params, headers={"Accept-Encoding": "identity"}
        ) as response:
            _identity(response)
            async for chunk in response.aiter_bytes(chunk_size=4096):
                _append(raw, chunk)
            return _parse(response, raw)
