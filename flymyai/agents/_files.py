"""Bounded owner Files V2 reads. No tree accumulation or effect retries."""
from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

_BASE = "/api/v1/agents/files"
_PAGE_BYTES = 512 * 1024
_RANGE_BYTES = 65536
_WORKSPACE = r"ws_[0-7][0-9A-HJKMNP-TV-Z]{25}"
_ARTIFACT = r"af_[0-7][0-9A-HJKMNP-TV-Z]{25}"


class FilesContractUnsupportedError(RuntimeError):
    """Server did not supply the bounded Files contract; no legacy fallback."""


class FileMetadata(BaseModel):
    id: str = Field(pattern="^" + _ARTIFACT + "$")
    name: str
    version: int = Field(ge=1, strict=True)
    size: int = Field(ge=0, strict=True)
    sha256: Optional[str] = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str
    executable: bool
    hidden: bool
    deleted: bool
    created_at: str
    updated_at: str
    origin: Dict[str, Any]

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"


class FilePage(BaseModel):
    workspace: str = Field(pattern="^" + _WORKSPACE + "$")
    revision: int = Field(ge=0, strict=True)
    total: int = Field(ge=0, strict=True)
    files: List[FileMetadata] = Field(max_length=50)
    next_cursor: Optional[str] = Field(max_length=128)


class FileWorkspace(BaseModel):
    workspace: str = Field(pattern="^" + _WORKSPACE + "$")
    kind: Literal["task", "library"]
    revision: int = Field(ge=0, strict=True)
    agent_uuid: Optional[UUID]
    name: str = Field(max_length=255)
    role: Literal["admin"]


class FileWorkspacePage(BaseModel):
    workspaces: List[FileWorkspace] = Field(max_length=50)
    next_cursor: Optional[str] = Field(pattern="^" + _WORKSPACE + "$")


class FileWorkspaceSubject(BaseModel):
    kind: Literal["task", "group"]
    id: UUID
    name: str = Field(max_length=255)


class FileWorkspaceSubjectPage(BaseModel):
    subjects: List[FileWorkspaceSubject] = Field(max_length=50)
    next_cursor: Optional[UUID]


class FileRange(BaseModel):
    """One byte window; a window may split UTF-8 or JSON tokens."""

    ref: str
    offset: int
    total_bytes: int
    sha256: str
    content_type: str
    content: bytes


def _ref(value: str, pattern: str) -> str:
    if not isinstance(value, str) or len(value) > 128 or re.fullmatch(pattern, value) is None:
        raise ValueError("Use the exact canonical reference returned by Files V2.")
    return value


def _page_params(limit: int, cursor: Optional[str]) -> Dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer from 1 to 50.")
    params: Dict[str, Any] = {"limit": limit}
    if cursor is not None:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 128:
            raise ValueError("Pass one cursor from the preceding page.")
        params["cursor"] = cursor
    return params


def _workspace_params(workspace: Optional[str]) -> Dict[str, str]:
    return {} if workspace is None else {"workspace": _ref(workspace, _WORKSPACE)}


def _list_params(workspace: Optional[str], agent_uuid: Optional[str],
                 limit: int, cursor: Optional[str], prefix: Optional[str]) -> Dict[str, Any]:
    if workspace is not None and agent_uuid is not None:
        raise ValueError("Choose workspace or agent_uuid, not both.")
    params = _page_params(limit, cursor)
    if workspace is not None:
        params["workspace"] = _ref(workspace, _WORKSPACE)
    if agent_uuid is not None:
        params["agent_uuid"] = str(UUID(str(agent_uuid)))
    if prefix is not None:
        if not isinstance(prefix, str) or len(prefix.encode("utf-8")) > 1024:
            raise ValueError("prefix must fit in 1024 UTF-8 bytes.")
        params["prefix"] = prefix
    return params


def _model(response: Any, model: Any) -> Any:
    try:
        return model.model_validate(response.json())
    except ValueError as exc:
        raise FilesContractUnsupportedError("Server did not return a bounded Files page/metadata response.") from exc


def _range_request(meta: FileMetadata, offset: int, limit: int) -> Dict[str, str]:
    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= _RANGE_BYTES:
        raise ValueError("Request an explicit offset and 1..65536 bytes.")
    if offset >= meta.size or meta.sha256 is None:
        raise ValueError("No readable bytes at that offset; inspect stat() for size and availability.")
    return {"Range": f"bytes={offset}-{min(meta.size, offset + limit) - 1}"}


def _range_result(response: Any, meta: FileMetadata, offset: int, limit: int) -> FileRange:
    end = min(meta.size, offset + limit) - 1
    if (response.status_code != 206
            or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{meta.size}"
            or response.headers.get("X-Artifact-Ref") != meta.ref
            or response.headers.get("X-Content-SHA256") != meta.sha256
            or len(response.content) != end - offset + 1):
        raise FilesContractUnsupportedError("Server did not return the exact pinned Files byte range.")
    if offset == 0 and end + 1 == meta.size and hashlib.sha256(response.content).hexdigest() != meta.sha256:
        raise FilesContractUnsupportedError("Complete artifact digest mismatch.")
    return FileRange(ref=meta.ref, offset=offset, total_bytes=meta.size,
                     sha256=meta.sha256, content_type=meta.content_type, content=response.content)


class Files:
    """One canonical owner library/task model. Each call reads one page/range.

    list() bootstraps the owner's library; list(agent_uuid=...) bootstraps that
    exact owner's personal agent workspace. workspaces() discovers existing
    persistent workspaces. Grant them with client.workspace_grants; selections
    are not authority and admission rechecks grants/leases. No write API or
    automatic retry is introduced here.
    """

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def _get(self, path: str, *, max_bytes: int = _PAGE_BYTES, **kwargs: Any) -> Any:
        from flymyai.agents._client import RunObservationUnsupportedError
        try:
            return self._c._bounded_get(_BASE + path, max_bytes=max_bytes, **kwargs)
        except RunObservationUnsupportedError as exc:
            raise FilesContractUnsupportedError("Bounded Files read unavailable; no legacy fallback.") from exc

    def workspaces(self, *, limit: int = 20, cursor: Optional[str] = None) -> FileWorkspacePage:
        params = _page_params(limit, cursor)
        if cursor is not None:
            params["cursor"] = _ref(cursor, _WORKSPACE)
        return _model(self._get("/workspaces", params=params), FileWorkspacePage)

    def subjects(self, kind: Literal["task", "group"], *, limit: int = 20,
                 cursor: Optional[str] = None) -> FileWorkspaceSubjectPage:
        if kind not in ("task", "group"):
            raise ValueError("kind must be task or group.")
        params = _page_params(limit, cursor)
        params["kind"] = kind
        if cursor is not None:
            params["cursor"] = str(UUID(cursor))
        return _model(self._get("/workspace-subjects", params=params), FileWorkspaceSubjectPage)

    def list(self, *, workspace: Optional[str] = None, agent_uuid: Optional[str] = None,
             limit: int = 20, cursor: Optional[str] = None, prefix: Optional[str] = None) -> FilePage:
        params = _list_params(workspace, agent_uuid, limit, cursor, prefix)
        return _model(self._get("/page", params=params), FilePage)

    def stat(self, file: str, *, workspace: Optional[str] = None) -> FileMetadata:
        file = _ref(file, _ARTIFACT + r"(?:@v[1-9][0-9]*)?")
        meta = _model(self._get(f"/{file}/meta", max_bytes=_RANGE_BYTES,
                               params=_workspace_params(workspace)), FileMetadata)
        if file not in (meta.id, meta.ref):
            raise FilesContractUnsupportedError("Artifact metadata identity mismatch.")
        return meta

    def read(self, file: Union[str, FileMetadata], *, offset: int, limit: int,
             workspace: Optional[str] = None) -> FileRange:
        """Read <=64 KiB. A string ref first makes one bounded stat to pin a version.

        Pass FileMetadata from list/stat to keep later windows on the same version.
        Pass that page's workspace to select the exact binding of a shared artifact.
        Empty files need no content request. Conflicts/expired versions propagate;
        refresh metadata explicitly, never silently switch versions mid-download.
        """
        meta = file if isinstance(file, FileMetadata) else self.stat(file, workspace=workspace)
        headers = _range_request(meta, offset, limit)
        response = self._get(f"/{meta.ref}/content", max_bytes=_RANGE_BYTES,
                             params={"view": "bounded_v1", **_workspace_params(workspace)}, headers=headers)
        return _range_result(response, meta, offset, limit)


class AsyncFiles:
    """Async equivalent of Files, with the same finite page/range contract."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def _get(self, path: str, *, max_bytes: int = _PAGE_BYTES, **kwargs: Any) -> Any:
        from flymyai.agents._client import RunObservationUnsupportedError
        try:
            return await self._c._bounded_get(_BASE + path, max_bytes=max_bytes, **kwargs)
        except RunObservationUnsupportedError as exc:
            raise FilesContractUnsupportedError("Bounded Files read unavailable; no legacy fallback.") from exc

    async def workspaces(self, *, limit: int = 20, cursor: Optional[str] = None) -> FileWorkspacePage:
        params = _page_params(limit, cursor)
        if cursor is not None:
            params["cursor"] = _ref(cursor, _WORKSPACE)
        return _model(await self._get("/workspaces", params=params), FileWorkspacePage)

    async def subjects(self, kind: Literal["task", "group"], *, limit: int = 20,
                       cursor: Optional[str] = None) -> FileWorkspaceSubjectPage:
        if kind not in ("task", "group"):
            raise ValueError("kind must be task or group.")
        params = _page_params(limit, cursor)
        params["kind"] = kind
        if cursor is not None:
            params["cursor"] = str(UUID(cursor))
        return _model(await self._get("/workspace-subjects", params=params), FileWorkspaceSubjectPage)

    async def list(self, *, workspace: Optional[str] = None, agent_uuid: Optional[str] = None,
                   limit: int = 20, cursor: Optional[str] = None, prefix: Optional[str] = None) -> FilePage:
        params = _list_params(workspace, agent_uuid, limit, cursor, prefix)
        return _model(await self._get("/page", params=params), FilePage)

    async def stat(self, file: str, *, workspace: Optional[str] = None) -> FileMetadata:
        file = _ref(file, _ARTIFACT + r"(?:@v[1-9][0-9]*)?")
        meta = _model(await self._get(f"/{file}/meta", max_bytes=_RANGE_BYTES,
                                     params=_workspace_params(workspace)), FileMetadata)
        if file not in (meta.id, meta.ref):
            raise FilesContractUnsupportedError("Artifact metadata identity mismatch.")
        return meta

    async def read(self, file: Union[str, FileMetadata], *, offset: int, limit: int,
                   workspace: Optional[str] = None) -> FileRange:
        """Read one explicit pinned window, with a bounded stat for string refs."""
        meta = file if isinstance(file, FileMetadata) else await self.stat(file, workspace=workspace)
        headers = _range_request(meta, offset, limit)
        response = await self._get(f"/{meta.ref}/content", max_bytes=_RANGE_BYTES,
                                   params={"view": "bounded_v1", **_workspace_params(workspace)}, headers=headers)
        return _range_result(response, meta, offset, limit)
