"""Frontend artifacts (``flymy.artifact.v1``): small web pages, presentations and mini
games with immutable versions, shared by link or with people by name, cloned into
your own copy.

The routes live on the agents host under ``/api/v1/artifacts/``. This is the v1
contract and it stays: a breaking change would ship as new methods next to these,
never in place. Unknown answer fields are kept on the models (``extra="allow"``).
"""

from __future__ import annotations

import base64
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from urllib.parse import quote
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

_ROOT = "/api/v1/artifacts/"
_HANDLE = re.compile(r"[A-Za-z0-9_-]{8,64}")
_TEXT_SUFFIXES = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".mjs",
    ".json",
    ".txt",
    ".md",
    ".svg",
    ".xml",
    ".csv",
    ".map",
    ".glsl",
    ".frag",
    ".vert",
    ".wgsl",
    ".ts",
}

T = TypeVar("T")


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)


class ArtifactAuthor(_Model):
    username: str


class ArtifactForkedFrom(_Model):
    name: str
    version: int
    url: Optional[str] = None


class ArtifactSummary(_Model):
    id: UUID
    name: str
    role: str  # owner, editor, member or viewer
    live_version: Optional[int] = None
    latest_version: int
    visibility: str  # private, link or public
    with_sources: bool
    clone_count: int = 0
    forked_from: Optional[ArtifactForkedFrom] = None
    owner_url: Optional[str] = None
    share_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ArtifactFrame(_Model):
    """A sandboxed frame URL, valid 8 hours."""

    url: str
    expires_at: datetime


class ArtifactView(ArtifactFrame):
    """The frame URL of one version (``view``)."""

    version: int
    entry: str


class ArtifactVersion(_Model):
    version: int
    entry: str
    message: Optional[str] = None
    files_count: int
    total_bytes: int
    live: bool
    created_at: datetime


class ArtifactLiveVersion(ArtifactVersion):
    view: Optional[ArtifactFrame] = None


class ArtifactShare(_Model):
    visibility: str
    with_sources: bool
    url: Optional[str] = None


class Artifact(ArtifactSummary):
    schema_: str = Field(alias="schema")
    description: str = ""
    author: ArtifactAuthor
    live: Optional[ArtifactLiveVersion] = None
    versions: List[ArtifactVersion] = Field(default_factory=list)
    share: Optional[ArtifactShare] = None


class ArtifactPlan(_Model):
    version: int
    entry: str
    files_count: int
    total_bytes: int
    new_bytes: int
    added: List[str] = Field(default_factory=list)
    changed: List[str] = Field(default_factory=list)
    removed: List[str] = Field(default_factory=list)
    no_change: bool = False
    live: Optional[bool] = None


class ArtifactReceipt(_Model):
    """The answer to a write: the artifact and the version it made."""

    artifact: ArtifactSummary
    version: ArtifactPlan


class ArtifactDryRun(_Model):
    """A dry run's plan; nothing changed."""

    dry_run: bool = True
    plan: ArtifactPlan


class ArtifactPage(_Model, Generic[T]):
    results: List[T]
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None


class ArtifactFile(_Model):
    path: str
    bytes: int
    sha256: str
    content_type: str


class ArtifactFiles(_Model):
    version: int
    entry: str
    files: List[ArtifactFile]


class ArtifactFileText(ArtifactFile):
    version: int
    content: str
    offset: int
    next_offset: Optional[int] = None
    truncated: bool


class ArtifactClone(_Model):
    from_version: int
    cloned_at: datetime
    clone: Optional[Dict[str, Any]] = None  # name, author, url - public clones only


class ArtifactLineage(_Model):
    forked_from: Optional[ArtifactForkedFrom] = None
    clone_count: int
    clones: ArtifactPage[ArtifactClone]


class ArtifactEvent(_Model):
    kind: str
    actor: Optional[str] = None
    version: Optional[int] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ArtifactMember(_Model):
    username: str
    role: str  # view or edit
    added_at: datetime
    email: Optional[str] = None  # the owner sees it


class ArtifactMembers(_Model):
    role: str
    owner: ArtifactAuthor
    members: List[ArtifactMember]
    max_members: int


class ArtifactMemberReceipt(_Model):
    member: ArtifactMember
    changed: bool


class ArtifactsStatus(_Model):
    enabled: bool
    limits: Dict[str, int] = Field(default_factory=dict)


def artifact_file(path: str, data: Union[str, bytes]) -> Dict[str, str]:
    """One file for ``create`` or ``publish``: text as ``content``, bytes as base64."""

    if isinstance(data, str):
        return {"path": path, "content": data}
    return {"path": path, "content_base64": base64.b64encode(data).decode("ascii")}


def artifact_files_from_directory(root: Union[str, Path]) -> List[Dict[str, str]]:
    """Every file under ``root`` as artifact files, paths relative to ``root``
    (a built site, an unpacked Claude artifact, a folder of a game)."""

    base = Path(root)
    files = []
    for item in sorted(base.rglob("*")):
        if not item.is_file() or any(
            part.startswith(".") for part in item.relative_to(base).parts
        ):
            continue
        path = item.relative_to(base).as_posix()
        raw = item.read_bytes()
        guessed = mimetypes.guess_type(path)[0] or ""
        if item.suffix.lower() in _TEXT_SUFFIXES or guessed.startswith("text/"):
            try:
                files.append(artifact_file(path, raw.decode("utf-8")))
                continue
            except UnicodeDecodeError:
                pass
        files.append(artifact_file(path, raw))
    return files


def _id(artifact_id: Any) -> str:
    return str(UUID(str(artifact_id)))


def _handle(share_link: str) -> str:
    """A share handle from ``https://.../artifacts/s/<handle>`` or the handle itself."""

    tail = str(share_link).rstrip("/").rsplit("/", 1)[-1]
    if not _HANDLE.fullmatch(tail):
        raise ValueError("share_link must be an artifacts share URL or its handle")
    return tail


def _base(artifact_id: Any, share_link: Optional[str]) -> str:
    if (artifact_id is None) == (share_link is None):
        raise ValueError("pass exactly one of artifact_id or share_link")
    if share_link is not None:
        return f"{_ROOT}shared/{_handle(share_link)}/"
    return f"{_ROOT}{_id(artifact_id)}/"


def _key(idempotency_key: str) -> Dict[str, str]:
    if (
        not isinstance(idempotency_key, str)
        or not 1 <= len(idempotency_key) <= 255
        or any(ord(c) < 0x21 or ord(c) > 0x7E for c in idempotency_key)
    ):
        raise ValueError(
            "idempotency_key must be 1-255 printable ASCII characters without spaces"
        )
    return {"Idempotency-Key": idempotency_key}


def _page(cursor: Optional[str], page_size: Optional[int]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if cursor:
        params["cursor"] = cursor
    if page_size is not None:
        if not 1 <= int(page_size) <= 100:
            raise ValueError("page_size must be 1-100")
        params["page_size"] = int(page_size)
    return params


def _body(**fields: Any) -> Dict[str, Any]:
    return {name: value for name, value in fields.items() if value is not None}


def _write(data: Dict[str, Any]) -> Union[ArtifactReceipt, ArtifactDryRun]:
    if data.get("dry_run"):
        return ArtifactDryRun.model_validate(data)
    return ArtifactReceipt.model_validate(data)


class _Spec:
    """The request of each call, shared by the sync and async resources."""

    @staticmethod
    def list(
        scope: str, cursor: Optional[str], page_size: Optional[int]
    ) -> Dict[str, Any]:
        if scope not in ("mine", "shared"):
            raise ValueError("scope must be mine or shared")
        params = _page(cursor, page_size)
        if scope == "shared":
            params["scope"] = "shared"
        return {"params": params}


class Artifacts:
    """``client.artifacts``: your frontend artifacts and the ones shared with you."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def status(self) -> ArtifactsStatus:
        """Whether artifacts are on for you, and the limits."""
        return ArtifactsStatus.model_validate(
            self._c._request("GET", f"{_ROOT}status/")
        )

    def list(
        self,
        *,
        scope: str = "mine",
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactPage[ArtifactSummary]:
        """Your artifacts, most recently changed first; ``scope="shared"`` lists the
        ones shared with you by name (``role`` editor or member)."""
        data = self._c._request("GET", _ROOT, **_Spec.list(scope, cursor, page_size))
        return ArtifactPage[ArtifactSummary].model_validate(data)

    def get(
        self, artifact_id: Any = None, *, share_link: Optional[str] = None
    ) -> Artifact:
        """One artifact: yours or shared with you by ``artifact_id``, or anyone's by
        ``share_link``."""
        return Artifact.model_validate(
            self._c._request("GET", _base(artifact_id, share_link))
        )

    def create(
        self,
        *,
        name: str,
        files: List[Dict[str, str]],
        idempotency_key: str,
        description: Optional[str] = None,
        entry: Optional[str] = None,
        message: Optional[str] = None,
        visibility: Optional[str] = None,
        with_sources: Optional[bool] = None,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        """Create an artifact from its files; version 1 goes live, private unless
        ``visibility`` says otherwise. Build ``files`` with :func:`artifact_file` or
        :func:`artifact_files_from_directory`."""
        body = _body(
            name=name,
            files=files,
            description=description,
            entry=entry,
            message=message,
            visibility=visibility,
            with_sources=with_sources,
            dry_run=dry_run or None,
        )
        return _write(
            self._c._request("POST", _ROOT, json=body, headers=_key(idempotency_key))
        )

    def publish(
        self,
        artifact_id: Any,
        *,
        base_version: int,
        idempotency_key: str,
        files: Optional[List[Dict[str, str]]] = None,
        delete: Optional[List[str]] = None,
        entry: Optional[str] = None,
        message: Optional[str] = None,
        publish: bool = True,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        """A new version on top of ``base_version`` (the latest): ``files`` add or
        replace paths, ``delete`` removes paths, the rest is kept. Raises
        :class:`ArtifactStaleBaseVersionError` when someone published meanwhile."""
        body = _body(
            base_version=base_version,
            files=files,
            delete=delete,
            entry=entry,
            message=message,
            publish=publish,
            dry_run=dry_run or None,
        )
        data = self._c._request(
            "POST",
            f"{_ROOT}{_id(artifact_id)}/versions/",
            json=body,
            headers=_key(idempotency_key),
        )
        return _write(data)

    def versions(
        self,
        artifact_id: Any,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactPage[ArtifactVersion]:
        data = self._c._request(
            "GET",
            f"{_ROOT}{_id(artifact_id)}/versions/",
            params=_page(cursor, page_size),
        )
        return ArtifactPage[ArtifactVersion].model_validate(data)

    def files(
        self, artifact_id: Any = None, *, version: int, share_link: Optional[str] = None
    ) -> ArtifactFiles:
        """A version's file list (path, bytes, sha256, content_type)."""
        path = f"{_base(artifact_id, share_link)}versions/{int(version)}/files/"
        return ArtifactFiles.model_validate(self._c._request("GET", path))

    def read_file(
        self,
        artifact_id: Any = None,
        *,
        version: int,
        path: str,
        share_link: Optional[str] = None,
        offset: int = 0,
        max_bytes: Optional[int] = None,
    ) -> ArtifactFileText:
        """One text file's content from ``offset``; follow ``next_offset`` while
        ``truncated``."""
        params = _body(path=path, offset=offset, max_bytes=max_bytes)
        url = f"{_base(artifact_id, share_link)}versions/{int(version)}/files/"
        return ArtifactFileText.model_validate(
            self._c._request("GET", url, params=params)
        )

    def view(
        self, artifact_id: Any = None, *, version: int, share_link: Optional[str] = None
    ) -> ArtifactView:
        """A sandboxed frame URL for a version, valid 8 hours. Give people the
        ``owner_url`` or ``share_url`` instead."""
        url = f"{_base(artifact_id, share_link)}versions/{int(version)}/view/"
        return ArtifactView.model_validate(self._c._request("GET", url))

    def update(
        self,
        artifact_id: Any,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        live_version: Optional[int] = None,
    ) -> Artifact:
        """Rename, describe or choose the live version (owner or editor)."""
        body = _body(name=name, description=description, live_version=live_version)
        return Artifact.model_validate(
            self._c._request("PATCH", f"{_ROOT}{_id(artifact_id)}/", json=body)
        )

    def share(
        self,
        artifact_id: Any,
        *,
        idempotency_key: str,
        visibility: Optional[str] = None,
        with_sources: Optional[bool] = None,
        reset_link: Optional[bool] = None,
    ) -> Artifact:
        """Who sees it: ``private``, ``link`` or ``public``, with or without sources
        (owner)."""
        body = _body(
            visibility=visibility, with_sources=with_sources, reset_link=reset_link
        )
        data = self._c._request(
            "PATCH",
            f"{_ROOT}{_id(artifact_id)}/share/",
            json=body,
            headers=_key(idempotency_key),
        )
        return Artifact.model_validate(data)

    def clone(
        self,
        artifact_id: Any = None,
        *,
        idempotency_key: str,
        share_link: Optional[str] = None,
        version: Optional[int] = None,
        name: Optional[str] = None,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        """Your own separate copy of an artifact (yours, shared with you, or by a
        link with sources); ``forked_from`` records where it came from."""
        body = _body(version=version, name=name, dry_run=dry_run or None)
        data = self._c._request(
            "POST",
            f"{_base(artifact_id, share_link)}clone/",
            json=body,
            headers=_key(idempotency_key),
        )
        return _write(data)

    def lineage(
        self,
        artifact_id: Any,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactLineage:
        data = self._c._request(
            "GET",
            f"{_ROOT}{_id(artifact_id)}/lineage/",
            params=_page(cursor, page_size),
        )
        return ArtifactLineage.model_validate(data)

    def history(
        self,
        artifact_id: Any,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactPage[ArtifactEvent]:
        data = self._c._request(
            "GET",
            f"{_ROOT}{_id(artifact_id)}/history/",
            params=_page(cursor, page_size),
        )
        return ArtifactPage[ArtifactEvent].model_validate(data)

    def members(self, artifact_id: Any) -> ArtifactMembers:
        """Who it is shared with by name (owner and members may read)."""
        return ArtifactMembers.model_validate(
            self._c._request("GET", f"{_ROOT}{_id(artifact_id)}/members/")
        )

    def add_member(
        self, artifact_id: Any, user: str, *, role: str = "view"
    ) -> ArtifactMemberReceipt:
        """Share with a FlyMyAI user (email or username) to ``view`` or ``edit``, or
        change their role (owner). Calling it again with the same role changes nothing.
        """
        if role not in ("view", "edit"):
            raise ValueError("role must be view or edit")
        data = self._c._request(
            "POST",
            f"{_ROOT}{_id(artifact_id)}/members/",
            json={"user": user, "role": role},
        )
        return ArtifactMemberReceipt.model_validate(data)

    def remove_member(self, artifact_id: Any, username: str) -> Dict[str, Any]:
        """Stop sharing with someone (owner); a member may remove themself to leave."""
        path = f"{_ROOT}{_id(artifact_id)}/members/{quote(username, safe='')}/"
        return self._c._request("DELETE", path)

    def delete(self, artifact_id: Any) -> Dict[str, Any]:
        """Delete the artifact (owner): its link and open frames stop; clones stay."""
        return self._c._request("DELETE", f"{_ROOT}{_id(artifact_id)}/")


class AsyncArtifacts:
    """``client.artifacts`` of :class:`AsyncAgentClient`."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def status(self) -> ArtifactsStatus:
        return ArtifactsStatus.model_validate(
            await self._c._request("GET", f"{_ROOT}status/")
        )

    async def list(
        self,
        *,
        scope: str = "mine",
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactPage[ArtifactSummary]:
        data = await self._c._request(
            "GET", _ROOT, **_Spec.list(scope, cursor, page_size)
        )
        return ArtifactPage[ArtifactSummary].model_validate(data)

    async def get(
        self, artifact_id: Any = None, *, share_link: Optional[str] = None
    ) -> Artifact:
        return Artifact.model_validate(
            await self._c._request("GET", _base(artifact_id, share_link))
        )

    async def create(
        self,
        *,
        name: str,
        files: List[Dict[str, str]],
        idempotency_key: str,
        description: Optional[str] = None,
        entry: Optional[str] = None,
        message: Optional[str] = None,
        visibility: Optional[str] = None,
        with_sources: Optional[bool] = None,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        body = _body(
            name=name,
            files=files,
            description=description,
            entry=entry,
            message=message,
            visibility=visibility,
            with_sources=with_sources,
            dry_run=dry_run or None,
        )
        return _write(
            await self._c._request(
                "POST", _ROOT, json=body, headers=_key(idempotency_key)
            )
        )

    async def publish(
        self,
        artifact_id: Any,
        *,
        base_version: int,
        idempotency_key: str,
        files: Optional[List[Dict[str, str]]] = None,
        delete: Optional[List[str]] = None,
        entry: Optional[str] = None,
        message: Optional[str] = None,
        publish: bool = True,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        body = _body(
            base_version=base_version,
            files=files,
            delete=delete,
            entry=entry,
            message=message,
            publish=publish,
            dry_run=dry_run or None,
        )
        data = await self._c._request(
            "POST",
            f"{_ROOT}{_id(artifact_id)}/versions/",
            json=body,
            headers=_key(idempotency_key),
        )
        return _write(data)

    async def versions(
        self,
        artifact_id: Any,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactPage[ArtifactVersion]:
        data = await self._c._request(
            "GET",
            f"{_ROOT}{_id(artifact_id)}/versions/",
            params=_page(cursor, page_size),
        )
        return ArtifactPage[ArtifactVersion].model_validate(data)

    async def files(
        self, artifact_id: Any = None, *, version: int, share_link: Optional[str] = None
    ) -> ArtifactFiles:
        path = f"{_base(artifact_id, share_link)}versions/{int(version)}/files/"
        return ArtifactFiles.model_validate(await self._c._request("GET", path))

    async def read_file(
        self,
        artifact_id: Any = None,
        *,
        version: int,
        path: str,
        share_link: Optional[str] = None,
        offset: int = 0,
        max_bytes: Optional[int] = None,
    ) -> ArtifactFileText:
        params = _body(path=path, offset=offset, max_bytes=max_bytes)
        url = f"{_base(artifact_id, share_link)}versions/{int(version)}/files/"
        return ArtifactFileText.model_validate(
            await self._c._request("GET", url, params=params)
        )

    async def view(
        self, artifact_id: Any = None, *, version: int, share_link: Optional[str] = None
    ) -> ArtifactView:
        url = f"{_base(artifact_id, share_link)}versions/{int(version)}/view/"
        return ArtifactView.model_validate(await self._c._request("GET", url))

    async def update(
        self,
        artifact_id: Any,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        live_version: Optional[int] = None,
    ) -> Artifact:
        body = _body(name=name, description=description, live_version=live_version)
        return Artifact.model_validate(
            await self._c._request("PATCH", f"{_ROOT}{_id(artifact_id)}/", json=body)
        )

    async def share(
        self,
        artifact_id: Any,
        *,
        idempotency_key: str,
        visibility: Optional[str] = None,
        with_sources: Optional[bool] = None,
        reset_link: Optional[bool] = None,
    ) -> Artifact:
        body = _body(
            visibility=visibility, with_sources=with_sources, reset_link=reset_link
        )
        data = await self._c._request(
            "PATCH",
            f"{_ROOT}{_id(artifact_id)}/share/",
            json=body,
            headers=_key(idempotency_key),
        )
        return Artifact.model_validate(data)

    async def clone(
        self,
        artifact_id: Any = None,
        *,
        idempotency_key: str,
        share_link: Optional[str] = None,
        version: Optional[int] = None,
        name: Optional[str] = None,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        body = _body(version=version, name=name, dry_run=dry_run or None)
        data = await self._c._request(
            "POST",
            f"{_base(artifact_id, share_link)}clone/",
            json=body,
            headers=_key(idempotency_key),
        )
        return _write(data)

    async def lineage(
        self,
        artifact_id: Any,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactLineage:
        data = await self._c._request(
            "GET",
            f"{_ROOT}{_id(artifact_id)}/lineage/",
            params=_page(cursor, page_size),
        )
        return ArtifactLineage.model_validate(data)

    async def history(
        self,
        artifact_id: Any,
        *,
        cursor: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> ArtifactPage[ArtifactEvent]:
        data = await self._c._request(
            "GET",
            f"{_ROOT}{_id(artifact_id)}/history/",
            params=_page(cursor, page_size),
        )
        return ArtifactPage[ArtifactEvent].model_validate(data)

    async def members(self, artifact_id: Any) -> ArtifactMembers:
        return ArtifactMembers.model_validate(
            await self._c._request("GET", f"{_ROOT}{_id(artifact_id)}/members/")
        )

    async def add_member(
        self, artifact_id: Any, user: str, *, role: str = "view"
    ) -> ArtifactMemberReceipt:
        if role not in ("view", "edit"):
            raise ValueError("role must be view or edit")
        data = await self._c._request(
            "POST",
            f"{_ROOT}{_id(artifact_id)}/members/",
            json={"user": user, "role": role},
        )
        return ArtifactMemberReceipt.model_validate(data)

    async def remove_member(self, artifact_id: Any, username: str) -> Dict[str, Any]:
        path = f"{_ROOT}{_id(artifact_id)}/members/{quote(username, safe='')}/"
        return await self._c._request("DELETE", path)

    async def delete(self, artifact_id: Any) -> Dict[str, Any]:
        return await self._c._request("DELETE", f"{_ROOT}{_id(artifact_id)}/")
