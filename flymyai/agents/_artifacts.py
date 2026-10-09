"""Frontend artifacts (``flymy.artifact.v1``): small web pages, presentations and mini
games with immutable versions, shared by link or with people by name, cloned into
your own copy.

A new artifact may start from a standard type - a presentation, a game, a landing
page, a report, a dashboard or a gallery (:meth:`Artifacts.types`).

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
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from flymyai.agents._resources import _idempotency_headers

_ROOT = "/api/v1/artifacts/"
_HANDLE = re.compile(r"[A-Za-z0-9_-]{8,64}")
# A share link is only https://<app>/artifacts/s/<handle>; a slash, a query string or
# a fragment after it is fine.
_SHARE_PATH = re.compile(r"/artifacts/s/([A-Za-z0-9_-]{8,64})/?\Z")
# The artifact's own page in the app, https://<app>/artifacts/<id> (owner_url, app_url,
# a version's page_url with ?v=<n>): an artifact id, never a share handle.
_PAGE_PATH = re.compile(
    r"/artifacts/([0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}"
    r"-[0-9A-Fa-f]{12})/?\Z"
)
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
    ".yaml",
    ".yml",
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
    # The standard type it started from (presentation, game, ...), None for an
    # artifact made from its own files.
    type: Optional[str] = None
    role: str  # owner, editor, member or viewer
    live_version: Optional[int] = None
    latest_version: int
    visibility: str  # private, link or public
    with_sources: bool
    clone_count: int = 0
    forked_from: Optional[ArtifactForkedFrom] = None
    owner_url: Optional[str] = None
    # The app page, for the owner and the people it is shared with by name.
    app_url: Optional[str] = None
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
    # The newest 20 versions the caller may see; None for a link viewer when the
    # author shared only the page (no sources).
    versions: Optional[List[ArtifactVersion]] = None
    share: Optional[ArtifactShare] = None
    # One sentence for an agent: what the caller does when asked to change it - a new
    # version next to the live one (owner, editor) or their own clone (anyone else).
    next: Optional[str] = None


class ArtifactPlan(_Model):
    # The version made; with ``no_change`` the version that already holds these
    # files (the base), as nothing new was made.
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
    # A published version's app page (``?v=<n>``), live or not: show it to people.
    page_url: Optional[str] = None


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
    """Someone the artifact is shared with by name. Nobody learns who an email or a
    username belongs to: the owner sees ``invited_as`` (exactly what they typed), a
    member sees only which row is theirs (``you``)."""

    id: Optional[str] = None  # remove_member takes it
    role: str  # view or edit
    status: str = "accepted"  # pending until the person accepts
    added_at: datetime
    accepted_at: Optional[datetime] = None
    you: bool = False
    invited_as: Optional[str] = None  # the owner sees it


class ArtifactMembers(_Model):
    role: str
    owner: ArtifactAuthor
    members: List[ArtifactMember]
    max_members: int


class ArtifactMemberReceipt(_Model):
    member: ArtifactMember
    changed: bool


class ArtifactInvitedTo(_Model):
    id: UUID
    name: str


class ArtifactInvitation(_Model):
    """An artifact someone shared with you by name, waiting for your answer: nothing of
    it is shared with you until you accept."""

    id: str
    artifact: ArtifactInvitedTo
    owner: ArtifactAuthor
    role: str  # view or edit
    invited_as: Optional[str] = None  # your email or username as the owner typed it
    status: str = "pending"
    invited_at: datetime


class ArtifactInvitationAccepted(_Model):
    invitation: ArtifactInvitation
    artifact: ArtifactSummary


class ArtifactInvitationDeclined(_Model):
    id: str
    declined: bool


class ArtifactsStatus(_Model):
    enabled: bool
    limits: Dict[str, int] = Field(default_factory=dict)


class ArtifactType(_Model):
    """A standard type a new artifact may start from (:meth:`Artifacts.types`): a
    template whose files become version 1. ``edit`` is the one file most changes
    touch (slides.md, site.json, game.config.js, ...); the template's README.md
    says how to change it."""

    key: str  # what create(type=...) takes: presentation, game, landing, ...
    name: str
    summary: str = ""
    edit: str = ""
    default_name: str = ""  # the name a new artifact gets when none is given
    example_change: str = ""  # one change people may ask for
    entry: str = "index.html"
    files_count: int = 0
    total_bytes: int = 0


class _ArtifactTypes(_Model):
    types: List[ArtifactType] = Field(default_factory=list)


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


def _link(link: str) -> Tuple[str, str]:
    """What a link someone gave you addresses: ``("handle", <handle>)`` for a share
    link ``https://<app>/artifacts/s/<handle>`` or a bare handle, ``("id", <uuid>)``
    for the artifact's own page ``https://<app>/artifacts/<id>``. A query string or a
    fragment (``?v=3``) is ignored; any other URL is refused, never read as a handle."""

    text = str(link).strip()
    if "/" not in text:
        if _HANDLE.fullmatch(text):
            return "handle", text
    else:
        path = urlsplit(text).path
        share = _SHARE_PATH.search(path)
        if share is not None:
            return "handle", share.group(1)
        page = _PAGE_PATH.search(path)
        if page is not None:
            return "id", _id(page.group(1))
    raise ValueError(
        "share_link must be a share link (https://app.flymy.ai/artifacts/s/<handle>),"
        " its handle, or the artifact's page (https://app.flymy.ai/artifacts/<id>)"
    )


def _artifact_id(artifact_id: Any) -> str:
    """An artifact's id: the UUID, or its page in the app (``owner_url``,
    ``app_url``, a version's ``page_url``) - never a share link."""

    if isinstance(artifact_id, str) and "/" in artifact_id:
        page = _PAGE_PATH.search(urlsplit(artifact_id.strip()).path)
        if page is None:
            raise ValueError(
                "artifact_id must be an artifact id or its page"
                " (https://app.flymy.ai/artifacts/<id>); a share link goes in"
                " share_link"
            )
        return _id(page.group(1))
    return _id(artifact_id)


def _base(artifact_id: Any, share_link: Optional[str]) -> str:
    if (artifact_id is None) == (share_link is None):
        raise ValueError("pass exactly one of artifact_id or share_link")
    if share_link is not None:
        kind, value = _link(share_link)
        if kind == "id":
            # the artifact's own page opens by its id (for its owner and its people)
            return f"{_ROOT}{value}/"
        return f"{_ROOT}shared/{value}/"
    return f"{_ROOT}{_artifact_id(artifact_id)}/"


def _key(idempotency_key: str) -> Dict[str, str]:
    # One rule for every caller-owned key, the backend's and the Agents MCP's: 1-255
    # printable ASCII characters, spaces inside allowed, none at either end.
    return _idempotency_headers(idempotency_key)


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

    @staticmethod
    def create(
        name: Optional[str],
        files: Optional[List[Dict[str, str]]],
        kind: Optional[str],
        **fields: Any,
    ) -> Dict[str, Any]:
        # a standard type gives version 1's files and a default name
        if kind is None and (name is None or files is None):
            raise ValueError("pass name and files, or a type (see types())")
        return _body(name=name, type=kind, files=files, **fields)

    @staticmethod
    def types(data: Any) -> List[ArtifactType]:
        return _ArtifactTypes.model_validate(data).types


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

    def types(self) -> List[ArtifactType]:
        """The standard types a new artifact may start from (presentation, game,
        landing page, report, dashboard, gallery): pass a ``key`` to :meth:`create`.
        ``edit`` names the file most changes touch."""
        return _Spec.types(self._c._request("GET", f"{_ROOT}types/"))

    def get(
        self, artifact_id: Any = None, *, share_link: Optional[str] = None
    ) -> Artifact:
        """One artifact: yours or shared with you by ``artifact_id`` (its UUID or
        its app page ``https://app.flymy.ai/artifacts/<id>``), or anyone's by
        ``share_link`` (``https://app.flymy.ai/artifacts/s/<handle>`` or the
        handle). ``next`` says what to do when the person asks for a change."""
        return Artifact.model_validate(
            self._c._request("GET", _base(artifact_id, share_link))
        )

    def create(
        self,
        *,
        name: Optional[str] = None,
        files: Optional[List[Dict[str, str]]] = None,
        idempotency_key: str,
        type: Optional[str] = None,
        description: Optional[str] = None,
        entry: Optional[str] = None,
        message: Optional[str] = None,
        visibility: Optional[str] = None,
        with_sources: Optional[bool] = None,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        """Create an artifact from its files; version 1 goes live, private unless
        ``visibility`` says otherwise. Build ``files`` with :func:`artifact_file` or
        :func:`artifact_files_from_directory`. With ``type`` (a ``key`` from
        :meth:`types`) version 1 starts from that template with ``files`` laid over
        it, so ``files`` and ``name`` (default: the type's ``default_name``) are
        optional."""
        body = _Spec.create(
            name,
            files,
            type,
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
        """A new version on top of ``base_version``: the latest version, or the live
        one (a saved draft above it then stays a draft). ``files`` add or replace
        paths, ``delete`` removes paths, the rest is kept. ``publish=False`` keeps
        the live version and saves this one next to it; ``version.page_url`` shows
        it, and :meth:`update` with ``live_version`` makes it live later. Raises
        :class:`ArtifactStaleBaseVersionError` when ``base_version`` is neither the
        latest nor the live version."""
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
            f"{_ROOT}{_artifact_id(artifact_id)}/versions/",
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
            f"{_ROOT}{_artifact_id(artifact_id)}/versions/",
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
        """Rename, describe or choose the live version (owner or editor):
        ``live_version`` makes a saved version the one people see, or rolls back."""
        body = _body(name=name, description=description, live_version=live_version)
        return Artifact.model_validate(
            self._c._request("PATCH", f"{_ROOT}{_artifact_id(artifact_id)}/", json=body)
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
            f"{_ROOT}{_artifact_id(artifact_id)}/share/",
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
            f"{_ROOT}{_artifact_id(artifact_id)}/lineage/",
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
            f"{_ROOT}{_artifact_id(artifact_id)}/history/",
            params=_page(cursor, page_size),
        )
        return ArtifactPage[ArtifactEvent].model_validate(data)

    def members(self, artifact_id: Any) -> ArtifactMembers:
        """Who it is shared with by name and who has not accepted yet (the owner and
        the members who accepted may read)."""
        return ArtifactMembers.model_validate(
            self._c._request("GET", f"{_ROOT}{_artifact_id(artifact_id)}/members/")
        )

    def add_member(
        self, artifact_id: Any, user: str, *, role: str = "view"
    ) -> ArtifactMemberReceipt:
        """Invite a FlyMyAI user (email or username) to ``view`` or ``edit``, or change
        their role (owner). The person accepts first; until then nothing is shared. An
        email answers the same whether an account has it. Calling it again with the same
        role changes nothing."""
        if role not in ("view", "edit"):
            raise ValueError("role must be view or edit")
        data = self._c._request(
            "POST",
            f"{_ROOT}{_artifact_id(artifact_id)}/members/",
            json={"user": user, "role": role},
        )
        return ArtifactMemberReceipt.model_validate(data)

    def remove_member(self, artifact_id: Any, member_id: Any) -> Dict[str, Any]:
        """Stop sharing with one person, invited or accepted, by the member's ``id``
        (owner); a member may remove their own row to leave."""
        path = f"{_ROOT}{_artifact_id(artifact_id)}/members/{_id(member_id)}/"
        return self._c._request("DELETE", path)

    def invitations(
        self, *, cursor: Optional[str] = None, page_size: Optional[int] = None
    ) -> ArtifactPage[ArtifactInvitation]:
        """Invitations waiting for you, newest first."""
        data = self._c._request(
            "GET", f"{_ROOT}invitations/", params=_page(cursor, page_size)
        )
        return ArtifactPage[ArtifactInvitation].model_validate(data)

    def accept_invitation(self, invitation_id: Any) -> ArtifactInvitationAccepted:
        """Accept: the artifact is shared with you from now on."""
        data = self._c._request(
            "POST", f"{_ROOT}invitations/{_id(invitation_id)}/accept/"
        )
        return ArtifactInvitationAccepted.model_validate(data)

    def decline_invitation(self, invitation_id: Any) -> ArtifactInvitationDeclined:
        """Decline: it goes and nothing is shared; the owner may invite again."""
        data = self._c._request(
            "POST", f"{_ROOT}invitations/{_id(invitation_id)}/decline/"
        )
        return ArtifactInvitationDeclined.model_validate(data)

    def delete(self, artifact_id: Any) -> Dict[str, Any]:
        """Delete the artifact (owner): its link and open frames stop; clones stay."""
        return self._c._request("DELETE", f"{_ROOT}{_artifact_id(artifact_id)}/")


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

    async def types(self) -> List[ArtifactType]:
        return _Spec.types(await self._c._request("GET", f"{_ROOT}types/"))

    async def get(
        self, artifact_id: Any = None, *, share_link: Optional[str] = None
    ) -> Artifact:
        return Artifact.model_validate(
            await self._c._request("GET", _base(artifact_id, share_link))
        )

    async def create(
        self,
        *,
        name: Optional[str] = None,
        files: Optional[List[Dict[str, str]]] = None,
        idempotency_key: str,
        type: Optional[str] = None,
        description: Optional[str] = None,
        entry: Optional[str] = None,
        message: Optional[str] = None,
        visibility: Optional[str] = None,
        with_sources: Optional[bool] = None,
        dry_run: bool = False,
    ) -> Union[ArtifactReceipt, ArtifactDryRun]:
        body = _Spec.create(
            name,
            files,
            type,
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
            f"{_ROOT}{_artifact_id(artifact_id)}/versions/",
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
            f"{_ROOT}{_artifact_id(artifact_id)}/versions/",
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
            await self._c._request(
                "PATCH", f"{_ROOT}{_artifact_id(artifact_id)}/", json=body
            )
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
            f"{_ROOT}{_artifact_id(artifact_id)}/share/",
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
            f"{_ROOT}{_artifact_id(artifact_id)}/lineage/",
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
            f"{_ROOT}{_artifact_id(artifact_id)}/history/",
            params=_page(cursor, page_size),
        )
        return ArtifactPage[ArtifactEvent].model_validate(data)

    async def members(self, artifact_id: Any) -> ArtifactMembers:
        return ArtifactMembers.model_validate(
            await self._c._request(
                "GET", f"{_ROOT}{_artifact_id(artifact_id)}/members/"
            )
        )

    async def add_member(
        self, artifact_id: Any, user: str, *, role: str = "view"
    ) -> ArtifactMemberReceipt:
        if role not in ("view", "edit"):
            raise ValueError("role must be view or edit")
        data = await self._c._request(
            "POST",
            f"{_ROOT}{_artifact_id(artifact_id)}/members/",
            json={"user": user, "role": role},
        )
        return ArtifactMemberReceipt.model_validate(data)

    async def remove_member(self, artifact_id: Any, member_id: Any) -> Dict[str, Any]:
        path = f"{_ROOT}{_artifact_id(artifact_id)}/members/{_id(member_id)}/"
        return await self._c._request("DELETE", path)

    async def invitations(
        self, *, cursor: Optional[str] = None, page_size: Optional[int] = None
    ) -> ArtifactPage[ArtifactInvitation]:
        data = await self._c._request(
            "GET", f"{_ROOT}invitations/", params=_page(cursor, page_size)
        )
        return ArtifactPage[ArtifactInvitation].model_validate(data)

    async def accept_invitation(self, invitation_id: Any) -> ArtifactInvitationAccepted:
        data = await self._c._request(
            "POST", f"{_ROOT}invitations/{_id(invitation_id)}/accept/"
        )
        return ArtifactInvitationAccepted.model_validate(data)

    async def decline_invitation(
        self, invitation_id: Any
    ) -> ArtifactInvitationDeclined:
        data = await self._c._request(
            "POST", f"{_ROOT}invitations/{_id(invitation_id)}/decline/"
        )
        return ArtifactInvitationDeclined.model_validate(data)

    async def delete(self, artifact_id: Any) -> Dict[str, Any]:
        return await self._c._request("DELETE", f"{_ROOT}{_artifact_id(artifact_id)}/")
