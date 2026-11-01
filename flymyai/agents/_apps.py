"""Apps: a project's config applied as a release. An app is one ``flymy.yaml``
(``flymy.ai/v1alpha1``, ``kind: App``) and the files it names - its pages, agents,
storages and budgets.

``plan`` shows what applying the files would create, change, keep or stop and what it
costs, and returns a ``plan_token``; nothing is spent. ``apply`` applies exactly that
plan as a release. ``files`` reads an applied app's template back to change it, and
``status`` follows a release. ``secrets`` sets the values of the app's ``secrets:``,
which only its rented servers get, and never reads one back. The routes live on the
agents host under ``/api/v1/agents/apps/``. Unknown answer fields are kept on the models
(``extra="allow"``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from flymyai.agents._artifacts import _body, _key

_ROOT = "/api/v1/agents/apps/"
_APP = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_FILE_PAGE_MAX_BYTES = 200_000
_SECRETS = f"{_ROOT}secrets/"
_SECRET_APP = re.compile(r"[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9._-]*")
_SECRET_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_SECRET_MAX_BYTES = 8 * 1024


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)


class AppPlan(_Model):
    """What applying would do (``changes``) and cost (``usd_per_hour``, the budgets),
    with the ``plan_token`` that applies exactly this plan. Nothing was spent."""

    plan_token: str
    app: Optional[str] = None
    changes: List[Dict[str, Any]] = []
    usd_per_hour: Optional[str] = None


class AppRelease(_Model):
    """One release of an app: ``status`` applying, succeeded or failed, its agents,
    pages and the error of a failed one."""

    release: str
    status: str
    app: Optional[str] = None
    number: Optional[int] = None


class AppFiles(_Model):
    """An applied app's template (``flymy.app-files.v1``): every file with its bytes,
    sha256 and whether it is text, the release and the template's digest."""

    files: List[Dict[str, Any]] = []


class AppFile(_Model):
    """One page of one file of the template (``flymy.app-file.v1``): text as
    ``content``, anything else as ``content_base64``; read on from ``next_offset``."""

    path: str
    content: Optional[str] = None
    content_base64: Optional[str] = None
    next_offset: Optional[int] = None


class AppSecret(_Model):
    """One secret of an app (``flymy.app-secret.v1``): whether it has a value, its
    ``version`` (every new value is one) and whether the platform ``generated`` it; in a
    listing also whether the applied release ``declared`` it and the server env entries
    that use it (``used_by``). Never its value."""

    # Only the declared metadata contract may leave this resource, even if an
    # upstream response accidentally adds a value-bearing field.
    model_config = ConfigDict(extra="ignore")

    name: str
    input_url: Optional[str] = None
    set: bool = True
    version: Optional[int] = None
    generated: Optional[bool] = None
    updated_at: Optional[str] = None
    declared: Optional[bool] = None
    used_by: List[str] = Field(default_factory=list)
    app: Optional[str] = None
    schema_: Optional[str] = Field(default=None, alias="schema")


class AppSecretList(_Model):
    """The app's secrets (``flymy.app-secrets.v1``): the ones with a value and the ones
    its applied release declares, by name."""

    model_config = ConfigDict(extra="ignore")

    app: str
    secrets: List[AppSecret] = Field(default_factory=list)
    input_url: Optional[str] = None
    schema_: Optional[str] = Field(default=None, alias="schema")


def _secret_app(app: str) -> str:
    if not isinstance(app, str) or len(app) > 160 or not _SECRET_APP.fullmatch(app):
        raise ValueError("app must be a lowercase owner/name of at most 160 characters")
    return app


def _secret_name(name: str) -> str:
    if not isinstance(name, str) or not _SECRET_NAME.fullmatch(name):
        raise ValueError(
            "name must be an env variable's name: letters, digits and _, at most 64,"
            " not starting with a digit"
        )
    return name


def _secret_value(value: str) -> str:
    # The message never repeats the value, including malformed Unicode input.
    try:
        size = len(value.encode("utf-8")) if isinstance(value, str) else 0
    except UnicodeEncodeError:
        raise ValueError("value must be valid UTF-8 text") from None
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or size > _SECRET_MAX_BYTES
    ):
        raise ValueError(
            f"value must be text of 1 to {_SECRET_MAX_BYTES} bytes, without NUL"
        )
    return value


def _generate(length: Optional[int]) -> Any:
    if length is None:
        return True
    if (
        isinstance(length, bool)
        or not isinstance(length, int)
        or not 16 <= length <= 128
    ):
        raise ValueError("length must be 16-128")
    return {"length": length}


def _app(app: str) -> str:
    if not isinstance(app, str) or not 3 <= len(app) <= 200 or not _APP.fullmatch(app):
        raise ValueError("app must be an applied app as owner/name")
    return app


def _token(plan_token: str) -> str:
    if not isinstance(plan_token, str) or not plan_token:
        raise ValueError("plan_token must be the plan_token plan() returned")
    return plan_token


def _source(
    files: Optional[List[Dict[str, str]]],
    app: Optional[str],
    overrides: Optional[List[Dict[str, str]]],
    instances: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if (files is None) == (app is None):
        raise ValueError(
            "pass exactly one of files (flymy.yaml at the root and the files it names)"
            " or app (owner/name of an app you applied)"
        )
    if overrides is not None and app is None:
        raise ValueError(
            "overrides change an applied app's template: pass them with app"
        )
    return _body(
        files=files,
        app=None if app is None else _app(app),
        overrides=overrides,
        instances=instances,
    )


def _files_query(
    app: str, path: Optional[str], offset: Optional[int], limit: Optional[int]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"app": _app(app)}
    if path is None:
        if offset is not None or limit is not None:
            raise ValueError("offset and limit read a page of one file: pass path too")
        return params
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(ord(c) < 0x20 for c in path)
        or ".." in path.split("/")
    ):
        raise ValueError("path must be a file of the template, for example site/app.js")
    params["path"] = path
    if offset is not None:
        if int(offset) < 0:
            raise ValueError("offset must be 0 or more")
        params["offset"] = int(offset)
    if limit is not None:
        if not 1 <= int(limit) <= _FILE_PAGE_MAX_BYTES:
            raise ValueError("limit must be 1-200000")
        params["limit"] = int(limit)
    return params


class AppSecrets:
    """``client.apps.secrets``: the values of an app's ``secrets:``. A value goes in and
    never comes out; only the app's rented servers get it, in their env at start
    (``env: {KEY: ${secrets.NAME}}``). Values may be set before the app's first apply:
    its plan asks for the ones it uses (``needs_secret``). ``set`` is for trusted owner
    code only: read from your own environment or secret store, never from an agent
    prompt or chat. Assistants use ``list().input_url`` to direct the owner to the
    authenticated write-only form, or ``generate`` to make a value server-side."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def list(self, app: str) -> AppSecretList:
        """The app's secrets, names and state only."""
        data = self._c._request("GET", _SECRETS, params={"app": _secret_app(app)})
        return AppSecretList.model_validate(data)

    def set(self, app: str, name: str, value: str) -> AppSecret:
        """Set a secret's value. A new value is a new version, the same value changes
        nothing; a running server gets it at its next start. Trusted owner code only;
        never pass a credential through an assistant's arguments or transcript."""
        body = {
            "app": _secret_app(app),
            "name": _secret_name(name),
            "value": _secret_value(value),
        }
        return AppSecret.model_validate(self._c._request("PUT", _SECRETS, json=body))

    def generate(self, app: str, name: str, length: Optional[int] = None) -> AppSecret:
        """Have the platform make a value (letters and digits, 32 or ``length`` 16-128)
        when the secret has none; a repeated call answers the same secret. To make a new
        value, :meth:`delete` it first."""
        body = {
            "app": _secret_app(app),
            "name": _secret_name(name),
            "generate": _generate(length),
        }
        return AppSecret.model_validate(self._c._request("PUT", _SECRETS, json=body))

    def delete(self, app: str, name: str) -> bool:
        """Delete a secret's value; True when there was one. A server that uses it fails
        its next start until a new value is set."""
        data = self._c._request(
            "DELETE",
            _SECRETS,
            params={"app": _secret_app(app), "name": _secret_name(name)},
        )
        return bool((data or {}).get("deleted"))


class AsyncAppSecrets:
    """``client.apps.secrets`` on the async client."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def list(self, app: str) -> AppSecretList:
        data = await self._c._request("GET", _SECRETS, params={"app": _secret_app(app)})
        return AppSecretList.model_validate(data)

    async def set(self, app: str, name: str, value: str) -> AppSecret:
        body = {
            "app": _secret_app(app),
            "name": _secret_name(name),
            "value": _secret_value(value),
        }
        data = await self._c._request("PUT", _SECRETS, json=body)
        return AppSecret.model_validate(data)

    async def generate(
        self, app: str, name: str, length: Optional[int] = None
    ) -> AppSecret:
        body = {
            "app": _secret_app(app),
            "name": _secret_name(name),
            "generate": _generate(length),
        }
        data = await self._c._request("PUT", _SECRETS, json=body)
        return AppSecret.model_validate(data)

    async def delete(self, app: str, name: str) -> bool:
        data = await self._c._request(
            "DELETE",
            _SECRETS,
            params={"app": _secret_app(app), "name": _secret_name(name)},
        )
        return bool((data or {}).get("deleted"))


class Apps:
    """``client.apps``: plan and apply a ``flymy.yaml``, read it back, follow a
    release; ``client.apps.secrets`` sets the values of its secrets."""

    def __init__(self, client: Any) -> None:
        self._c = client
        self.secrets = AppSecrets(client)

    def plan(
        self,
        *,
        files: Optional[List[Dict[str, str]]] = None,
        app: Optional[str] = None,
        overrides: Optional[List[Dict[str, str]]] = None,
        instances: Optional[Dict[str, Any]] = None,
    ) -> AppPlan:
        """What applying ``files`` (or the stored template of ``app``, with
        ``overrides``) would do and cost. Build ``files`` with
        :func:`artifact_file` or :func:`artifact_files_from_directory`."""
        body = _source(files, app, overrides, instances)
        return AppPlan.model_validate(
            self._c._request("POST", f"{_ROOT}plan/", json=body)
        )

    def apply(
        self,
        *,
        plan_token: str,
        idempotency_key: str,
        files: Optional[List[Dict[str, str]]] = None,
        app: Optional[str] = None,
        overrides: Optional[List[Dict[str, str]]] = None,
        instances: Optional[Dict[str, Any]] = None,
    ) -> AppRelease:
        """Apply exactly the plan the user confirmed: the same source as :meth:`plan`
        plus its ``plan_token``. The release is ``applying`` while agents freeze:
        follow it with :meth:`status`."""
        body = {
            **_source(files, app, overrides, instances),
            "plan_token": _token(plan_token),
        }
        data = self._c._request(
            "POST", f"{_ROOT}apply/", json=body, headers=_key(idempotency_key)
        )
        return AppRelease.model_validate(data)

    def files(
        self,
        app: str,
        *,
        path: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Any:
        """The applied app's template (:class:`AppFiles`), or with ``path`` one page of
        that file (:class:`AppFile`)."""
        data = self._c._request(
            "GET", f"{_ROOT}files/", params=_files_query(app, path, offset, limit)
        )
        if path is None:
            return AppFiles.model_validate(data)
        return AppFile.model_validate(data)

    def status(self, release_id: Any) -> AppRelease:
        """One release, by the ``release`` id :meth:`apply` returned."""
        data = self._c._request("GET", f"{_ROOT}releases/{UUID(str(release_id))}/")
        return AppRelease.model_validate(data)


class AsyncApps:
    """``client.apps`` on the async client."""

    def __init__(self, client: Any) -> None:
        self._c = client
        self.secrets = AsyncAppSecrets(client)

    async def plan(
        self,
        *,
        files: Optional[List[Dict[str, str]]] = None,
        app: Optional[str] = None,
        overrides: Optional[List[Dict[str, str]]] = None,
        instances: Optional[Dict[str, Any]] = None,
    ) -> AppPlan:
        body = _source(files, app, overrides, instances)
        data = await self._c._request("POST", f"{_ROOT}plan/", json=body)
        return AppPlan.model_validate(data)

    async def apply(
        self,
        *,
        plan_token: str,
        idempotency_key: str,
        files: Optional[List[Dict[str, str]]] = None,
        app: Optional[str] = None,
        overrides: Optional[List[Dict[str, str]]] = None,
        instances: Optional[Dict[str, Any]] = None,
    ) -> AppRelease:
        body = {
            **_source(files, app, overrides, instances),
            "plan_token": _token(plan_token),
        }
        data = await self._c._request(
            "POST", f"{_ROOT}apply/", json=body, headers=_key(idempotency_key)
        )
        return AppRelease.model_validate(data)

    async def files(
        self,
        app: str,
        *,
        path: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Any:
        data = await self._c._request(
            "GET", f"{_ROOT}files/", params=_files_query(app, path, offset, limit)
        )
        if path is None:
            return AppFiles.model_validate(data)
        return AppFile.model_validate(data)

    async def status(self, release_id: Any) -> AppRelease:
        data = await self._c._request(
            "GET", f"{_ROOT}releases/{UUID(str(release_id))}/"
        )
        return AppRelease.model_validate(data)
