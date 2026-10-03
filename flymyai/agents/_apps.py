"""Apps: a project's config applied as a release. An app is one ``flymy.yaml``
(``flymy.ai/v1alpha1``, ``kind: App``) and the files it names - its pages, agents,
storages and budgets.

``plan`` shows what applying the files would create, change, keep or stop and what it
costs, and returns a ``plan_token``; nothing is spent. ``apply`` applies exactly that
plan as a release. ``files`` reads an applied app's template back to change it, and
``status`` follows a release. The routes live on the agents host under
``/api/v1/agents/apps/``. Unknown answer fields are kept on the models
(``extra="allow"``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from flymyai.agents._artifacts import _body, _key

_ROOT = "/api/v1/agents/apps/"
_APP = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
_FILE_PAGE_MAX_BYTES = 200_000


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


class Apps:
    """``client.apps``: plan and apply a ``flymy.yaml``, read it back, follow a
    release."""

    def __init__(self, client: Any) -> None:
        self._c = client

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
