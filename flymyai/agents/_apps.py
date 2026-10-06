"""Versioned app/project control APIs. Writes are sent once with the caller's key.

Answers retain their ``schema`` and all additional fields. Planning is separate
from applying: the caller reviews a plan and supplies its exact plan_token.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from flymyai.agents._artifacts import _key

_APPS = "/api/v1/agents/apps/"
_PROJECTS = "/api/v1/agents/projects/"
_APP = re.compile(r"[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9._-]*")


def _source(files, app, instances, overrides) -> Dict[str, Any]:
    if (files is None) == (app is None):
        raise ValueError("pass exactly one of files or app")
    if overrides is not None and app is None:
        raise ValueError("overrides require an existing app")
    body = {"files": files} if files is not None else {"app": app}
    if instances is not None:
        body["instances"] = instances
    if overrides is not None:
        body["overrides"] = overrides
    return body


def _project(project: str) -> str:
    kind, _, value = project.partition(":")
    if kind == "app" and _APP.fullmatch(value):
        return f"{_PROJECTS}{project}/"
    if kind in ("site", "fleet"):
        return f"{_PROJECTS}{kind}:{UUID(value)}/"
    raise ValueError("project must be app:owner/name, site:UUID or fleet:UUID")


def _token(plan_token: str) -> str:
    if not isinstance(plan_token, str) or not plan_token:
        raise ValueError("plan_token must be the token of the reviewed plan")
    return plan_token


class Apps:
    def __init__(self, client: Any) -> None:
        self._client = client

    def list(self) -> Dict[str, Any]:
        return self._client._request("GET", _APPS)

    def catalog(self) -> Dict[str, Any]:
        return self._client._request("GET", _APPS + "catalog/")

    def plan(
        self,
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        app: Optional[str] = None,
        instances: Optional[Dict[str, Any]] = None,
        overrides: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self._client._request(
            "POST", _APPS + "plan/", json=_source(files, app, instances, overrides)
        )

    def apply(
        self,
        *,
        plan_token: str,
        idempotency_key: str,
        files: Optional[List[Dict[str, Any]]] = None,
        app: Optional[str] = None,
        instances: Optional[Dict[str, Any]] = None,
        overrides: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send the reviewed plan once; a stale plan remains an explicit API error."""
        body = _source(files, app, instances, overrides)
        body["plan_token"] = _token(plan_token)
        return self._client._request(
            "POST", _APPS + "apply/", json=body, headers=_key(idempotency_key)
        )

    def release(self, release_id: Any) -> Dict[str, Any]:
        return self._client._request("GET", f"{_APPS}releases/{UUID(str(release_id))}/")

    def files(
        self,
        app: str,
        *,
        path: Optional[str] = None,
        offset: int = 0,
        limit: int = 65536,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"app": app}
        if path is not None:
            params.update(path=path, offset=offset, limit=limit)
        return self._client._request("GET", _APPS + "files/", params=params)

    def remove_instance(
        self,
        app: str,
        module: str,
        key: str,
        *,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        return self._client._request(
            "POST",
            _APPS + "remove/",
            json={"app": app, "module": module, "key": key},
            headers=_key(idempotency_key),
        )


class AsyncApps:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(self) -> Dict[str, Any]:
        return await self._client._request("GET", _APPS)

    async def catalog(self) -> Dict[str, Any]:
        return await self._client._request("GET", _APPS + "catalog/")

    async def plan(
        self,
        *,
        files: Optional[List[Dict[str, Any]]] = None,
        app: Optional[str] = None,
        instances: Optional[Dict[str, Any]] = None,
        overrides: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return await self._client._request(
            "POST", _APPS + "plan/", json=_source(files, app, instances, overrides)
        )

    async def apply(
        self,
        *,
        plan_token: str,
        idempotency_key: str,
        files: Optional[List[Dict[str, Any]]] = None,
        app: Optional[str] = None,
        instances: Optional[Dict[str, Any]] = None,
        overrides: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        body = _source(files, app, instances, overrides)
        body["plan_token"] = _token(plan_token)
        return await self._client._request(
            "POST", _APPS + "apply/", json=body, headers=_key(idempotency_key)
        )

    async def release(self, release_id: Any) -> Dict[str, Any]:
        return await self._client._request(
            "GET", f"{_APPS}releases/{UUID(str(release_id))}/"
        )

    async def files(
        self,
        app: str,
        *,
        path: Optional[str] = None,
        offset: int = 0,
        limit: int = 65536,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"app": app}
        if path is not None:
            params.update(path=path, offset=offset, limit=limit)
        return await self._client._request("GET", _APPS + "files/", params=params)

    async def remove_instance(
        self,
        app: str,
        module: str,
        key: str,
        *,
        idempotency_key: str,
    ) -> Dict[str, Any]:
        return await self._client._request(
            "POST",
            _APPS + "remove/",
            json={"app": app, "module": module, "key": key},
            headers=_key(idempotency_key),
        )


def _new_project(
    name: str, template: str, description: Optional[str], budget: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "name": name, "template": template,
        **({"description": description} if description is not None else {}),
        **({"budget": budget} if budget is not None else {}),
    }


class Projects:
    def __init__(self, client: Any) -> None:
        self._client = client

    def list(self) -> Dict[str, Any]:
        return self._client._request("GET", _PROJECTS)

    def get(self, project: str) -> Dict[str, Any]:
        return self._client._request("GET", _project(project))

    def templates(self) -> Dict[str, Any]:
        """Gallery metadata, including optional public demo_url and thumbnail preview_url."""
        return self._client._request("GET", _PROJECTS + "templates/")

    def plan(
        self, name: str, *, template: str = "empty",
        description: Optional[str] = None, budget: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Preview an empty, gallery or link:<handle> project without spending."""
        return self._client._request(
            "POST", _PROJECTS,
            json=_new_project(name, template, description, budget),
        )

    def create(
        self, name: str, *, plan_token: str, idempotency_key: str,
        template: str = "empty", description: Optional[str] = None,
        budget: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Apply the reviewed plan once; link copies are private with empty stores."""
        body = _new_project(name, template, description, budget)
        body["plan_token"] = _token(plan_token)
        return self._client._request(
            "POST", _PROJECTS, json=body, headers=_key(idempotency_key),
        )

    def errors(
        self, project: str, *, before: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        return self._client._request(
            "GET",
            _project(project) + "errors/",
            params={
                "limit": limit,
                **({"before": before} if before is not None else {}),
            },
        )

    def stop(self, project: str, *, idempotency_key: str) -> Dict[str, Any]:
        return self._client._request(
            "POST", _project(project) + "stop/", json={}, headers=_key(idempotency_key)
        )

    def start_plan(self, project: str) -> Dict[str, Any]:
        return self._client._request("POST", _project(project) + "start/", json={})

    def start(
        self, project: str, *, plan_token: str, idempotency_key: str
    ) -> Dict[str, Any]:
        return self._client._request(
            "POST",
            _project(project) + "start/",
            json={"plan_token": _token(plan_token)},
            headers=_key(idempotency_key),
        )


class AsyncProjects:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(self) -> Dict[str, Any]:
        return await self._client._request("GET", _PROJECTS)

    async def get(self, project: str) -> Dict[str, Any]:
        return await self._client._request("GET", _project(project))

    async def templates(self) -> Dict[str, Any]:
        """Gallery metadata, including optional public demo_url and thumbnail preview_url."""
        return await self._client._request("GET", _PROJECTS + "templates/")

    async def plan(
        self, name: str, *, template: str = "empty",
        description: Optional[str] = None, budget: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Preview an empty, gallery or link:<handle> project without spending."""
        return await self._client._request(
            "POST", _PROJECTS,
            json=_new_project(name, template, description, budget),
        )

    async def create(
        self, name: str, *, plan_token: str, idempotency_key: str,
        template: str = "empty", description: Optional[str] = None,
        budget: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Apply the reviewed plan once; link copies are private with empty stores."""
        body = _new_project(name, template, description, budget)
        body["plan_token"] = _token(plan_token)
        return await self._client._request(
            "POST", _PROJECTS, json=body, headers=_key(idempotency_key),
        )

    async def errors(
        self, project: str, *, before: Optional[str] = None, limit: int = 50
    ) -> Dict[str, Any]:
        return await self._client._request(
            "GET",
            _project(project) + "errors/",
            params={
                "limit": limit,
                **({"before": before} if before is not None else {}),
            },
        )

    async def stop(self, project: str, *, idempotency_key: str) -> Dict[str, Any]:
        return await self._client._request(
            "POST", _project(project) + "stop/", json={}, headers=_key(idempotency_key)
        )

    async def start_plan(self, project: str) -> Dict[str, Any]:
        return await self._client._request(
            "POST", _project(project) + "start/", json={}
        )

    async def start(
        self, project: str, *, plan_token: str, idempotency_key: str
    ) -> Dict[str, Any]:
        return await self._client._request(
            "POST",
            _project(project) + "start/",
            json={"plan_token": _token(plan_token)},
            headers=_key(idempotency_key),
        )
