"""Projects (``flymy.project.v1``): what you build and run as one unit - an app applied
from one ``flymy.yaml``, a page published on its own, or a fleet (a lead agent and the
agents it starts).

A project is its config: creating it, stopping and starting it are a plan and an apply
of its ``flymy.yaml``, and each leaves a release. A plan creates nothing and returns a
``plan_token``; show its price to the user, then create or start with that token and an
idempotency key. The routes live on the agents host under ``/api/v1/agents/projects/``
(backend ``docs/protocols/PROJECTS_API.md``). Unknown answer fields are kept on the
models (``extra="allow"``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from flymyai.agents._artifacts import _body, _key

_ROOT = "/api/v1/agents/projects/"
_APP_ID = re.compile(r"app:([a-z0-9][a-z0-9-]*)/([a-z0-9][a-z0-9._-]*)")
_SITE_ID = re.compile(
    r"site:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_FLEET_ID = re.compile(
    r"fleet:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_ID_MAX_CHARS = 200
_ERRORS_LIMIT_MAX = 200


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)


class Project(_Model):
    """A project (``flymy.project.v1``): ``id``, ``kind`` (app, site or fleet),
    ``name``, ``status``, ``lifecycle``, ``money``, ``errors``, ``budget`` and, read
    by id, its pages, agents, servers, storages and diagram."""

    id: str
    kind: str
    name: str
    status: Optional[str] = None


class ProjectList(_Model):
    """Your apps and pages, then your fleets; ``modules`` names what this platform
    serves."""

    modules: List[str] = []
    projects: List[Project] = []


class ProjectTemplates(_Model):
    """The template gallery (``flymy.project-templates.v1``)."""

    templates: List[Dict[str, Any]] = []


class ProjectPlan(_Model):
    """What creating or starting would do and cost (``plan["usd_per_hour"]``, its
    ``changes`` and ``budgets``), and the ``plan_token`` that applies exactly this
    plan. Nothing was created or changed."""

    project: Optional[str] = None
    plan: Dict[str, Any] = {}
    plan_token: Optional[str] = None


class ProjectCreated(_Model):
    """The created project's id and its first release, applying."""

    project: str
    release: Dict[str, Any] = {}


class ProjectErrors(_Model):
    """One page of a project's errors journal (``flymy.project-errors.v1``), newest
    first; pass ``next_before`` back as ``before`` for older entries."""

    project: Optional[str] = None
    errors: List[Dict[str, Any]] = []
    next_before: Optional[str] = None


class ProjectAgent(_Model):
    """The agent in charge of a project, created on first use."""

    project: str
    agent: str
    name: Optional[str] = None
    model: Optional[str] = None
    created: Optional[bool] = None


def _path(project_id: Any, route: str = "", *, fleets: bool = True) -> str:
    """The route of a project id from ``list()``; a fleet has only its detail and its
    errors journal."""

    value = str(project_id)
    if len(value) <= _ID_MAX_CHARS:
        app = _APP_ID.fullmatch(value)
        if app:
            return f"{_ROOT}app:{app.group(1)}/{app.group(2)}/{route}"
        site = _SITE_ID.fullmatch(value)
        if site:
            return f"{_ROOT}site:{site.group(1)}/{route}"
        fleet = _FLEET_ID.fullmatch(value)
        if fleet:
            if not fleets:
                raise ValueError(
                    "a fleet has no project agent and is not stopped or started:"
                    " talk to its lead"
                )
            return f"{_ROOT}fleet:{fleet.group(1)}/{route}"
    raise ValueError(
        "project_id must be an id from list(): app:<owner>/<name>, site:<page id>"
        " or fleet:<lead agent id>"
    )


def _errors_query(limit: Optional[int], before: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if limit is not None:
        if not 1 <= int(limit) <= _ERRORS_LIMIT_MAX:
            raise ValueError("limit must be 1-200")
        params["limit"] = int(limit)
    if before is not None:
        params["before"] = before
    return params


def _project_body(
    name: str,
    template: Optional[str],
    description: Optional[str],
    budget: Optional[Dict[str, str]],
    plan_token: Optional[str] = None,
) -> Dict[str, Any]:
    return _body(
        name=name,
        description=description,
        template=template,
        budget=budget,
        plan_token=plan_token,
    )


def _start_body(
    plan_token: Optional[str], idempotency_key: Optional[str]
) -> Dict[str, Any]:
    if plan_token is None:
        if idempotency_key is not None:
            raise ValueError(
                "start() without plan_token only returns the plan: call it without an"
                " idempotency_key, then again with the plan_token and a key"
            )
        return {"json": {}}
    if not plan_token or idempotency_key is None:
        raise ValueError("start() with a plan_token needs an idempotency_key")
    return {"json": {"plan_token": plan_token}, "headers": _key(idempotency_key)}


class Projects:
    """``client.projects``: your apps, pages and fleets as projects."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def list(self) -> ProjectList:
        """Your apps and pages, then your fleets, with status, money and errors."""
        return ProjectList.model_validate(self._c._request("GET", _ROOT))

    def get(self, project_id: str) -> Project:
        """One project by its id from :meth:`list`."""
        return Project.model_validate(self._c._request("GET", _path(project_id)))

    def errors(
        self,
        project_id: str,
        *,
        limit: Optional[int] = None,
        before: Optional[str] = None,
    ) -> ProjectErrors:
        """A page of the project's errors journal, newest first, last 30 days."""
        data = self._c._request(
            "GET", _path(project_id, "errors/"), params=_errors_query(limit, before)
        )
        return ProjectErrors.model_validate(data)

    def templates(self) -> ProjectTemplates:
        """The gallery a project starts from (``template`` ids)."""
        return ProjectTemplates.model_validate(
            self._c._request("GET", f"{_ROOT}templates/")
        )

    def plan(
        self,
        *,
        name: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        budget: Optional[Dict[str, str]] = None,
    ) -> ProjectPlan:
        """What creating this project would make and cost; creates nothing."""
        body = _project_body(name, template, description, budget)
        return ProjectPlan.model_validate(self._c._request("POST", _ROOT, json=body))

    def create(
        self,
        *,
        name: str,
        plan_token: str,
        idempotency_key: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        budget: Optional[Dict[str, str]] = None,
    ) -> ProjectCreated:
        """Create exactly the plan the user confirmed: send the same fields as
        :meth:`plan` plus its ``plan_token``. Returns the project id and its first
        release."""
        body = _project_body(name, template, description, budget, plan_token)
        data = self._c._request("POST", _ROOT, json=body, headers=_key(idempotency_key))
        return ProjectCreated.model_validate(data)

    def agent(self, project_id: str) -> ProjectAgent:
        """The agent in charge of an app or a page, created on first use."""
        data = self._c._request("POST", _path(project_id, "agent/", fleets=False))
        return ProjectAgent.model_validate(data)

    def stop(self, project_id: str, *, idempotency_key: str) -> Project:
        """Stop an app or a page: a release with ``runtime.state: stopped``; its
        servers stop and the rest of their holds is refunded."""
        data = self._c._request(
            "POST",
            _path(project_id, "stop/", fleets=False),
            json={},
            headers=_key(idempotency_key),
        )
        return Project.model_validate(data)

    def start(
        self,
        project_id: str,
        *,
        plan_token: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        """Without ``plan_token``: the plan of the release that starts it (nothing
        changes). With the token the user confirmed and an ``idempotency_key``: the
        project, that release applying."""
        data = self._c._request(
            "POST",
            _path(project_id, "start/", fleets=False),
            **_start_body(plan_token, idempotency_key),
        )
        if plan_token is None:
            return ProjectPlan.model_validate(data)
        return Project.model_validate(data)


class AsyncProjects:
    """``client.projects`` on the async client."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def list(self) -> ProjectList:
        return ProjectList.model_validate(await self._c._request("GET", _ROOT))

    async def get(self, project_id: str) -> Project:
        data = await self._c._request("GET", _path(project_id))
        return Project.model_validate(data)

    async def errors(
        self,
        project_id: str,
        *,
        limit: Optional[int] = None,
        before: Optional[str] = None,
    ) -> ProjectErrors:
        data = await self._c._request(
            "GET", _path(project_id, "errors/"), params=_errors_query(limit, before)
        )
        return ProjectErrors.model_validate(data)

    async def templates(self) -> ProjectTemplates:
        data = await self._c._request("GET", f"{_ROOT}templates/")
        return ProjectTemplates.model_validate(data)

    async def plan(
        self,
        *,
        name: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        budget: Optional[Dict[str, str]] = None,
    ) -> ProjectPlan:
        body = _project_body(name, template, description, budget)
        data = await self._c._request("POST", _ROOT, json=body)
        return ProjectPlan.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        plan_token: str,
        idempotency_key: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        budget: Optional[Dict[str, str]] = None,
    ) -> ProjectCreated:
        body = _project_body(name, template, description, budget, plan_token)
        data = await self._c._request(
            "POST", _ROOT, json=body, headers=_key(idempotency_key)
        )
        return ProjectCreated.model_validate(data)

    async def agent(self, project_id: str) -> ProjectAgent:
        data = await self._c._request("POST", _path(project_id, "agent/", fleets=False))
        return ProjectAgent.model_validate(data)

    async def stop(self, project_id: str, *, idempotency_key: str) -> Project:
        data = await self._c._request(
            "POST",
            _path(project_id, "stop/", fleets=False),
            json={},
            headers=_key(idempotency_key),
        )
        return Project.model_validate(data)

    async def start(
        self,
        project_id: str,
        *,
        plan_token: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        data = await self._c._request(
            "POST",
            _path(project_id, "start/", fleets=False),
            **_start_body(plan_token, idempotency_key),
        )
        if plan_token is None:
            return ProjectPlan.model_validate(data)
        return Project.model_validate(data)
