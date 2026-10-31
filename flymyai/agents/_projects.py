"""Projects (``flymy.project.v1``): what you build and run as one unit - an app applied
from one ``flymy.yaml``, a page published on its own, a fleet (a lead agent and the
agents it starts) or a frontend artifact.

A project is its config: creating it, stopping and starting it, renaming it and
changing its budget are a plan and an apply of its ``flymy.yaml``, and each leaves a
release. A plan creates nothing and returns a ``plan_token``; show its price to the
user, then create or start with that token and an idempotency key. Deleting it removes
a project with no history and archives one with history (``flymy.project-edit.v1``).
The routes live on the agents host under ``/api/v1/agents/projects/`` (backend
``docs/protocols/PROJECTS_API.md``). Unknown answer fields are kept on the models
(``extra="allow"``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

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
_ARTIFACT_ID = re.compile(
    r"artifact:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_ID_MAX_CHARS = 200
_ERRORS_LIMIT_MAX = 200
# A project's link is its page in the app with the id URI-encoded (":" as %3A, "/" as
# %2F), as Share copies it: only an app or a fleet, with exactly the id shapes above and
# no other escape, held to an id's length (backend docs/protocols/SHARING.md).
_LINK_PREFIX = "https://app.flymy.ai/projects/"
_LINK = re.compile(
    r"https://app\.flymy\.ai/projects/"
    r"(?:app(?::|%3[Aa])[a-z0-9][a-z0-9-]*(?:/|%2[Ff])[a-z0-9][a-z0-9._-]*"
    r"|fleet(?::|%3[Aa])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/?"
)
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,39}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)


class Project(_Model):
    """A project (``flymy.project.v1``): ``id``, ``kind`` (app, site, fleet or
    artifact), ``name``, ``status``, ``lifecycle``, ``money``, ``errors``, ``budget``
    and, read by id, its pages, agents, servers, storages and diagram."""

    id: str
    kind: str
    name: str
    status: Optional[str] = None


class ProjectList(_Model):
    """Your apps and pages, then your fleets and your artifacts; ``modules`` names what
    this platform serves."""

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


class ProjectCopy(_Model):
    """A project copied into your account: ``project`` (its ``id``, ``name`` and
    ``url``), the ``agents`` made, and ``setup`` - ``needs`` (connections, secrets,
    inputs and schedules the copy still needs; with ``connect_later``, ``to_connect``:
    each service still to connect with its ``connect_url``) and ``chat``, the execution
    id of the chat that finishes the setup with you (the Transfer agent's, or an
    official template's welcome chat)."""

    project: Dict[str, Any] = {}
    agents: List[Dict[str, Any]] = []
    setup: Dict[str, Any] = {}


class ProjectAgent(_Model):
    """The agent in charge of a project, created on first use."""

    project: str
    agent: str
    name: Optional[str] = None
    model: Optional[str] = None
    created: Optional[bool] = None


class ProjectEdit(_Model):
    """A project changed (``flymy.project-edit.v1``): what changed - ``title`` (the
    name it lists under), ``budget`` (the project budget after it: ``active`` and
    ``day``, ``week``, ``month`` with ``cap_usd``, ``spent_usd``, ``held_usd`` and
    ``left_usd``) or ``deleted`` (``removed`` or ``archived``) with ``at`` - and the
    ``release`` that made it, ``None`` when nothing had to change."""

    project: str
    release: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    budget: Optional[Dict[str, Any]] = None
    deleted: Optional[str] = None
    at: Optional[str] = None


def _agent_body(model: Optional[str]) -> Dict[str, Any]:
    """The project agent request: no body keeps its model; ``{"model"}`` picks one."""
    if model is None:
        return {}
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be an agent model id from list_agent_models.")
    return {"json": {"model": model.strip()}}


def _path(project_id: Any, route: str = "", *, fleets: bool = True) -> str:
    """The route of a project id from ``list()``; a fleet and an artifact have only
    their detail and their errors journal."""

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
        artifact = _ARTIFACT_ID.fullmatch(value)
        if artifact:
            if not fleets:
                raise ValueError(
                    "an artifact has no project agent and is not stopped or started:"
                    " change it with client.artifacts"
                )
            return f"{_ROOT}artifact:{artifact.group(1)}/{route}"
    raise ValueError(
        "project_id must be an id from list(): app:<owner>/<name>, site:<page id>,"
        " fleet:<lead agent id> or artifact:<artifact id>"
    )


def _shared_path(project: Any, route: str) -> str:
    """The ``public/`` or ``copy/`` route of an app or a fleet, by id or by its link."""
    value = str(project)
    if len(value) <= _ID_MAX_CHARS and _LINK.fullmatch(value):
        value = unquote(value[len(_LINK_PREFIX) :].rstrip("/"))
    if len(value) <= _ID_MAX_CHARS:
        app = _APP_ID.fullmatch(value)
        if app:
            return f"{_ROOT}app:{app.group(1)}/{app.group(2)}/{route}"
        fleet = _FLEET_ID.fullmatch(value)
        if fleet:
            return f"{_ROOT}fleet:{fleet.group(1)}/{route}"
        if _SITE_ID.fullmatch(value):
            raise ValueError(
                "a page on its own is shared and copied as a site (share, remix)"
            )
        if _ARTIFACT_ID.fullmatch(value):
            raise ValueError("an artifact is shared and copied with client.artifacts")
    raise ValueError(
        "project must be an app or a fleet - app:<owner>/<name> or"
        " fleet:<lead agent id> - or its link https://app.flymy.ai/projects/<id>"
    )


def _share_body(public: Any) -> Dict[str, Any]:
    if not isinstance(public, bool):
        raise ValueError("public must be True or False")
    return {"json": {"public": public}}


def _copy_body(
    name: Optional[str], idempotency_key: str, connect_later: bool = False
) -> Dict[str, Any]:
    if name is not None and not (isinstance(name, str) and _NAME.fullmatch(name)):
        raise ValueError(
            "name must be 2-40 lowercase letters, digits and dashes,"
            " starting with a letter or digit"
        )
    if not isinstance(connect_later, bool):
        raise ValueError("connect_later must be True or False")
    body: Dict[str, Any] = {"name": name} if name is not None else {}
    if connect_later:
        body["connect_later"] = True
    return {"json": body, "headers": _key(idempotency_key)}


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


_PERIODS = ("per_day_usd", "per_week_usd", "per_month_usd")
# What each change takes (backend PROJECTS_API.md, "Change a project"): an app, and
# for a rename a page on its own too; any other kind is refused with where it changes
# instead.
_EDIT_REFUSALS = {
    "rename": {
        "fleet": (
            "a fleet without a named AgentTeam uses its lead's name: rename the lead"
            " with agents.update; a named AgentTeam has no rename endpoint"
        ),
        "artifact": "an artifact is renamed with client.artifacts.update",
        "invalid": (
            "project_id must be an id from list(): app:<owner>/<name> or site:<page id>"
        ),
    },
    "budget": {
        "site": (
            "a page on its own has no project budget: only an app (app:<owner>/<name>)"
            " has one"
        ),
        "fleet": (
            "a fleet has no project budget: its lead starts its agents within a Team's"
            " run cap or a chat's subagent limits"
        ),
        "artifact": "an artifact has no project budget",
        "invalid": "project_id must be an app id from list(): app:<owner>/<name>",
    },
    "delete": {
        "site": "a page on its own is not deleted here: stop it with stop()",
        "fleet": (
            "a fleet is not deleted as a whole: delete its agents with agents.delete"
        ),
        "artifact": "an artifact is deleted with client.artifacts.delete",
        "invalid": "project_id must be an app id from list(): app:<owner>/<name>",
    },
}


def _edit_path(project_id: Any, edit: str) -> str:
    """The route of a change: an app's, and for a rename a page's on its own too."""
    refusals = _EDIT_REFUSALS[edit]
    value = str(project_id)
    if len(value) <= _ID_MAX_CHARS:
        if _APP_ID.fullmatch(value) or (
            "site" not in refusals and _SITE_ID.fullmatch(value)
        ):
            return _path(value, "budget/" if edit == "budget" else "")
        for kind, shape in (
            ("site", _SITE_ID),
            ("fleet", _FLEET_ID),
            ("artifact", _ARTIFACT_ID),
        ):
            if kind in refusals and shape.fullmatch(value):
                raise ValueError(refusals[kind])
    raise ValueError(refusals["invalid"])


def _title_body(title: Optional[str], idempotency_key: str) -> Dict[str, Any]:
    if title is not None and not isinstance(title, str):
        raise ValueError(
            "title must be a string of 1 to 80 characters, or None to drop it"
        )
    return {"json": {"title": title}, "headers": _key(idempotency_key)}


def _budget_body(
    budget: Dict[str, Optional[str]], idempotency_key: str
) -> Dict[str, Any]:
    if not isinstance(budget, dict) or not budget or set(budget) - set(_PERIODS):
        raise ValueError(
            "budget is {per_day_usd, per_week_usd, per_month_usd}: an amount sets that"
            " period, None removes it, a period left out stays"
        )
    return {"json": dict(budget), "headers": _key(idempotency_key)}


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

    def agent(self, project_id: str, *, model: Optional[str] = None) -> ProjectAgent:
        """The agent in charge of an app or a page, created on first use.

        ``model`` picks its model: an id from ``GET /api/v1/agents/llm-models/``
        (``list_agent_models`` in the Agents MCP). It stays the agent's model; omit it to
        keep the current one (a new agent starts on the platform's top model). An id the
        platform does not offer for agents is refused with 400 ``bad_model``.
        """
        data = self._c._request(
            "POST", _path(project_id, "agent/", fleets=False), **_agent_body(model)
        )
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

    def share(self, project_id: str, *, public: bool) -> Project:
        """Make an app or a fleet public or private, as an agent's public flag.

        ``public=True``: anyone with its link, ``https://app.flymy.ai/projects/<id>``,
        sees what it holds - its agents, pages and what a copy needs to connect, never
        its data, conversations, money or connections - and copies it with
        :meth:`copy`. ``public=False``: only you; copies already made stay theirs. A
        fleet shares through its lead. Safe to repeat."""
        data = self._c._request(
            "POST", _shared_path(project_id, "public/"), **_share_body(public)
        )
        return Project.model_validate(data)

    def copy(
        self,
        project: str,
        *,
        idempotency_key: str,
        name: Optional[str] = None,
        connect_later: bool = False,
    ) -> ProjectCopy:
        """Copy a public project - by its link or id - or one of yours into your
        account, as an agent copies: its agents (a lead with all its subagents, all
        or none), pages and config under your own budget; never its data,
        conversations, money or connections. ``name`` defaults to the source's (with
        -2, -3 if you have one). ``connect_later=True``: a toolkit you have not
        connected becomes your own connection to set up instead of a 422 - listed in
        ``setup["needs"]["to_connect"]`` with its ``connect_url``, and the setup chat
        asks for it first. Refused before anything is created: 422 ``missing_tools``
        (without ``connect_later``, or a toolkit it cannot add), 404 (private or
        gone), 402 (balance), 409 ``project_exists`` (only with an explicit
        ``name``)."""
        data = self._c._request(
            "POST",
            _shared_path(project, "copy/"),
            **_copy_body(name, idempotency_key, connect_later),
        )
        return ProjectCopy.model_validate(data)

    def rename(
        self, project_id: str, *, title: Optional[str], idempotency_key: str
    ) -> ProjectEdit:
        """Rename an app or a page on its own: ``title`` is the name it lists under, 1
        to 80 characters (runs of spaces fold into one); ``None`` drops it, and it lists
        by its name again. An app's ``metadata.title`` changes by a release planned and
        applied at once - its id keeps its name, and ``release`` is ``None`` when the
        title already is so; a page takes it as its name, with no release. Refused
        before anything changed: 400 ``bad_value``, 409 ``busy`` (a release is
        applying) or ``project_stopped`` (start it first)."""
        data = self._c._request(
            "PATCH",
            _edit_path(project_id, "rename"),
            **_title_body(title, idempotency_key),
        )
        return ProjectEdit.model_validate(data)

    def set_budget(
        self,
        project_id: str,
        budget: Dict[str, Optional[str]],
        *,
        idempotency_key: str,
    ) -> ProjectEdit:
        """Change an app's project budget (``billing.budgets.project``). In ``budget``
        an amount in USD (above 0, at most 10000) sets ``per_day_usd``,
        ``per_week_usd`` or ``per_month_usd``, ``None`` removes that period, and a
        period left out stays; at least one stays. A release planned and applied at
        once: money spent and held stays, only the caps change. Returns the budget
        after it; ``release`` is ``None`` when it already is so. Refused before
        anything changed: 400 ``invalid_budget``, 409 ``busy`` or ``project_stopped``.
        """
        data = self._c._request(
            "PATCH",
            _edit_path(project_id, "budget"),
            **_budget_body(budget, idempotency_key),
        )
        return ProjectEdit.model_validate(data)

    def delete(self, project_id: str, *, idempotency_key: str) -> ProjectEdit:
        """Delete an app. One with no history - nothing ran, was charged, called or
        stored - is removed for good: its releases, budgets, pages and links go and
        its name is free again. One with history is archived: it stops as
        :meth:`stop` does, its agents are archived (their schedules and triggers go
        off), its public link closes and it leaves the lists; its runs, ledger
        entries, releases, storages and name stay. ``deleted`` says which; the same
        key answers the same again. 409 ``busy`` while a release applies."""
        data = self._c._request(
            "DELETE", _edit_path(project_id, "delete"), headers=_key(idempotency_key)
        )
        return ProjectEdit.model_validate(data)


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

    async def agent(
        self, project_id: str, *, model: Optional[str] = None
    ) -> ProjectAgent:
        data = await self._c._request(
            "POST", _path(project_id, "agent/", fleets=False), **_agent_body(model)
        )
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

    async def share(self, project_id: str, *, public: bool) -> Project:
        data = await self._c._request(
            "POST", _shared_path(project_id, "public/"), **_share_body(public)
        )
        return Project.model_validate(data)

    async def copy(
        self,
        project: str,
        *,
        idempotency_key: str,
        name: Optional[str] = None,
        connect_later: bool = False,
    ) -> ProjectCopy:
        data = await self._c._request(
            "POST",
            _shared_path(project, "copy/"),
            **_copy_body(name, idempotency_key, connect_later),
        )
        return ProjectCopy.model_validate(data)

    async def rename(
        self, project_id: str, *, title: Optional[str], idempotency_key: str
    ) -> ProjectEdit:
        data = await self._c._request(
            "PATCH",
            _edit_path(project_id, "rename"),
            **_title_body(title, idempotency_key),
        )
        return ProjectEdit.model_validate(data)

    async def set_budget(
        self,
        project_id: str,
        budget: Dict[str, Optional[str]],
        *,
        idempotency_key: str,
    ) -> ProjectEdit:
        data = await self._c._request(
            "PATCH",
            _edit_path(project_id, "budget"),
            **_budget_body(budget, idempotency_key),
        )
        return ProjectEdit.model_validate(data)

    async def delete(self, project_id: str, *, idempotency_key: str) -> ProjectEdit:
        data = await self._c._request(
            "DELETE", _edit_path(project_id, "delete"), headers=_key(idempotency_key)
        )
        return ProjectEdit.model_validate(data)
