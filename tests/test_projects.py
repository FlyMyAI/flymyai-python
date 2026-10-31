import asyncio
import json
import re

import httpx
import pytest

from flymyai.agents import (
    AsyncProjects,
    Project,
    ProjectCreated,
    ProjectEdit,
    ProjectPlan,
    Projects,
)
from flymyai.agents._client import AsyncAgentClient, FlyMyAIAgentError
from tests.test_mcp_teams import client_with, response

APP_ID = "app:denis/paint-arena"
FLEET_ID = "fleet:0b9c6f36-3f1f-4f5b-9a55-1f2a7c1f0e11"
ARTIFACT_ID = "artifact:9d3f6a2e-1b4c-4e7d-8a5f-0c2b7e9d1a36"
PROJECT = {
    "schema": "flymy.project.v1",
    "id": APP_ID,
    "kind": "app",
    "name": "paint-arena",
    "status": "active",
    "lifecycle": {"state": "running", "actions": {"stop": True, "start": False}},
    "money": {"today": {"usd": "0.12"}},
}
PLAN = {
    "schema": "flymy.project-create.v1",
    "project": APP_ID,
    "plan": {
        "release": 1,
        "summary": {"pages": 1, "agents": 2, "servers": 0, "storages": 1},
        "changes": [],
        "usd_per_hour": "0.000000",
        "budgets": {"per_day_usd": "5"},
    },
    "plan_token": "plan-1",
}


def test_lists_reads_and_pages_the_journal_with_the_owners_key():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        assert request.headers["X-API-KEY"] == "personal-key"
        if request.url.path == "/api/v1/agents/projects/":
            return response({
                "schema": "flymy.project.v1",
                "modules": ["apps", "pages", "fleets"],
                "projects": [PROJECT],
            })
        if request.url.path.endswith("/errors/"):
            return response({
                "schema": "flymy.project-errors.v1",
                "project": FLEET_ID,
                "errors": [{"kind": "agent_run", "code": "timeout", "count": 2}],
                "next_before": None,
            })
        return response({**PROJECT, "graph": {"nodes": []}})

    client = client_with(handler)
    listed = client.projects.list()
    one = client.projects.get(APP_ID)
    journal = client.projects.errors(FLEET_ID, limit=5, before="2026-10-01")

    assert listed.modules == ["apps", "pages", "fleets"]
    assert listed.projects[0].id == APP_ID
    # a field the backend adds within v1 is kept
    assert one.graph == {"nodes": []} and one.money["today"]["usd"] == "0.12"
    assert journal.errors[0]["count"] == 2 and journal.next_before is None
    assert seen == [
        ("GET", "/api/v1/agents/projects/", {}),
        ("GET", "/api/v1/agents/projects/app:denis/paint-arena/", {}),
        (
            "GET",
            f"/api/v1/agents/projects/{FLEET_ID}/errors/",
            {"limit": "5", "before": "2026-10-01"},
        ),
    ]


def test_an_artifact_is_a_project_read_by_its_id():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith("/errors/"):
            return response({"project": ARTIFACT_ID, "errors": [], "next_before": None})
        return response({"id": ARTIFACT_ID, "kind": "artifact", "name": "Pod racer"})

    client = client_with(handler)
    assert client.projects.get(ARTIFACT_ID).kind == "artifact"
    assert client.projects.errors(ARTIFACT_ID).errors == []
    assert seen == [
        f"/api/v1/agents/projects/{ARTIFACT_ID}/",
        f"/api/v1/agents/projects/{ARTIFACT_ID}/errors/",
    ]


def test_a_plan_creates_nothing_and_the_create_applies_it_with_the_callers_key():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.headers.get("Idempotency-Key"), body))
        if "plan_token" in body:
            return response(
                {
                    "schema": "flymy.project-create.v1",
                    "project": APP_ID,
                    "release": {"release": 1, "status": "applying"},
                },
                202,
            )
        return response(PLAN)

    client = client_with(handler)
    plan = client.projects.plan(name="paint-arena", budget={"per_day_usd": "5"})
    made = client.projects.create(
        name="paint-arena",
        budget={"per_day_usd": "5"},
        plan_token=plan.plan_token,
        idempotency_key="create-paint-arena-1",
    )

    assert isinstance(plan, ProjectPlan) and plan.plan["usd_per_hour"] == "0.000000"
    assert isinstance(made, ProjectCreated) and made.project == APP_ID
    assert made.release["status"] == "applying"
    assert seen == [
        (None, {"name": "paint-arena", "budget": {"per_day_usd": "5"}}),
        (
            "create-paint-arena-1",
            {
                "name": "paint-arena",
                "budget": {"per_day_usd": "5"},
                "plan_token": "plan-1",
            },
        ),
    ]


def test_stop_and_a_two_step_start():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers.get("Idempotency-Key"), body))
        if request.url.path.endswith("/start/") and "plan_token" not in body:
            return response({
                "schema": "flymy.project-lifecycle.v1",
                "project": APP_ID,
                "action": "start",
                "plan": {"state": "running", "usd_per_hour": "0.050000"},
                "plan_token": "start-plan-1",
            })
        return response({**PROJECT, "status": "stopped"})

    client = client_with(handler)
    stopped = client.projects.stop(APP_ID, idempotency_key="stop-1")
    preview = client.projects.start(APP_ID)
    started = client.projects.start(
        APP_ID, plan_token=preview.plan_token, idempotency_key="start-1"
    )

    assert isinstance(stopped, Project) and stopped.status == "stopped"
    assert isinstance(preview, ProjectPlan) and preview.action == "start"
    assert preview.plan["usd_per_hour"] == "0.050000"
    assert isinstance(started, Project)
    root = "/api/v1/agents/projects/app:denis/paint-arena/"
    assert seen == [
        (f"{root}stop/", "stop-1", {}),
        (f"{root}start/", None, {}),
        (f"{root}start/", "start-1", {"plan_token": "start-plan-1"}),
    ]


def test_the_project_agent_keeps_or_picks_its_model():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        return response({
            "schema": "flymy.project-agent.v1",
            "project": APP_ID,
            "agent": "4c1f2e7a-0000-4000-8000-000000000001",
            "name": "paint-arena agent",
            "model": (body or {}).get("model", "claude-opus-5-5"),
            "created": False,
        })

    client = client_with(handler)
    kept = client.projects.agent(APP_ID)
    picked = client.projects.agent(APP_ID, model="gpt-6-luna")

    path = "/api/v1/agents/projects/app:denis/paint-arena/agent/"
    assert seen == [("POST", path, None), ("POST", path, {"model": "gpt-6-luna"})]
    assert kept.model == "claude-opus-5-5" and picked.model == "gpt-6-luna"


def test_a_refused_model_comes_back_as_the_api_error():
    def handler(request):
        return response(
            {"code": "bad_model", "message": "x is not offered for agents now"},
            status=400,
        )

    with pytest.raises(Exception) as caught:
        client_with(handler).projects.agent(APP_ID, model="x")
    assert "bad_model" in str(caught.value) or "not offered" in str(caught.value)


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.projects.get("app:denis/../x"),
        lambda c: c.projects.get("denis/paint-arena"),
        lambda c: c.projects.errors(APP_ID, limit=0),
        # a fleet has no project agent and is not stopped or started: talk to its lead
        lambda c: c.projects.agent(FLEET_ID),
        lambda c: c.projects.stop(FLEET_ID, idempotency_key="stop-1"),
        # an artifact changes through client.artifacts
        lambda c: c.projects.agent(ARTIFACT_ID),
        lambda c: c.projects.start(ARTIFACT_ID),
        lambda c: c.projects.get(ARTIFACT_ID.upper()),
        # a start without plan_token is only the plan, so it takes no key
        lambda c: c.projects.start(APP_ID, idempotency_key="start-1"),
        lambda c: c.projects.start(APP_ID, plan_token="start-plan-1"),
        # printable ASCII with spaces inside only, as the backend takes it
        lambda c: c.projects.stop(APP_ID, idempotency_key=" bad key"),
        # a model is an id from list_agent_models, never blank
        lambda c: c.projects.agent(APP_ID, model="  "),
        # a change is a durable write: its key follows the same rule
        lambda c: c.projects.rename(APP_ID, title="x", idempotency_key=" bad key"),
        lambda c: c.projects.delete(APP_ID, idempotency_key=""),
    ],
)
def test_a_bad_id_or_call_is_refused_before_any_request(call):
    def handler(request):
        raise AssertionError(f"unexpected request {request.url}")

    with pytest.raises(ValueError):
        call(client_with(handler))


def test_the_async_client_has_the_same_projects():
    def handler(request):
        if request.method == "GET":
            return response({"modules": ["apps"], "projects": [PROJECT]})
        return response(PLAN)

    async def go():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            listed = await client.projects.list()
            plan = await client.projects.plan(name="paint-arena")
        finally:
            await client.close()
        return listed, plan

    listed, plan = asyncio.run(go())
    assert listed.projects[0].name == "paint-arena" and plan.plan_token == "plan-1"
    sync = {name for name in dir(Projects) if not name.startswith("_")}
    assert sync == {name for name in dir(AsyncProjects) if not name.startswith("_")}


LINK = "https://app.flymy.ai/projects/app%3Adenis%2Fpaint-arena"
COPIED = {
    "schema": "flymy.project.v1",
    "project": {"id": "app:me/paint-arena", "name": "paint-arena"},
    "agents": [{"uuid": "a1", "name": "lead"}],
    "setup": {"chat": "abc-defg-hij", "needs": {"connections": ["telegram_bot"]}},
}


def test_share_and_copy_a_project_like_an_agent_by_id_or_link():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, request.headers.get("Idempotency-Key"), body))
        if request.url.path.endswith("/copy/"):
            return response(COPIED)
        return response({**PROJECT, "public": True})

    client = client_with(handler)
    shared = client.projects.share(APP_ID, public=True)
    assert shared.public is True
    copy = client.projects.copy(LINK, idempotency_key="copy-1", name="arena-copy")
    assert copy.project["id"] == "app:me/paint-arena"
    assert copy.setup["chat"] == "abc-defg-hij"
    client.projects.copy(FLEET_ID, idempotency_key="copy-2")
    client.projects.share(
        "https://app.flymy.ai/projects/fleet%3a" + FLEET_ID[6:] + "/", public=False
    )
    assert seen == [
        (
            "/api/v1/agents/projects/app:denis/paint-arena/public/",
            None,
            {"public": True},
        ),
        (
            "/api/v1/agents/projects/app:denis/paint-arena/copy/",
            "copy-1",
            {"name": "arena-copy"},
        ),
        (f"/api/v1/agents/projects/{FLEET_ID}/copy/", "copy-2", {}),
        (f"/api/v1/agents/projects/{FLEET_ID}/public/", None, {"public": False}),
    ]


@pytest.mark.parametrize(
    "project",
    [
        "https://evil.example/projects/app%3Adenis%2Fpaint-arena",
        "http://app.flymy.ai/projects/app%3Adenis%2Fpaint-arena",
        "https://app.flymy.ai/projects/app%3Adenis%2Fpaint-arena%2Fagent",
        "https://app.flymy.ai/projects/app%3Adenis%2F..%2Fx",
        "https://app.flymy.ai/projects/app%3Adenis%2Fpaint-arena?x=1",
        "https://app.flymy.ai/projects/app%253Adenis%252Fpaint-arena",
        "https://app.flymy.ai/projects/app%3Adenis%2F" + "a" * 170,
        "app:denis/paint-arena/agent",
        "site:01a0e5e1-d4b6-bbf5-0168-59a3c984c4fd",
        ARTIFACT_ID,
        "",
    ],
)
def test_a_link_or_id_that_is_not_exactly_an_app_or_fleet_never_sends(project):
    def handler(request):
        raise AssertionError("no request")

    client = client_with(handler)
    with pytest.raises(ValueError):
        client.projects.copy(project, idempotency_key="k")
    with pytest.raises(ValueError):
        client.projects.share(project, public=True)


def test_share_needs_a_bool_and_copy_a_valid_name():
    def handler(request):
        raise AssertionError("no request")

    client = client_with(handler)
    with pytest.raises(ValueError):
        client.projects.share(APP_ID, public="yes")
    with pytest.raises(ValueError):
        client.projects.copy(APP_ID, idempotency_key="k", name="Bad Name")


def test_the_async_client_shares_and_copies():
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers.get("Idempotency-Key")))
        if request.url.path.endswith("/copy/"):
            return response(COPIED)
        return response({**PROJECT, "public": True})

    async def run():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            assert isinstance(client.projects, AsyncProjects)
            await client.projects.share(APP_ID, public=True)
            copy = await client.projects.copy(LINK, idempotency_key="copy-a")
            assert copy.agents[0]["name"] == "lead"
        finally:
            await client._http.aclose()

    asyncio.run(run())
    assert seen == [
        ("/api/v1/agents/projects/app:denis/paint-arena/public/", None),
        ("/api/v1/agents/projects/app:denis/paint-arena/copy/", "copy-a"),
    ]


# A project changed after it is created (backend PROJECTS_API.md, "Change a project",
# flymy.project-edit.v1): a rename, the project budget and the deletion, each answered
# once per Idempotency-Key.
SITE_ID = "site:01a0e5e1-d4b6-bbf5-0168-59a3c984c4fd"
EDITED = {"schema": "flymy.project-edit.v1", "project": APP_ID}


def test_rename_set_budget_and_delete_send_their_routes_with_the_callers_key():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((
            request.method,
            request.url.path,
            request.headers.get("Idempotency-Key"),
            body,
        ))
        if request.method == "DELETE":
            return response({
                **EDITED,
                "deleted": "removed",
                "at": "2026-10-30T11:00:00.123456+00:00",
                "future_field": "kept",
            })
        if request.url.path.endswith("/budget/"):
            return response({
                **EDITED,
                "budget": {"active": True, "day": {"cap_usd": "7.000000"}},
                "release": None,
            })
        if request.url.path.endswith(f"/{SITE_ID}/"):
            return response({
                **EDITED,
                "project": SITE_ID,
                "title": body["title"] or "Page",
                "release": None,
            })
        release = {"number": 2, "status": "applying"}
        title = body["title"] or "paint-arena"
        return response({**EDITED, "title": title, "release": release}, status=202)

    client = client_with(handler)
    renamed = client.projects.rename(
        APP_ID, title="Paint Arena", idempotency_key="rename-1"
    )
    dropped = client.projects.rename(SITE_ID, title=None, idempotency_key="rename-2")
    budget = client.projects.set_budget(
        APP_ID, {"per_day_usd": "7", "per_week_usd": None}, idempotency_key="budget-1"
    )
    deleted = client.projects.delete(APP_ID, idempotency_key="delete-1")

    root = "/api/v1/agents/projects/"
    assert seen == [
        (
            "PATCH",
            f"{root}app:denis/paint-arena/",
            "rename-1",
            {"title": "Paint Arena"},
        ),
        ("PATCH", f"{root}{SITE_ID}/", "rename-2", {"title": None}),
        (
            "PATCH",
            f"{root}app:denis/paint-arena/budget/",
            "budget-1",
            {"per_day_usd": "7", "per_week_usd": None},
        ),
        ("DELETE", f"{root}app:denis/paint-arena/", "delete-1", None),
    ]
    assert isinstance(renamed, ProjectEdit) and renamed.title == "Paint Arena"
    assert renamed.release == {"number": 2, "status": "applying"}
    assert dropped.project == SITE_ID and dropped.title == "Page"
    assert dropped.release is None
    assert budget.budget["day"] == {"cap_usd": "7.000000"} and budget.release is None
    assert deleted.deleted == "removed" and deleted.release is None
    assert deleted.at == "2026-10-30T11:00:00.123456+00:00"
    # a field added within the version reaches the caller
    assert deleted.model_extra["future_field"] == "kept"


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda c: c.projects.rename(FLEET_ID, title="x", idempotency_key="k"),
            (
                "a fleet without a named AgentTeam uses its lead's name: rename the"
                " lead with agents.update; a named AgentTeam has no rename endpoint"
            ),
        ),
        (
            lambda c: c.projects.rename(ARTIFACT_ID, title="x", idempotency_key="k"),
            "an artifact is renamed with client.artifacts.update",
        ),
        (
            lambda c: c.projects.rename(
                "app:denis/../x", title="x", idempotency_key="k"
            ),
            (
                "project_id must be an id from list(): app:<owner>/<name> or"
                " site:<page id>"
            ),
        ),
        (
            lambda c: c.projects.rename(APP_ID, title=42, idempotency_key="k"),
            "title must be a string of 1 to 80 characters, or None to drop it",
        ),
        (
            lambda c: c.projects.set_budget(
                SITE_ID, {"per_day_usd": "5"}, idempotency_key="k"
            ),
            "a page on its own has no project budget",
        ),
        (
            lambda c: c.projects.set_budget(
                FLEET_ID, {"per_day_usd": "5"}, idempotency_key="k"
            ),
            "its lead starts its agents within a Team's run cap",
        ),
        (
            lambda c: c.projects.set_budget(
                ARTIFACT_ID, {"per_day_usd": "5"}, idempotency_key="k"
            ),
            "an artifact has no project budget",
        ),
        (
            lambda c: c.projects.set_budget(APP_ID, {}, idempotency_key="k"),
            "budget is {per_day_usd, per_week_usd, per_month_usd}",
        ),
        (
            lambda c: c.projects.set_budget(
                APP_ID, {"per_year_usd": "5"}, idempotency_key="k"
            ),
            "budget is {per_day_usd, per_week_usd, per_month_usd}",
        ),
        (
            lambda c: c.projects.set_budget(APP_ID, "5", idempotency_key="k"),
            "budget is {per_day_usd, per_week_usd, per_month_usd}",
        ),
        (
            lambda c: c.projects.delete(SITE_ID, idempotency_key="k"),
            "a page on its own is not deleted here: stop it with stop()",
        ),
        (
            lambda c: c.projects.delete(FLEET_ID, idempotency_key="k"),
            "delete its agents with agents.delete",
        ),
        (
            lambda c: c.projects.delete(ARTIFACT_ID, idempotency_key="k"),
            "an artifact is deleted with client.artifacts.delete",
        ),
        (
            lambda c: c.projects.delete(
                "app:denis/paint-arena/agent", idempotency_key="k"
            ),
            "project_id must be an app id from list(): app:<owner>/<name>",
        ),
    ],
)
def test_a_change_the_project_does_not_take_is_refused_before_any_request(
    call, message
):
    def handler(request):
        raise AssertionError(f"unexpected request {request.url}")

    with pytest.raises(ValueError, match=re.escape(message)):
        call(client_with(handler))


def test_a_refused_change_comes_back_as_the_api_error_with_its_code():
    def handler(request):
        return response(
            {
                "code": "project_stopped",
                "message": "denis/paint-arena is stopped: start it first",
            },
            status=409,
        )

    with pytest.raises(FlyMyAIAgentError) as caught:
        client_with(handler).projects.set_budget(
            APP_ID, {"per_day_usd": "5"}, idempotency_key="budget-1"
        )
    assert caught.value.status_code == 409
    assert caught.value.response_body["code"] == "project_stopped"


def test_the_async_client_renames_sets_the_budget_and_deletes():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        if request.method == "DELETE":
            return response({**EDITED, "deleted": "archived", "at": "2026-10-30"})
        return response({**EDITED, "release": None})

    async def go():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            await client.projects.rename(APP_ID, title="Arena", idempotency_key="r-1")
            await client.projects.set_budget(
                APP_ID, {"per_month_usd": "30"}, idempotency_key="b-1"
            )
            return await client.projects.delete(APP_ID, idempotency_key="d-1")
        finally:
            await client.close()

    deleted = asyncio.run(go())
    root = "/api/v1/agents/projects/app:denis/paint-arena/"
    assert seen == [
        ("PATCH", root, {"title": "Arena"}),
        ("PATCH", f"{root}budget/", {"per_month_usd": "30"}),
        ("DELETE", root, None),
    ]
    assert isinstance(deleted, ProjectEdit) and deleted.deleted == "archived"
