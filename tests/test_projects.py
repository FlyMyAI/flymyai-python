import asyncio
import json

import httpx
import pytest

from flymyai.agents import (
    AsyncProjects,
    Project,
    ProjectCreated,
    ProjectPlan,
    Projects,
)
from flymyai.agents._client import AsyncAgentClient
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
