import asyncio
import inspect
import json
from uuid import UUID

import httpx
import pytest

from flymyai.agents import AsyncAgentClient, FlyMyAIAgentError
from tests.test_mcp_teams import client_with, response

APP = "author/maze"
PROJECT = "app:" + APP
RELEASE = UUID("8252e015-521d-4545-98b0-9146e53c2682")
FILES = [{"path": "flymy.yaml", "content": "kind: App"}]
ANSWER = {
    "schema": "flymy.app-plan.v1",
    "plan_token": "reviewed",
    "future": {"kept": True},
    "project": PROJECT,
    "id": PROJECT, "kind": "app", "name": "maze",
}


@pytest.mark.parametrize("async_client", [False, True])
def test_shared_project_plan_and_create_preserve_exact_source_and_budget(async_client):
    seen = []

    def handler(request):
        seen.append((request.url.path, dict(request.headers), json.loads(request.content)))
        return response(ANSWER)

    options = {"template": "link:zUy4UnQ6jW90-QDW", "budget": {"per_day_usd": "1"}}
    assert run_call(async_client, handler, "projects", "plan", name="my-room", **options) == ANSWER
    assert run_call(async_client, handler, "projects", "create", name="my-room", plan_token="reviewed", idempotency_key="room-copy", **options) == ANSWER
    assert seen[0][0].endswith("/projects/")
    assert seen[0][2] == {"name": "my-room", **options}
    assert seen[1][2] == {**seen[0][2], "plan_token": "reviewed"}
    assert seen[1][1]["idempotency-key"] == "room-copy"


def run_call(async_client, handler, surface, method, *args, **kwargs):
    if not async_client:
        with client_with(handler) as client:
            result = getattr(getattr(client, surface), method)(*args, **kwargs)
            return result.model_dump(exclude_unset=True) if hasattr(result, "model_dump") else result

    async def run():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test",
            headers={"X-API-KEY": "personal-key"},
            transport=httpx.MockTransport(handler),
        )
        async with client:
            result = await getattr(getattr(client, surface), method)(*args, **kwargs)
            return result.model_dump(exclude_unset=True) if hasattr(result, "model_dump") else result

    return asyncio.run(run())


CASES = [
    ("apps", "list", (), {}, "GET", "/apps/", None),
    ("apps", "catalog", (), {}, "GET", "/apps/catalog/", None),
    ("apps", "plan", (), {"files": FILES}, "POST", "/apps/plan/", {"files": FILES}),
    (
        "apps",
        "plan",
        (),
        {"app": APP, "overrides": FILES},
        "POST",
        "/apps/plan/",
        {"app": APP, "overrides": FILES},
    ),
    (
        "apps",
        "apply",
        (),
        {
            "files": FILES,
            "instances": {"rooms": {}},
            "plan_token": "reviewed",
            "idempotency_key": "write-one",
        },
        "POST",
        "/apps/apply/",
        {"files": FILES, "instances": {"rooms": {}}, "plan_token": "reviewed"},
    ),
    ("apps", "release", (RELEASE,), {}, "GET", f"/apps/releases/{RELEASE}/", None),
    ("apps", "files", (APP,), {}, "GET", "/apps/files/", None),
    (
        "apps",
        "remove_instance",
        (APP, "rooms", "room-one"),
        {"idempotency_key": "write-one"},
        "POST",
        "/apps/remove/",
        {"app": APP, "module": "rooms", "key": "room-one"},
    ),
    ("projects", "list", (), {}, "GET", "/projects/", None),
    ("projects", "get", (PROJECT,), {}, "GET", f"/projects/{PROJECT}/", None),
    (
        "projects",
        "get",
        (f"site:{RELEASE}",),
        {},
        "GET",
        f"/projects/site:{RELEASE}/",
        None,
    ),
    (
        "projects",
        "get",
        (f"fleet:{RELEASE}",),
        {},
        "GET",
        f"/projects/fleet:{RELEASE}/",
        None,
    ),
    ("projects", "templates", (), {}, "GET", "/projects/templates/", None),
    (
        "projects",
        "errors",
        (PROJECT,),
        {"limit": 7, "before": "2026-10-03T10:00:00Z"},
        "GET",
        f"/projects/{PROJECT}/errors/",
        None,
    ),
    (
        "projects",
        "stop",
        (PROJECT,),
        {"idempotency_key": "write-one"},
        "POST",
        f"/projects/{PROJECT}/stop/",
        {},
    ),
    (
        "projects",
        "start_plan",
        (PROJECT,),
        {},
        "POST",
        f"/projects/{PROJECT}/start/",
        {},
    ),
    (
        "projects",
        "start",
        (PROJECT,),
        {"plan_token": "reviewed", "idempotency_key": "write-one"},
        "POST",
        f"/projects/{PROJECT}/start/",
        {"plan_token": "reviewed"},
    ),
]


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("surface,method,args,kwargs,verb,path,body", CASES)
def test_sync_async_versioned_control_contract(
    async_client, surface, method, args, kwargs, verb, path, body
):
    seen = []
    expected_answer = {**ANSWER, **({"release": str(RELEASE), "status": "applying"} if surface == "apps" and method == "apply" else {})}

    def handler(request):
        seen.append(request)
        assert request.headers["X-API-KEY"] == "personal-key"
        assert request.method == verb
        assert request.url.path == "/api/v1/agents" + path
        assert (json.loads(request.content) if request.content else None) == body
        assert request.headers.get("Idempotency-Key") == kwargs.get("idempotency_key")
        if method == "errors":
            assert dict(request.url.params) == {
                "limit": "7",
                "before": "2026-10-03T10:00:00Z",
            }
        return response(expected_answer)

    assert run_call(async_client, handler, surface, method, *args, **kwargs) == expected_answer
    assert len(seen) == 1


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize(
    "body", [{}, {"files": FILES, "app": APP}, {"files": FILES, "overrides": FILES}]
)
def test_ambiguous_source_is_rejected_before_network(async_client, body):
    def handler(request):
        pytest.fail("invalid source must not make an HTTP request")

    with pytest.raises(ValueError):
        run_call(async_client, handler, "apps", "plan", **body)


@pytest.mark.parametrize("async_client", [False, True])
@pytest.mark.parametrize("failure", ["stale", "unknown"])
def test_failed_apply_is_not_replanned_or_retried(async_client, failure):
    seen = []

    def handler(request):
        seen.append(request)
        if failure == "unknown":
            raise httpx.ReadTimeout("answer lost after dispatch", request=request)
        return response(
            {"code": "stale_plan", "plan": {"plan_token": "new-token"}}, 409
        )

    expected = httpx.ReadTimeout if failure == "unknown" else FlyMyAIAgentError
    with pytest.raises(expected):
        run_call(
            async_client,
            handler,
            "apps",
            "apply",
            app=APP,
            plan_token="reviewed",
            idempotency_key="write-one",
        )
    assert len(seen) == 1
    assert json.loads(seen[0].content)["plan_token"] == "reviewed"


def test_sync_async_signatures_match():
    from flymyai.agents._apps import Apps, AsyncApps
    from flymyai.agents._projects import Projects, AsyncProjects

    for sync, asynchronous in ((Apps, AsyncApps), (Projects, AsyncProjects)):
        for name in vars(sync):
            if not name.startswith("_"):
                assert inspect.signature(getattr(sync, name)) == inspect.signature(
                    getattr(asynchronous, name)
                )
