import asyncio
import json

import httpx
import pytest

from flymyai.agents import (
    AppFile,
    AppFiles,
    AppPlan,
    AppRelease,
    Apps,
    AsyncApps,
    artifact_file,
)
from flymyai.agents._client import AsyncAgentClient
from tests.test_mcp_teams import client_with, response

RELEASE = "3f2a9c1e-7b4d-4e8a-9c6f-1d2e3f4a5b6c"
FLYMY_YAML = """apiVersion: flymy.ai/v1alpha1
kind: App
metadata: {name: notes}
resources:
  files: {kind: Storage, type: files}
"""
FILES = [
    artifact_file("flymy.yaml", FLYMY_YAML),
    artifact_file("site/index.html", "<p>hi</p>"),
]
PLAN = {
    "app": "denis/notes",
    "changes": [{"resource": "files", "kind": "Storage", "action": "create"}],
    "usd_per_hour": "0.000000",
    "storages": {"files": {"type": "files", "scope": "app", "retention": None}},
    "revision": 0,
    "plan_token": "plan-1",
}
RELEASE_VIEW = {"release": RELEASE, "app": "denis/notes", "number": 1}


def test_a_plan_spends_nothing_and_the_apply_sends_the_same_files_and_the_key():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers.get("Idempotency-Key"), body))
        if request.url.path.endswith("/apply/"):
            return response({**RELEASE_VIEW, "status": "applying"}, 202)
        return response(PLAN)

    client = client_with(handler)
    plan = client.apps.plan(files=FILES)
    release = client.apps.apply(
        files=FILES, plan_token=plan.plan_token, idempotency_key="notes-apply-1"
    )

    assert isinstance(plan, AppPlan) and plan.changes[0]["kind"] == "Storage"
    # a field the backend adds is kept
    assert plan.storages["files"]["type"] == "files"
    assert isinstance(release, AppRelease) and release.status == "applying"
    assert seen == [
        ("/api/v1/agents/apps/plan/", None, {"files": FILES}),
        (
            "/api/v1/agents/apps/apply/",
            "notes-apply-1",
            {"files": FILES, "plan_token": "plan-1"},
        ),
    ]


def test_an_applied_app_is_changed_by_overrides_and_read_back():
    seen = []

    def handler(request):
        seen.append((
            request.method,
            request.url.path,
            dict(request.url.params),
            json.loads(request.content) if request.content else None,
        ))
        if request.url.path.endswith("/files/"):
            if "path" in request.url.params:
                return response(
                    {"path": "flymy.yaml", "content": FLYMY_YAML, "next_offset": None}
                )
            return response({"files": [{"path": "flymy.yaml", "text": True}]})
        if request.url.path.endswith("/plan/"):
            return response(PLAN)
        return response({**RELEASE_VIEW, "status": "succeeded"})

    client = client_with(handler)
    listing = client.apps.files("denis/notes")
    page = client.apps.files("denis/notes", path="flymy.yaml", offset=0, limit=1000)
    client.apps.plan(
        app="denis/notes", overrides=[artifact_file("site/index.html", "<p>v2</p>")]
    )
    done = client.apps.status(RELEASE)

    assert isinstance(listing, AppFiles) and listing.files[0]["path"] == "flymy.yaml"
    assert isinstance(page, AppFile) and page.content == FLYMY_YAML
    assert done.status == "succeeded" and done.number == 1
    assert seen == [
        ("GET", "/api/v1/agents/apps/files/", {"app": "denis/notes"}, None),
        (
            "GET",
            "/api/v1/agents/apps/files/",
            {
                "app": "denis/notes",
                "path": "flymy.yaml",
                "offset": "0",
                "limit": "1000",
            },
            None,
        ),
        (
            "POST",
            "/api/v1/agents/apps/plan/",
            {},
            {
                "app": "denis/notes",
                "overrides": [{"path": "site/index.html", "content": "<p>v2</p>"}],
            },
        ),
        ("GET", f"/api/v1/agents/apps/releases/{RELEASE}/", {}, None),
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.apps.plan(),
        lambda c: c.apps.plan(files=FILES, app="denis/notes"),
        lambda c: c.apps.plan(files=FILES, overrides=FILES),
        lambda c: c.apps.plan(app="denis"),
        lambda c: c.apps.plan(app="denis/../x/y"),
        lambda c: c.apps.apply(files=FILES, plan_token="", idempotency_key="k1"),
        # printable ASCII with spaces inside only, as the backend takes it
        lambda c: c.apps.apply(files=FILES, plan_token="p1", idempotency_key="a b "),
        lambda c: c.apps.files("denis/notes", offset=10),
        lambda c: c.apps.files("denis/notes", path="../secret"),
        lambda c: c.apps.files("denis/notes", path="/etc/passwd"),
        lambda c: c.apps.files("denis/notes", path="a\nb"),
        lambda c: c.apps.files("denis/notes", path="flymy.yaml", limit=0),
        lambda c: c.apps.status("not-a-release"),
    ],
)
def test_a_bad_source_or_argument_is_refused_before_any_request(call):
    def handler(request):
        raise AssertionError(f"unexpected request {request.url}")

    with pytest.raises(ValueError):
        call(client_with(handler))


def test_the_async_client_has_the_same_apps():
    def handler(request):
        if request.url.path.endswith("/plan/"):
            return response(PLAN)
        return response({**RELEASE_VIEW, "status": "applying"}, 202)

    async def go():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            plan = await client.apps.plan(files=FILES)
            release = await client.apps.apply(
                files=FILES, plan_token=plan.plan_token, idempotency_key="k1"
            )
        finally:
            await client.close()
        return plan, release

    plan, release = asyncio.run(go())
    assert plan.plan_token == "plan-1" and release.release == RELEASE
    sync = {name for name in dir(Apps) if not name.startswith("_")}
    assert sync == {name for name in dir(AsyncApps) if not name.startswith("_")}
