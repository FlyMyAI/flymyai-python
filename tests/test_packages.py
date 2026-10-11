import asyncio
import json

import httpx
import pytest

from flymyai.agents import (
    AsyncPackages,
    FleetInstaller,
    Package,
    PackageImport,
    PackagePlan,
    Packages,
)
from flymyai.agents._client import AsyncAgentClient, FlyMyAIAgentError
from tests.test_mcp_teams import client_with, response

AGENT = "7e3621c6-9f78-42b7-9942-3adb964ae17d"
TOKEN = "fpk_" + "a" * 43
PLAN_TOKEN = "0123456789abcdef" * 4
IMPORT = "pkgi_Ab12Cd34Ef56Gh78"
DOCUMENT = {
    "schema": "flymy.agent-package.v1",
    "name": "F2D",
    "agents": [{"key": "f2d", "name": "F2D", "system": "Test.", "tools": []}],
    "fleets": [],
    "requires": {"connections": []},
    "example_inputs": {},
    "acceptance": [],
}
PLAN = {
    "schema": "flymy.agent-package-plan.v1",
    "package": {"name": "F2D", "sha256": "ab" * 32, "source": "share", "agents": 1},
    "creates": [{"key": "f2d", "name": "F2D", "model": None, "tools": []}],
    "connections": [{
        "toolkit": "github",
        "status": "ambiguous",
        "candidates": [{"id": 1, "alias": "default"}, {"id": 2, "alias": "work"}],
    }],
    "acceptance": [],
    "warnings": [],
    "blocking": True,
    "plan_token": PLAN_TOKEN,
}
REPORT = {
    "schema": "flymy.agent-package-import.v1",
    "id": IMPORT,
    "status": "running",
    "agents": [{"key": "f2d", "uuid": AGENT, "name": "F2D"}],
    "cases": [{"name": "finds the site", "run_id": "akc-fwgb-zhb", "verdict": None}],
    "frozen": {},
    "error": None,
}
# v1: a frozen agent and a pinned client keep these names
V1_METHODS = {
    "export",
    "share",
    "shares",
    "revoke_share",
    "plan",
    "apply",
    "status",
    "imports",
    "installer",
}


def _public(cls):
    return {name for name in vars(cls) if not name.startswith("_")}


def test_v1_methods_are_pinned_on_both_clients():
    assert _public(Packages) == V1_METHODS
    assert _public(AsyncPackages) == V1_METHODS


def test_export_share_plan_apply_status_and_installer_reach_their_routes():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((
            request.method,
            request.url.path,
            request.headers.get("Idempotency-Key"),
            body,
        ))
        assert request.headers["X-API-KEY"] == "personal-key"
        path = request.url.path
        if path.endswith("/export/"):
            return response(DOCUMENT)
        if path.endswith("/shares/") and request.method == "POST":
            return response({"id": 3, "token": TOKEN, "name": "F2D"}, 201)
        if path.endswith("/plan/"):
            return response(PLAN)
        if path.endswith("/installer/"):
            return response(
                {"agent": AGENT, "name": "Fleet installer", "created": True}
            )
        if path.endswith("/imports/"):
            return response({"imports": [REPORT]})
        return response(REPORT, 201 if request.method == "POST" else 200)

    client = client_with(handler)
    package = client.packages.export([AGENT], name="F2D")
    share = client.packages.share(
        [AGENT], idempotency_key="share-1", recipient_email="bo@example.com"
    )
    plan = client.packages.plan(share_token=TOKEN)
    replanned = client.packages.plan(document=package, mappings={"github": 2})
    report = client.packages.apply(
        plan_token=PLAN_TOKEN,
        idempotency_key="apply-1",
        share_token=TOKEN,
        mappings={"github": 2},
    )
    status = client.packages.status(IMPORT)
    recent = client.packages.imports()
    installer = client.packages.installer()

    assert isinstance(package, Package) and package.document() == DOCUMENT
    assert share.token == TOKEN
    assert isinstance(plan, PackagePlan) and plan.blocking is True
    assert plan.connections[0]["candidates"][1]["alias"] == "work"
    assert replanned.plan_token == PLAN_TOKEN
    assert isinstance(report, PackageImport) and report.id == IMPORT
    assert status.cases[0]["run_id"] == "akc-fwgb-zhb"
    assert recent.imports[0].status == "running"
    assert isinstance(installer, FleetInstaller) and installer.created is True
    assert [(method, path, key) for method, path, key, _ in seen] == [
        ("POST", "/api/v1/agents/packages/export/", None),
        ("POST", "/api/v1/agents/packages/shares/", "share-1"),
        ("POST", "/api/v1/agents/packages/plan/", None),
        ("POST", "/api/v1/agents/packages/plan/", None),
        ("POST", "/api/v1/agents/packages/apply/", "apply-1"),
        ("GET", f"/api/v1/agents/packages/imports/{IMPORT}/", None),
        ("GET", "/api/v1/agents/packages/imports/", None),
        ("POST", "/api/v1/agents/packages/installer/", None),
    ]
    assert seen[0][3] == {"agents": [AGENT], "name": "F2D"}
    assert seen[1][3] == {"agents": [AGENT], "recipient_email": "bo@example.com"}
    assert seen[2][3] == {"source": {"share_token": TOKEN}}
    assert seen[3][3] == {"source": {"document": DOCUMENT}, "mappings": {"github": 2}}
    assert seen[4][3] == {
        "source": {"share_token": TOKEN},
        "plan_token": PLAN_TOKEN,
        "mappings": {"github": 2},
    }


def test_bad_calls_never_leave_and_a_stale_plan_carries_the_new_one():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return response(
            {"code": "stale_plan", "detail": "review the new plan", "data": PLAN}, 409
        )

    client = client_with(handler)
    with pytest.raises(ValueError, match="exactly one"):
        client.packages.plan(document=DOCUMENT, share_token=TOKEN)
    with pytest.raises(ValueError, match="exactly one"):
        client.packages.plan()
    with pytest.raises(ValueError, match="plan_token"):
        client.packages.apply(plan_token="", idempotency_key="k", share_token=TOKEN)
    with pytest.raises(ValueError):
        client.packages.apply(plan_token=PLAN_TOKEN, idempotency_key=" ", url="x")
    with pytest.raises(ValueError, match="pkgi_"):
        client.packages.status("../shares/")
    with pytest.raises(ValueError, match="list"):
        client.packages.export(AGENT)
    assert calls == []

    with pytest.raises(FlyMyAIAgentError) as stale:
        client.packages.apply(
            plan_token=PLAN_TOKEN, idempotency_key="apply-2", share_token=TOKEN
        )
    assert stale.value.status_code == 409
    assert stale.value.response_body["data"]["plan_token"] == PLAN_TOKEN


def test_the_async_client_imports_the_same_way():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/plan/"):
            return response(PLAN)
        return response(REPORT, 201)

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
        plan = await client.packages.plan(url="https://github.com/o/r/blob/main/p.yaml")
        report = await client.packages.apply(
            plan_token=plan.plan_token,
            idempotency_key="apply-3",
            url="https://github.com/o/r/blob/main/p.yaml",
        )
        await client._http.aclose()
        return report

    report = asyncio.run(run())
    assert report.status == "running"
    assert seen == [
        ("POST", "/api/v1/agents/packages/plan/"),
        ("POST", "/api/v1/agents/packages/apply/"),
    ]
