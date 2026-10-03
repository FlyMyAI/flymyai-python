import asyncio
import base64
import json
from uuid import uuid4

import httpx
import pytest

from flymyai import ArtifactStaleBaseVersionError
from flymyai.agents import (
    ArtifactDryRun,
    ArtifactReceipt,
    Artifacts,
    AsyncArtifacts,
    artifact_file,
    artifact_files_from_directory,
)
from flymyai.agents._client import AsyncAgentClient
from tests.test_mcp_teams import client_with, response

ARTIFACT = str(uuid4())
NOW = "2026-10-02T23:16:44Z"
SUMMARY = {
    "id": ARTIFACT,
    "name": "Pod racer",
    "role": "owner",
    "live_version": 1,
    "latest_version": 1,
    "visibility": "private",
    "with_sources": False,
    "clone_count": 0,
    "forked_from": None,
    "owner_url": f"https://app.test/artifacts/{ARTIFACT}",
    "share_url": None,
    "created_at": NOW,
    "updated_at": NOW,
}
PLAN = {
    "version": 1,
    "entry": "index.html",
    "files_count": 2,
    "total_bytes": 20,
    "new_bytes": 20,
    "added": ["index.html", "car.png"],
    "changed": [],
    "removed": [],
    "no_change": False,
}


def test_create_sends_the_files_and_the_callers_key():
    files = [
        artifact_file("index.html", "<p>hi</p>"),
        artifact_file("car.png", b"\x89PNG"),
    ]

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/artifacts/"
        assert request.headers["Idempotency-Key"] == "create-pod-racer-1"
        assert json.loads(request.content) == {
            "name": "Pod racer",
            "files": [
                {"path": "index.html", "content": "<p>hi</p>"},
                {
                    "path": "car.png",
                    "content_base64": base64.b64encode(b"\x89PNG").decode(),
                },
            ],
            "visibility": "link",
        }
        return response(
            {"schema": "flymy.artifact.v1", "artifact": SUMMARY, "version": PLAN}, 201
        )

    with client_with(handler) as client:
        made = client.artifacts.create(
            name="Pod racer",
            files=files,
            visibility="link",
            idempotency_key="create-pod-racer-1",
        )
    assert isinstance(made, ArtifactReceipt)
    assert str(made.artifact.id) == ARTIFACT and made.version.added == [
        "index.html",
        "car.png",
    ]


def test_a_dry_run_answers_the_plan():
    def handler(request):
        assert json.loads(request.content)["dry_run"] is True
        return response({"schema": "flymy.artifact.v1", "dry_run": True, "plan": PLAN})

    with client_with(handler) as client:
        plan = client.artifacts.create(
            name="Pod racer",
            files=[artifact_file("index.html", "<p>hi</p>")],
            idempotency_key="plan-1",
            dry_run=True,
        )
    assert isinstance(plan, ArtifactDryRun) and plan.plan.files_count == 2


def test_lists_whats_shared_with_me_and_keeps_fields_added_later_in_v1():
    def handler(request):
        assert request.url.path == "/api/v1/artifacts/"
        assert dict(request.url.params) == {"scope": "shared", "page_size": "10"}
        row = {**SUMMARY, "role": "editor", "a_field_added_later": 1}
        return response(
            {"results": [row], "next_cursor": None, "previous_cursor": None}
        )

    with client_with(handler) as client:
        page = client.artifacts.list(scope="shared", page_size=10)
    assert page.results[0].role == "editor"
    assert page.results[0].model_extra == {"a_field_added_later": 1}
    with pytest.raises(ValueError):
        client.artifacts.list(scope="everything")


def test_someones_artifact_by_its_share_link_and_a_clone_of_it():
    seen = []

    def handler(request):
        seen.append(
            (request.method, request.url.path, request.headers.get("Idempotency-Key"))
        )
        if request.method == "GET":
            live = {
                "version": 2,
                "entry": "index.html",
                "message": "",
                "files_count": 1,
                "total_bytes": 9,
                "live": True,
                "created_at": NOW,
                "view": {"url": "https://u.test/v/t/", "expires_at": NOW},
            }
            return response({
                **SUMMARY,
                "schema": "flymy.artifact.v1",
                "role": "viewer",
                "author": {"username": "ana"},
                "live": live,
                "versions": [live],
            })
        return response(
            {"schema": "flymy.artifact.v1", "artifact": SUMMARY, "version": PLAN}, 201
        )

    with client_with(handler) as client:
        shared = client.artifacts.get(
            share_link="https://app.test/artifacts/s/AbCdEfGh12345678"
        )
        client.artifacts.clone(
            share_link="AbCdEfGh12345678", version=2, idempotency_key="clone-1"
        )
        with pytest.raises(ValueError):
            client.artifacts.get(ARTIFACT, share_link="AbCdEfGh12345678")
    assert shared.author.username == "ana"
    assert (
        shared.live.view.url == "https://u.test/v/t/"
        and shared.versions[0].version == 2
    )
    assert seen == [
        ("GET", "/api/v1/artifacts/shared/AbCdEfGh12345678/", None),
        ("POST", "/api/v1/artifacts/shared/AbCdEfGh12345678/clone/", "clone-1"),
    ]


def test_a_stale_base_version_names_the_latest():
    def handler(request):
        assert json.loads(request.content)["base_version"] == 2
        return response(
            {
                "code": "stale_base_version",
                "detail": "The artifact is at version 3.",
                "details": {"latest_version": 3},
            },
            412,
        )

    with client_with(handler) as client:
        with pytest.raises(ArtifactStaleBaseVersionError) as raised:
            client.artifacts.publish(
                ARTIFACT,
                base_version=2,
                files=[artifact_file("a.js", "1")],
                idempotency_key="publish-1",
            )
    assert raised.value.latest_version == 3 and raised.value.status_code == 412


def test_people_by_name():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.raw_path.decode(), request.content))
        if request.method == "POST":
            return response(
                {
                    "schema": "flymy.artifact.v1",
                    "changed": True,
                    "member": {"username": "bo", "role": "edit", "added_at": NOW},
                },
                201,
            )
        return response({"id": ARTIFACT, "removed": "bo b"})

    with client_with(handler) as client:
        added = client.artifacts.add_member(ARTIFACT, "bo@example.test", role="edit")
        client.artifacts.remove_member(ARTIFACT, "bo b")
        with pytest.raises(ValueError):
            client.artifacts.add_member(ARTIFACT, "bo", role="admin")
    assert added.member.role == "edit" and added.changed
    assert seen[0][:2] == ("POST", f"/api/v1/artifacts/{ARTIFACT}/members/")
    assert json.loads(seen[0][2]) == {"user": "bo@example.test", "role": "edit"}
    assert seen[1][:2] == ("DELETE", f"/api/v1/artifacts/{ARTIFACT}/members/bo%20b/")


def test_files_from_a_directory_keep_text_as_text_and_skip_hidden(tmp_path):
    (tmp_path / "js").mkdir()
    (tmp_path / "index.html").write_text("<p>hi</p>", encoding="utf-8")
    (tmp_path / "js" / "app.js").write_text("go()", encoding="utf-8")
    (tmp_path / "car.png").write_bytes(b"\x89PNG\x00")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    # an app template's config is text too
    (tmp_path / "flymy.yaml").write_text("kind: App\n", encoding="utf-8")
    files = artifact_files_from_directory(tmp_path)
    assert files == [
        {
            "path": "car.png",
            "content_base64": base64.b64encode(b"\x89PNG\x00").decode(),
        },
        {"path": "flymy.yaml", "content": "kind: App\n"},
        {"path": "index.html", "content": "<p>hi</p>"},
        {"path": "js/app.js", "content": "go()"},
    ]


def test_the_async_client_has_the_same_artifacts():
    def handler(request):
        if request.method == "GET":
            return response(
                {"results": [SUMMARY], "next_cursor": None, "previous_cursor": None}
            )
        return response(
            {"schema": "flymy.artifact.v1", "artifact": SUMMARY, "version": PLAN}, 201
        )

    async def go():
        client = AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        )
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        try:
            page = await client.artifacts.list()
            made = await client.artifacts.create(
                name="Pod racer",
                files=[artifact_file("index.html", "<p>hi</p>")],
                idempotency_key="create-1",
            )
        finally:
            await client.close()
        return page, made

    page, made = asyncio.run(go())
    assert page.results[0].name == "Pod racer" and made.version.version == 1


# The v1 methods stay (a frozen agent keeps the calls it was frozen with): one may be
# added, none renamed or removed.
V1_METHODS = {
    "status",
    "list",
    "get",
    "create",
    "publish",
    "versions",
    "files",
    "read_file",
    "view",
    "update",
    "share",
    "clone",
    "lineage",
    "history",
    "members",
    "add_member",
    "remove_member",
    "delete",
}


@pytest.mark.parametrize("resource", [Artifacts, AsyncArtifacts])
def test_the_v1_methods_stay(resource):
    methods = {name for name in dir(resource) if not name.startswith("_")}
    assert V1_METHODS <= methods
