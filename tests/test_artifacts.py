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


MEMBER = str(uuid4())
PENDING = {
    "id": MEMBER,
    "role": "edit",
    "status": "pending",
    "added_at": NOW,
    "accepted_at": None,
    "you": False,
    "invited_as": "bo@example.test",
}


def test_people_by_name_are_invited_and_removed_by_member_id():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.raw_path.decode(), request.content))
        if request.method == "POST":
            return response(
                {"schema": "flymy.artifact.v1", "changed": True, "member": PENDING}, 201
            )
        return response({"id": ARTIFACT, "removed": MEMBER})

    with client_with(handler) as client:
        added = client.artifacts.add_member(ARTIFACT, "bo@example.test", role="edit")
        client.artifacts.remove_member(ARTIFACT, added.member.id)
        for bad in (
            lambda: client.artifacts.add_member(ARTIFACT, "bo", role="admin"),
            # never a username or an email in a path
            lambda: client.artifacts.remove_member(ARTIFACT, "bo b"),
            lambda: client.artifacts.remove_member(ARTIFACT, "bo@example.test"),
        ):
            with pytest.raises(ValueError):
                bad()
    member = added.member
    assert member.status == "pending" and member.accepted_at is None
    assert member.invited_as == "bo@example.test" and not member.you
    assert added.changed
    assert seen[0][:2] == ("POST", f"/api/v1/artifacts/{ARTIFACT}/members/")
    assert json.loads(seen[0][2]) == {"user": "bo@example.test", "role": "edit"}
    assert seen[1][:2] == ("DELETE", f"/api/v1/artifacts/{ARTIFACT}/members/{MEMBER}/")
    assert len(seen) == 2


def test_the_invited_person_lists_accepts_and_declines():
    invitation = {
        "id": MEMBER,
        "artifact": {"id": ARTIFACT, "name": "Pod racer"},
        "owner": {"username": "ana"},
        "role": "edit",
        "invited_as": "bo@example.test",
        "status": "pending",
        "invited_at": NOW,
    }
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/accept/"):
            return response({
                "schema": "flymy.artifact.v1",
                "invitation": {**invitation, "status": "accepted"},
                "artifact": {**SUMMARY, "role": "editor"},
            })
        if request.url.path.endswith("/decline/"):
            return response(
                {"schema": "flymy.artifact.v1", "id": MEMBER, "declined": True}
            )
        return response(
            {"results": [invitation], "next_cursor": None, "previous_cursor": None}
        )

    with client_with(handler) as client:
        waiting = client.artifacts.invitations(page_size=5)
        accepted = client.artifacts.accept_invitation(waiting.results[0].id)
        declined = client.artifacts.decline_invitation(MEMBER)
        with pytest.raises(ValueError):
            client.artifacts.accept_invitation("../x")
    first = waiting.results[0]
    assert first.artifact.name == "Pod racer" and first.owner.username == "ana"
    assert accepted.artifact.role == "editor" and declined.declined
    assert seen == [
        ("GET", "/api/v1/artifacts/invitations/", {"page_size": "5"}),
        ("POST", f"/api/v1/artifacts/invitations/{MEMBER}/accept/", {}),
        ("POST", f"/api/v1/artifacts/invitations/{MEMBER}/decline/", {}),
    ]


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
    # S2 (2026-10-03, before v1 reached production): sharing by name is an invitation
    # the person accepts, and remove_member takes the member's id
    "invitations",
    "accept_invitation",
    "decline_invitation",
}


@pytest.mark.parametrize("resource", [Artifacts, AsyncArtifacts])
def test_the_v1_methods_stay(resource):
    methods = {name for name in dir(resource) if not name.startswith("_")}
    assert V1_METHODS <= methods


@pytest.mark.parametrize("asynchronous", [False, True])
def test_optional_runtime_binding_inherit_and_explicit_detach(asynchronous):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from flymyai.agents._client import SyncAgentClient

    runtime = {
        "schema": "flymy.artifact-runtime.v1",
        "source": {"site": ARTIFACT, "version": 1},
        "calls": {"ask": "decide"},
    }
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            assert self.path.startswith("/api/v1/artifacts/")
            payload = json.dumps(
                {"schema": "flymy.artifact.v1", "artifact": SUMMARY, "version": PLAN}
            ).encode()
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        if asynchronous:

            async def go():
                client = AsyncAgentClient(api_key="fixture-key", base_url=address)
                try:
                    await client.artifacts.create(
                        name="Bound",
                        files=[],
                        runtime=runtime,
                        idempotency_key="create",
                    )
                    await client.artifacts.publish(
                        ARTIFACT, base_version=1, idempotency_key="inherit"
                    )
                    await client.artifacts.publish(
                        ARTIFACT, base_version=2, runtime=None, idempotency_key="detach"
                    )
                finally:
                    await client._http.aclose()

            asyncio.run(go())
        else:
            with SyncAgentClient(api_key="fixture-key", base_url=address) as client:
                client.artifacts.create(
                    name="Bound", files=[], runtime=runtime, idempotency_key="create"
                )
                client.artifacts.publish(
                    ARTIFACT, base_version=1, idempotency_key="inherit"
                )
                client.artifacts.publish(
                    ARTIFACT, base_version=2, runtime=None, idempotency_key="detach"
                )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
    assert not worker.is_alive()
    assert seen[0]["runtime"] == runtime
    assert "runtime" not in seen[1]
    assert seen[2]["runtime"] is None
