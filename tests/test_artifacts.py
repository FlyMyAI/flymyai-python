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
    # an answer without the live version (a backend before it was added)
    assert raised.value.live_version is None


def test_a_stale_base_version_names_the_live_version_too():
    def handler(request):
        return response(
            {
                "code": "stale_base_version",
                "detail": "The artifact is at version 4, live version 2.",
                "details": {"latest_version": 4, "live_version": 2},
            },
            412,
        )

    with client_with(handler) as client:
        with pytest.raises(ArtifactStaleBaseVersionError) as raised:
            client.artifacts.publish(
                ARTIFACT, base_version=1, publish=False, idempotency_key="publish-2"
            )
    assert (raised.value.latest_version, raised.value.live_version) == (4, 2)


def test_a_version_next_to_the_live_one_keeps_it_and_has_its_page():
    page_url = f"https://app.test/artifacts/{ARTIFACT}?v=3"

    def handler(request):
        assert request.url.path == f"/api/v1/artifacts/{ARTIFACT}/versions/"
        # base_version may be the live version while a saved draft sits above it
        assert json.loads(request.content) == {
            "base_version": 1,
            "files": [{"path": "game.config.js", "content": "enemies = 2"}],
            "publish": False,
        }
        return response(
            {
                "schema": "flymy.artifact.v1",
                "artifact": {
                    **SUMMARY,
                    "type": "game",
                    "latest_version": 3,
                    "app_url": f"https://app.test/artifacts/{ARTIFACT}",
                },
                "version": {
                    **PLAN,
                    "version": 3,
                    "added": [],
                    "changed": ["game.config.js"],
                    "live": False,
                    "page_url": page_url,
                },
            },
            201,
        )

    with client_with(handler) as client:
        made = client.artifacts.publish(
            ARTIFACT,
            base_version=1,
            files=[artifact_file("game.config.js", "enemies = 2")],
            publish=False,
            idempotency_key="game-v3",
        )
    assert made.version.live is False and made.version.page_url == page_url
    assert made.artifact.type == "game" and made.artifact.live_version == 1
    assert made.artifact.app_url == f"https://app.test/artifacts/{ARTIFACT}"


def test_an_unchanged_version_names_the_version_that_holds_the_files():
    def handler(request):
        return response(
            {
                "schema": "flymy.artifact.v1",
                "artifact": SUMMARY,
                "version": {**PLAN, "version": 1, "no_change": True, "live": True},
            },
            201,
        )

    with client_with(handler) as client:
        made = client.artifacts.publish(
            ARTIFACT,
            base_version=1,
            files=[artifact_file("index.html", "<p>hi</p>")],
            idempotency_key="same-1",
        )
    assert made.version.no_change and made.version.version == 1
    assert made.version.page_url is None


def test_a_link_viewer_without_sources_gets_no_versions():
    def handler(request):
        assert request.url.path == "/api/v1/artifacts/shared/AbCdEfGh12345678/"
        return response({
            **SUMMARY,
            "schema": "flymy.artifact.v1",
            "role": "viewer",
            "visibility": "link",
            "owner_url": None,
            "author": {"username": "ana"},
            "live": {
                "version": 1,
                "entry": "index.html",
                "message": "",
                "files_count": 1,
                "total_bytes": 9,
                "live": True,
                "created_at": NOW,
                "view": {"url": "https://u.test/v/t/", "expires_at": NOW},
            },
            # the author shared only the page
            "versions": None,
            "next": "Its author shared only the page: it can be viewed.",
        })

    with client_with(handler) as client:
        shared = client.artifacts.get(
            share_link="https://app.test/artifacts/s/AbCdEfGh12345678"
        )
    assert shared.versions is None and shared.live.version == 1
    assert shared.next.startswith("Its author shared only the page")
    assert shared.type is None and shared.app_url is None


def test_links_are_a_share_link_its_handle_or_the_artifacts_own_page():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            return response(
                {"schema": "flymy.artifact.v1", "artifact": SUMMARY, "version": PLAN},
                201,
            )
        if request.url.path.endswith("/files/"):
            return response({"version": 2, "entry": "index.html", "files": []})
        return response({
            **SUMMARY,
            "schema": "flymy.artifact.v1",
            "author": {"username": "ana"},
            "versions": [],
        })

    page = f"https://app.test/artifacts/{ARTIFACT}"
    with client_with(handler) as client:
        # a share link with a query string, a slash and a fragment, or pasted with a
        # newline, and the bare handle
        for link in (
            "https://app.test/artifacts/s/AbCdEfGh12345678?utm_source=chat",
            "https://app.test/artifacts/s/AbCdEfGh12345678/#top",
            " https://app.test/artifacts/s/AbCdEfGh12345678\n",
            "AbCdEfGh12345678",
        ):
            client.artifacts.get(share_link=link)
        # the artifact's own page (owner_url, app_url, page_url) is its id, never a
        # share handle - in share_link and in artifact_id alike
        client.artifacts.get(share_link=page)
        client.artifacts.get(f"{page}?v=3")
        client.artifacts.files(
            share_link=f"https://app.test/artifacts/{ARTIFACT.upper()}", version=2
        )
        client.artifacts.publish(
            f"{page}/?v=2", base_version=2, idempotency_key="page-publish-1"
        )
        for bad in (
            # a handle-looking last segment of any other URL
            lambda: client.artifacts.get(
                share_link="https://app.test/x/AbCdEfGh12345678"
            ),
            lambda: client.artifacts.get(
                share_link="https://app.test/artifacts/s/AbCdEfGh12345678/versions"
            ),
            lambda: client.artifacts.get(share_link="https://app.test/artifacts/abc"),
            lambda: client.artifacts.get(share_link="too short"),
            # a share link is never an artifact id
            lambda: client.artifacts.get(
                "https://app.test/artifacts/s/AbCdEfGh12345678"
            ),
            lambda: client.artifacts.delete(
                "https://app.test/artifacts/s/AbCdEfGh1234"
            ),
        ):
            with pytest.raises(ValueError):
                bad()
    shared = ("GET", "/api/v1/artifacts/shared/AbCdEfGh12345678/")
    by_id = ("GET", f"/api/v1/artifacts/{ARTIFACT}/")
    assert seen == [
        shared,
        shared,
        shared,
        shared,
        by_id,
        by_id,
        ("GET", f"/api/v1/artifacts/{ARTIFACT}/versions/2/files/"),
        ("POST", f"/api/v1/artifacts/{ARTIFACT}/versions/"),
    ]


def test_idempotency_keys_take_what_the_backend_and_the_mcp_take():
    keys = []

    def handler(request):
        keys.append(request.headers["Idempotency-Key"])
        return response(
            {"schema": "flymy.artifact.v1", "artifact": SUMMARY, "version": PLAN}, 201
        )

    good = ["publish v2 of the racer", "x", "~" * 255, "a b-c_d.e:f/g"]
    with client_with(handler) as client:
        for key in good:
            client.artifacts.publish(ARTIFACT, base_version=1, idempotency_key=key)
        for key in ("", " lead", "trail ", "a" * 256, "tab\tkey", "new\nline", "ключ"):
            with pytest.raises(ValueError):
                client.artifacts.publish(ARTIFACT, base_version=1, idempotency_key=key)
        with pytest.raises(ValueError):
            client.artifacts.publish(ARTIFACT, base_version=1, idempotency_key=None)
    assert keys == good


TYPES = {
    "schema": "flymy.artifact.v1",
    "types": [
        {
            "key": "presentation",
            "name": "Presentation",
            "summary": "A slide deck written in Markdown.",
            "edit": "slides.md",
            "default_name": "Presentation",
            "example_change": "add a slide about pricing after the product slide",
            "entry": "index.html",
            "files_count": 6,
            "total_bytes": 48000,
        },
        {
            "key": "game",
            "name": "Game",
            "summary": "A 3D game in three.js.",
            "edit": "game.config.js",
            "default_name": "Golden Harvest",
            "example_change": "make v2 where the enemies are different",
            "entry": "index.html",
            "files_count": 9,
            "total_bytes": 900000,
            "a_field_added_later": True,
        },
    ],
}


def test_the_standard_types_and_an_artifact_made_from_one():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "GET":
            return response(TYPES)
        body = json.loads(request.content)
        made = {**SUMMARY, "name": body.get("name", "Golden Harvest"), "type": "game"}
        return response(
            {"schema": "flymy.artifact.v1", "artifact": made, "version": PLAN}, 201
        )

    with client_with(handler) as client:
        types = client.artifacts.types()
        game = client.artifacts.create(type="game", idempotency_key="game-1")
        deck = client.artifacts.create(
            type="presentation",
            name="Q3 review",
            files=[artifact_file("slides.md", "# Q3 review")],
            visibility="link",
            idempotency_key="deck-1",
        )
        # without a type, the name and the files are the caller's
        for missing in (
            {},
            {"name": "Page"},
            {"files": [artifact_file("index.html", "<p>hi</p>")]},
        ):
            with pytest.raises(ValueError):
                client.artifacts.create(idempotency_key="no-type-1", **missing)
    assert [item.key for item in types] == ["presentation", "game"]
    assert (
        types[1].edit == "game.config.js" and types[1].default_name == "Golden Harvest"
    )
    assert types[1].model_extra == {"a_field_added_later": True}
    assert game.artifact.type == "game" and game.artifact.name == "Golden Harvest"
    assert deck.artifact.name == "Q3 review"
    assert seen == [
        ("GET", "/api/v1/artifacts/types/"),
        ("POST", "/api/v1/artifacts/"),
        ("POST", "/api/v1/artifacts/"),
    ]


def test_a_create_with_a_type_sends_only_what_was_given():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response({"schema": "flymy.artifact.v1", "dry_run": True, "plan": PLAN})

    with client_with(handler) as client:
        client.artifacts.create(type="game", idempotency_key="g-1", dry_run=True)
        client.artifacts.create(
            type="report",
            files=[artifact_file("report.md", "# Costs")],
            idempotency_key="r-1",
        )
    assert bodies == [
        {"type": "game", "dry_run": True},
        {"type": "report", "files": [{"path": "report.md", "content": "# Costs"}]},
    ]


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
    bodies = []

    def handler(request):
        if request.url.path.endswith("/types/"):
            return response(TYPES)
        if request.method == "GET":
            return response(
                {"results": [SUMMARY], "next_cursor": None, "previous_cursor": None}
            )
        bodies.append(json.loads(request.content))
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
            types = await client.artifacts.types()
            await client.artifacts.create(type="landing", idempotency_key="landing-1")
            with pytest.raises(ValueError):
                await client.artifacts.create(name="Page", idempotency_key="page-1")
        finally:
            await client.close()
        return page, made, types

    page, made, types = asyncio.run(go())
    assert page.results[0].name == "Pod racer" and made.version.version == 1
    assert [item.key for item in types] == ["presentation", "game"]
    assert bodies[1] == {"type": "landing"}


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
    # 2026-10-09: the standard types a new artifact may start from (create(type=...))
    "types",
}


@pytest.mark.parametrize("resource", [Artifacts, AsyncArtifacts])
def test_the_v1_methods_stay(resource):
    methods = {name for name in dir(resource) if not name.startswith("_")}
    assert V1_METHODS <= methods
