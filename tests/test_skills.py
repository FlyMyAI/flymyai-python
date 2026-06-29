"""Unit tests for the Agents SDK Skills surface.

All HTTP is mocked with ``httpx.MockTransport`` - nothing hits the wire. They
assert the client's behaviour: correct method/path, request body, and parsing,
mirroring tests/test_agents_variables.py. (The backend Skills endpoints are not
deployed yet, so these are intentionally transport-mocked, not live.)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

import httpx

from flymyai.agents._client import AsyncAgentClient, SyncAgentClient
from flymyai.agents._types import Skill


Route = Callable[[httpx.Request], httpx.Response]


def _handler(routes: Dict[Tuple[str, str], Route], seen: List[Tuple[str, str]]):
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        seen.append(key)
        route = routes.get(key)
        if route is None:
            return httpx.Response(404, json={"detail": f"No mock route for {key}"})
        return route(request)

    return handler


def _sync_client(routes: Dict[Tuple[str, str], Route], seen: List[Tuple[str, str]]):
    client = SyncAgentClient(api_key="test-key", base_url="http://testserver")
    client._http = httpx.Client(
        base_url="http://testserver",
        headers={"X-API-KEY": "test-key"},
        transport=httpx.MockTransport(_handler(routes, seen)),
    )
    return client


def _async_client(routes: Dict[Tuple[str, str], Route], seen: List[Tuple[str, str]]):
    client = AsyncAgentClient(api_key="test-key", base_url="http://testserver")
    client._http = httpx.AsyncClient(
        base_url="http://testserver",
        headers={"X-API-KEY": "test-key"},
        transport=httpx.MockTransport(_handler(routes, seen)),
    )
    return client


def _skill_json(skill_id: int, **over: Any) -> Dict[str, Any]:
    base = {
        "id": skill_id,
        "name": f"Skill {skill_id}",
        "slug": f"skill-{skill_id}",
        "description": "How to do X.",
        "is_favorite": False,
        "is_active": True,
        "is_public": False,
    }
    base.update(over)
    return base


def _agent_json(uuid: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "uuid": uuid,
        "name": "Test agent",
        "user_prompt": "Do something",
        "created_at": now,
        "updated_at": now,
    }


def test_skills_list() -> None:
    seen: List[Tuple[str, str]] = []
    routes = {
        ("GET", "/api/v1/agents/skills/"): lambda r: httpx.Response(
            200, json=[_skill_json(1), _skill_json(2)]
        )
    }
    skills = _sync_client(routes, seen).skills.list()
    assert ("GET", "/api/v1/agents/skills/") in seen
    assert [s.slug for s in skills] == ["skill-1", "skill-2"]
    assert all(isinstance(s, Skill) for s in skills)


def test_skills_get_includes_body() -> None:
    seen: List[Tuple[str, str]] = []
    routes = {
        ("GET", "/api/v1/agents/skills/7/"): lambda r: httpx.Response(
            200, json=_skill_json(7, instructions_md="# Body\n\nstep 1")
        )
    }
    skill = _sync_client(routes, seen).skills.get(7)
    assert skill.id == 7
    assert skill.instructions_md == "# Body\n\nstep 1"


def test_skills_attach_sends_skill_ids() -> None:
    seen: List[Tuple[str, str]] = []
    captured: Dict[str, Any] = {}

    def attach(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=[_skill_json(1), _skill_json(2)])

    routes = {("POST", "/api/v1/agents/tasks/abc/skills/attach/"): attach}
    result = _sync_client(routes, seen).skills.attach("abc", [1, 2])
    assert captured["body"] == {"skill_ids": [1, 2]}
    assert [s.id for s in result] == [1, 2]


def test_skills_detach_sends_skill_ids() -> None:
    seen: List[Tuple[str, str]] = []
    captured: Dict[str, Any] = {}

    def detach(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=[_skill_json(2)])

    routes = {("POST", "/api/v1/agents/tasks/abc/skills/detach/"): detach}
    result = _sync_client(routes, seen).skills.detach("abc", [1])
    assert captured["body"] == {"skill_ids": [1]}
    assert [s.id for s in result] == [2]


def test_agent_create_with_skills_attaches_after_create() -> None:
    seen: List[Tuple[str, str]] = []
    captured: Dict[str, Any] = {}

    def create(request: httpx.Request) -> httpx.Response:
        captured["create_body"] = json.loads(request.content)
        return httpx.Response(201, json=_agent_json("new-uuid"))

    def attach(request: httpx.Request) -> httpx.Response:
        captured["attach_body"] = json.loads(request.content)
        return httpx.Response(200, json=[_skill_json(5)])

    routes = {
        ("POST", "/api/v1/agents/tasks/"): create,
        ("POST", "/api/v1/agents/tasks/new-uuid/skills/attach/"): attach,
    }
    agent = _sync_client(routes, seen).agents.create(name="A", goal="g", skills=[5])
    assert agent.uuid == "new-uuid"
    # skills are NOT put in the create body (no write-serializer bypass) ...
    assert "available_skills" not in captured["create_body"]
    # ... they go through the validated attach endpoint after create.
    assert captured["attach_body"] == {"skill_ids": [5]}
    assert ("POST", "/api/v1/agents/tasks/new-uuid/skills/attach/") in seen


def test_async_skills_attach() -> None:
    # Driven via asyncio.run so the test needs no pytest-asyncio plugin.
    seen: List[Tuple[str, str]] = []
    captured: Dict[str, Any] = {}

    def attach(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=[_skill_json(3)])

    routes = {("POST", "/api/v1/agents/tasks/zzz/skills/attach/"): attach}

    async def _run() -> List[Skill]:
        client = _async_client(routes, seen)
        return await client.skills.attach("zzz", [3])

    result = asyncio.run(_run())
    assert captured["body"] == {"skill_ids": [3]}
    assert [s.id for s in result] == [3]
