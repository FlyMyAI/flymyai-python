"""The first task reaches the backend in the same idempotent start request."""

import asyncio
import json

import httpx
import pytest

from flymyai.agents._client import AsyncAgentClient, SyncAgentClient


def handle(request):
    assert request.url.path == "/api/v1/agents/tasks/agent-id/run-loop/"
    assert request.headers["Idempotency-Key"] == "initial-task"
    assert json.loads(request.content) == {
        "message": "Собери отчёт сейчас.\nВерни таблицу.",
        "variables": {"account": "sample"},
    }
    return httpx.Response(
        201,
        json={
            "id": "abc-defg-hij",
            "user_agent_task": 1,
            "original_prompt": "Collect a report",
            "status": "running",
            "created_at": "2026-10-18T00:00:00Z",
            "updated_at": "2026-10-18T00:00:00Z",
        },
    )


@pytest.mark.parametrize("alias", [False, True])
def test_sync_first_message(alias):
    client = SyncAgentClient(api_key="test-key", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle)
    )
    try:
        action = client.runs.create if alias else client.agents.run
        result = action(
            agent_id="agent-id",
            idempotency_key="initial-task",
            message="Собери отчёт сейчас.\nВерни таблицу.",
            variables={"account": "sample"},
        )
        assert result.id == "abc-defg-hij"
    finally:
        client._http.close()


@pytest.mark.parametrize("alias", [False, True])
def test_async_first_message(alias):
    async def check():
        client = AsyncAgentClient(api_key="test-key", base_url="http://testserver")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="http://testserver", transport=httpx.MockTransport(handle)
        )
        try:
            action = client.runs.create if alias else client.agents.run
            result = await action(
                agent_id="agent-id",
                idempotency_key="initial-task",
                message="Собери отчёт сейчас.\nВерни таблицу.",
                variables={"account": "sample"},
            )
            assert result.id == "abc-defg-hij"
        finally:
            await client._http.aclose()

    asyncio.run(check())
