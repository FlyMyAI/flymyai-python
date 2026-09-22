"""Both SDK transports preserve the owner limits on ordinary and frozen runs."""

import asyncio
import json

import httpx
import pytest

from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

LIMITS = {"cap_usd": "0.500000", "max_children": 2, "max_parallel": 1}


def handle(request):
    assert json.loads(request.content)["subagent_limits"] == LIMITS
    assert request.headers["Idempotency-Key"] == "limited-run"
    return httpx.Response(
        201,
        json={
            "id": "abc-defg-hij",
            "user_agent_task": 1,
            "original_prompt": "Check independently",
            "status": "pending",
            "created_at": "2026-09-22T00:00:00Z",
            "updated_at": "2026-09-22T00:00:00Z",
            "subagent_limits": LIMITS,
        },
    )


@pytest.mark.parametrize("frozen", [False, True])
def test_sync_owner_limits(frozen):
    client = SyncAgentClient(api_key="test-key", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle)
    )
    action = client.compilations.run_instruction if frozen else client.agents.run
    try:
        result = action(
            1 if frozen else "agent-id",
            idempotency_key="limited-run",
            subagent_limits=LIMITS,
        )
        assert result.subagent_limits == LIMITS
    finally:
        client._http.close()


@pytest.mark.parametrize("frozen", [False, True])
def test_async_owner_limits(frozen):
    async def check():
        client = AsyncAgentClient(api_key="test-key", base_url="http://testserver")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="http://testserver", transport=httpx.MockTransport(handle)
        )
        action = client.compilations.run_instruction if frozen else client.agents.run
        try:
            result = await action(
                1 if frozen else "agent-id",
                idempotency_key="limited-run",
                subagent_limits=LIMITS,
            )
            assert result.subagent_limits == LIMITS
        finally:
            await client._http.aclose()

    asyncio.run(check())
