"""Instruction references cross both SDK transports without content rewriting."""

import asyncio
import json

import httpx
import pytest

from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

SOURCE = {"agent_file_id": 42}


def handle(request):
    assert json.loads(request.content) == {
        "name": "From file",
        "user_prompt_source": SOURCE,
    }
    return httpx.Response(
        201,
        json={
            "uuid": "helper",
            "name": "From file",
            "user_prompt": "Complete instructions",
            "user_prompt_source": SOURCE,
            "created_at": "2026-10-18T00:00:00Z",
            "updated_at": "2026-10-18T00:00:00Z",
        },
    )


def test_sync_instruction_source():
    client = SyncAgentClient(api_key="test", base_url="http://testserver")
    client._http.close()
    client._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle)
    )
    try:
        agent = client.agents.create(name="From file", user_prompt_source=SOURCE)
        assert agent.uuid == "helper"
        assert agent.user_prompt_source == SOURCE
        with pytest.raises(ValueError, match="exactly one"):
            client.agents.create(
                name="From file", goal="Conflicting literal", user_prompt_source=SOURCE
            )
    finally:
        client._http.close()


def test_async_instruction_source():
    async def check():
        client = AsyncAgentClient(api_key="test", base_url="http://testserver")
        await client._http.aclose()
        client._http = httpx.AsyncClient(
            base_url="http://testserver", transport=httpx.MockTransport(handle)
        )
        try:
            agent = await client.agents.create(
                name="From file", user_prompt_source=SOURCE
            )
            assert agent.uuid == "helper"
            assert agent.user_prompt_source == SOURCE
            with pytest.raises(ValueError, match="exactly one"):
                await client.agents.create(
                    name="From file",
                    goal="Conflicting literal",
                    user_prompt_source=SOURCE,
                )
        finally:
            await client._http.aclose()

    asyncio.run(check())
