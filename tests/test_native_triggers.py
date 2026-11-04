import asyncio
import json

import httpx

from flymyai.agents import AsyncAgentClient
from tests.test_mcp_teams import client_with, response
from tests.test_triggers import AGENT, ROOT, ROW, TRIGGER

CATALOG = {
    "schema": "flymy.trigger-catalog.v1",
    "providers": [{"provider": "granola", "events": ["note.generated"]}],
}
NATIVE = {
    **ROW,
    "kind": "mcp_event",
    "webhook_url": None,
    "source": {
        "id": TRIGGER,
        "provider": "granola",
        "connection_id": TRIGGER,
        "status": "pending",
        "detail": "",
    },
}


def handler(seen):
    def answer(request):
        seen.append(request)
        return response(CATALOG if request.method == "GET" else NATIVE)

    return answer


def check(seen):
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", ROOT + "catalog/"),
        ("POST", ROOT + "subscribe/"),
    ]
    assert seen[1].headers["Idempotency-Key"] == "native-one"
    assert json.loads(seen[1].content) == {
        "agent_id": AGENT,
        "connection_id": TRIGGER,
        "events": ["note.generated"],
        "scopes": ["personal"],
    }


def test_sync_native_subscription():
    seen = []
    with client_with(handler(seen)) as client:
        assert client.triggers_v1.catalog() == CATALOG
        result = client.triggers_v1.subscribe(
            AGENT,
            TRIGGER,
            events=["note.generated"],
            scopes=["personal"],
            idempotency_key="native-one",
        )
        assert result.source["status"] == "pending"
        assert result.webhook_url is None
    check(seen)


def test_async_native_subscription():
    seen = []

    async def run():
        async with AsyncAgentClient(
            api_key="personal-key", base_url="https://example.test"
        ) as client:
            await client._http.aclose()
            client._http = httpx.AsyncClient(
                base_url="https://example.test",
                headers={"X-API-KEY": "personal-key"},
                transport=httpx.MockTransport(handler(seen)),
            )
            assert await client.triggers_v1.catalog() == CATALOG
            result = await client.triggers_v1.subscribe(
                AGENT,
                TRIGGER,
                events=["note.generated"],
                scopes=["personal"],
                idempotency_key="native-one",
            )
            assert result.source["status"] == "pending"

    asyncio.run(run())
    check(seen)
