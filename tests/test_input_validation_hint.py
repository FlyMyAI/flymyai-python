import asyncio
import json
from pathlib import Path

import httpx
import pytest

from flymyai import VariablesValidationError
from flymyai.agents._client import AsyncAgentClient, SyncAgentClient


BODY = json.loads(
    (Path(__file__).parent / "fixtures" / "frozen-input-validation.json").read_text()
)


@pytest.mark.parametrize("frozen", [True, False])
def test_sync_keeps_first_error_and_legacy_messages_without_retry(frozen):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, json=BODY)

    with SyncAgentClient(api_key="fixture", base_url="http://fixture.test") as client:
        client._http.close()
        client._http = httpx.Client(
            base_url="http://fixture.test", transport=httpx.MockTransport(handler)
        )
        call = client.compilations.run_instruction if frozen else client.agents.run
        with pytest.raises(VariablesValidationError) as error:
            call(
                134 if frozen else "fixture-agent",
                variables={"records_json": "x" * 8193},
                idempotency_key="fixture-invalid",
            )
        assert error.value.messages == BODY["variables"]
        assert error.value.response_body["input_validation"] == BODY["input_validation"]
        assert error.value.status_code == 400
        assert len(calls) == 1
        assert calls[0].method == "POST"


@pytest.mark.parametrize("frozen", [True, False])
def test_async_keeps_first_error_and_legacy_messages_without_retry(frozen):
    async def check():
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(400, json=BODY)

        async with AsyncAgentClient(
            api_key="fixture", base_url="http://fixture.test"
        ) as client:
            await client._http.aclose()
            client._http = httpx.AsyncClient(
                base_url="http://fixture.test", transport=httpx.MockTransport(handler)
            )
            call = client.compilations.run_instruction if frozen else client.agents.run
            with pytest.raises(VariablesValidationError) as error:
                await call(
                    134 if frozen else "fixture-agent",
                    variables={"records_json": "x" * 8193},
                    idempotency_key="fixture-invalid",
                )
            assert error.value.messages == BODY["variables"]
            assert (
                error.value.response_body["input_validation"]
                == BODY["input_validation"]
            )
            assert error.value.status_code == 400
            assert len(calls) == 1
            assert calls[0].method == "POST"

    asyncio.run(check())
