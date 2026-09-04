from unittest.mock import MagicMock

import httpx
import pytest

from flymyai.core._response import FlyMyAIResponse
from flymyai.core.clients.AsyncClient import BaseAsyncClient
from flymyai.core.clients.SyncClient import BaseSyncClient
from flymyai.core.exceptions import FlyMyAIPredictException
from flymyai.multipart import MultipartPayload


def _success_response() -> FlyMyAIResponse:
    return FlyMyAIResponse(
        200,
        json={"output_data": {"ok": True}, "status": 200},
        request=httpx.Request("POST", "https://api.flymy.ai/predict/"),
    )


def test_sync_predict_reuses_exact_key_for_internal_retry(monkeypatch):
    client = BaseSyncClient("test", "owner/model", max_retries=2)
    client._client.close()
    seen_keys = []

    def fake_predict(payload, client_info, *, idempotency_key):
        seen_keys.append(idempotency_key)
        if len(seen_keys) == 1:
            raise FlyMyAIPredictException("retry", requires_retry=True)
        return _success_response()

    monkeypatch.setattr(client, "_predict", fake_predict)

    response = client.predict({}, idempotency_key="same-sync-operation")

    assert response.output_data == {"ok": True}
    assert seen_keys == ["same-sync-operation", "same-sync-operation"]


@pytest.mark.asyncio
async def test_async_predict_reuses_exact_key_for_internal_retry(monkeypatch):
    client = BaseAsyncClient("test", "owner/model", max_retries=2)
    await client._client.aclose()
    seen_keys = []

    async def fake_predict(client_info, payload, *, idempotency_key):
        seen_keys.append(idempotency_key)
        if len(seen_keys) == 1:
            raise FlyMyAIPredictException("retry", requires_retry=True)
        return _success_response()

    monkeypatch.setattr(client, "_predict", fake_predict)

    response = await client.predict({}, idempotency_key="same-async-operation")

    assert response.output_data == {"ok": True}
    assert seen_keys == ["same-async-operation", "same-async-operation"]


@pytest.mark.parametrize("key", ["", " leading", "trailing ", "bad\nkey", "é"])
def test_predict_rejects_invalid_keys_before_dispatch(key):
    client = BaseSyncClient("test", "owner/model")
    client._client.close()
    client._client = MagicMock()

    with pytest.raises(ValueError, match="idempotency_key"):
        client.predict({}, idempotency_key=key)

    client._client.stream.assert_not_called()


def test_sync_stream_forwards_exact_key():
    client = BaseSyncClient("test", "owner/model")
    client._client.close()
    client._client = MagicMock()

    client._stream_iterator(
        client.amend_client_info(),
        MultipartPayload({}),
        is_long_stream=True,
        idempotency_key="same-stream-operation",
    )

    assert client._client.stream.call_args.kwargs["headers"]["Idempotency-Key"] == (
        "same-stream-operation"
    )


def test_keyless_public_effect_calls_fail_locally():
    client = BaseSyncClient("test", "owner/model")
    client._client.close()

    with pytest.raises(TypeError, match="idempotency_key"):
        client.predict({})
    with pytest.raises(TypeError, match="idempotency_key"):
        client.stream({})
    with pytest.raises(TypeError, match="idempotency_key"):
        BaseSyncClient.run_predict("test", "owner/model", {})
