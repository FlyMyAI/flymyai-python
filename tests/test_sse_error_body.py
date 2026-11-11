"""An error body in SSE form, as the gateway answers on its SSE routes (2026-11-11:
a 422 "data: {...}" raised JSONDecodeError instead of the FlyMyAI error)."""

import httpx
import pytest

from flymyai.core._response import FlyMyAIResponse
from flymyai.core.clients.base_client import _status_error
from flymyai.core.exceptions import BaseFlyMyAIException

BODY = b'data: {"detail":[{"loc":["prompt"],"msg":"Field required","type":"missing"}],"status_code":422}\n\n'


def _response(status: int, content: bytes) -> httpx.Response:
    request = httpx.Request("POST", "https://api.flymy.ai/api/v1/u/p/predict/stream/")
    return httpx.Response(
        status,
        content=content,
        request=request,
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize(
    "content",
    [
        BODY,
        b"event: error\n" + BODY,
        b'data{"detail":"x","status_code":422}',
    ],
)
def test_sse_body_reads_as_json(content):
    response = FlyMyAIResponse(
        status_code=422, content=content, request=httpx.Request("GET", "https://x")
    )
    data = response.json()
    assert data["status_code"] == 422 or data.get("detail")


def test_a_422_in_sse_form_is_a_flymyai_error():
    stream = _response(422, BODY)
    error = _status_error(stream, BODY)
    assert isinstance(error, BaseFlyMyAIException)
    assert "Field required" in str(error) or "prompt" in str(error)
    assert not str(error).startswith("HTTP 422"), "read by the 422 class, as on main"


def test_an_unreadable_body_is_still_a_flymyai_error():
    stream = _response(502, b"<html>bad gateway</html>")
    error = _status_error(stream, b"<html>bad gateway</html>")
    assert isinstance(error, BaseFlyMyAIException)


def test_an_error_event_with_status_code_is_an_error():
    """predict on a streaming model: the gateway answers 200 with one error event."""
    from flymyai.core._streaming import ServerSentEvent
    from flymyai.core.response_factory.plain_inference_response_factory import (
        SSEInferenceResponseFactory,
    )

    request = httpx.Request("POST", "https://api.flymy.ai/api/v1/u/p/predict")
    stream = httpx.Response(
        200, request=request, headers={"content-type": "text/event-stream"}
    )
    sse = ServerSentEvent(
        data='{"detail":"Predict is not available for this model","status_code":421}'
    )
    with pytest.raises(BaseFlyMyAIException) as caught:
        SSEInferenceResponseFactory(
            sse=sse, httpx_request=request, httpx_response=stream
        ).construct()
    assert caught.value.response.status_code == 421


def test_a_result_event_stays_a_result():
    from flymyai.core._streaming import ServerSentEvent
    from flymyai.core.response_factory.plain_inference_response_factory import (
        SSEInferenceResponseFactory,
    )

    request = httpx.Request("POST", "https://api.flymy.ai/api/v1/u/p/predict")
    stream = httpx.Response(
        200, request=request, headers={"content-type": "text/event-stream"}
    )
    sse = ServerSentEvent(data='{"output_data":{"output":["hi"]},"status":200}')
    response = SSEInferenceResponseFactory(
        sse=sse, httpx_request=request, httpx_response=stream
    ).construct()
    assert response.status_code == 200
