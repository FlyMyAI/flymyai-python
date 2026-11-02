"""Native operations exercise the actual dedicated HTTP transport, without a network."""

import asyncio
import inspect
import json
import subprocess
import sys
import time
from contextlib import contextmanager

import httpx
import pytest
import respx

from flymyai.agents import AgentClient, AsyncAgentClient, ModelProxyError


ORIGIN = "https://sdk-native.invalid"
ROOT = ORIGIN + "/api/v1/model-proxy/"
REVISION = "a" * 64
OPERATION = "aaaaaaaa-1234-4321-abcd-123456789012"
KEY = "native-intent:original-key"
API_KEY = "synthetic-caller-key"


class Body(httpx.AsyncByteStream):
    def __init__(self, *chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self):
        self.closed = True


def response(value, *, status=200, headers=None):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return httpx.Response(status, stream=Body(raw), headers=headers)


def operation(**overrides):
    value = {
        "operation_id": OPERATION,
        "prediction_id": OPERATION,
        "contract_revision": REVISION,
        "status": "queued",
        "stage": "submit",
        "cancel_requested": False,
    }
    value.update(overrides)
    return value


@pytest.fixture(autouse=True)
def isolated_http():
    # Every HTTPX route must be declared. Never let a test call a live endpoint.
    # Cancelled header reads do not create a completed respx call receipt;
    # the timeout cases count actual handler entry separately.
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        yield router


def test_sync_catalog_uses_native_transport(isolated_http):
    route = isolated_http.get(ROOT + "catalog/").respond(
        stream=Body(b'{"release":null,"items":[],"next_cursor":null}')
    )
    with AgentClient(api_key=API_KEY, base_url=ORIGIN) as client:
        page = client.model_proxy.catalog(limit=1)
    assert page.items == [] and page.release is None
    assert route.call_count == 1
    assert route.calls[0].request.headers["X-API-KEY"] == API_KEY


@pytest.mark.asyncio
async def test_async_admission_retains_exact_identity(isolated_http):
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=response({"operation": operation(), "created": True}, status=202)
    )
    async with AsyncAgentClient(api_key=API_KEY, base_url=ORIGIN) as client:
        created = await client.model_proxy.create(
            REVISION, {"messages": [], "seed": 2**63 + 19}, idempotency_key=KEY
        )
    assert created.created is True
    assert created.operation.operation_id == OPERATION
    assert created.operation.contract_revision == REVISION
    assert route.call_count == 1
    request = route.calls[0].request
    assert request.headers["Idempotency-Key"] == KEY
    assert json.loads(request.content)["inputs"]["seed"] == 2**63 + 19


def test_closed_client_is_proven_pre_dispatch(isolated_http):
    client = AgentClient(api_key=API_KEY, base_url=ORIGIN)
    client.close()
    with pytest.raises(ModelProxyError) as raised:
        client.model_proxy.create(REVISION, {}, idempotency_key=KEY)
    assert raised.value.code == "client_closed"
    assert raised.value.outcome_unknown is False
    assert raised.value.idempotency_key == KEY
    assert len(isolated_http.calls) == 0


@pytest.mark.asyncio
async def test_sync_inside_event_loop_refuses_without_dispatch(isolated_http):
    with AgentClient(api_key=API_KEY, base_url=ORIGIN) as client:
        with pytest.raises(ModelProxyError) as raised:
            client.model_proxy.create(REVISION, {}, idempotency_key=KEY)
    assert raised.value.code == "use_async_client"
    assert raised.value.outcome_unknown is False
    assert len(isolated_http.calls) == 0


def test_claimed_is_a_valid_backend_state(isolated_http):
    isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        return_value=response({"operation": operation(status="claimed")})
    )
    with AgentClient(api_key=API_KEY, base_url=ORIGIN) as client:
        result = client.model_proxy.get(OPERATION, contract_revision=REVISION)
    assert result.status == "claimed"


@contextmanager
def native_client(mode, **kwargs):
    cls = AgentClient if mode == "sync" else AsyncAgentClient
    client = cls(api_key=API_KEY, base_url=ORIGIN, max_retries=9, **kwargs)

    def call(method, *args, **options):
        result = getattr(client.model_proxy, method)(*args, **options)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    try:
        yield client, call
    finally:
        if mode == "sync":
            client.close()
        else:
            asyncio.run(client.close())


def native_output():
    return {
        "id": "synthetic-completion",
        "object": "chat.completion",
        "model": "synthetic-native-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning": "пример 🐈",
                "tool_calls": [{
                    "id": "call-1",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"x":9007199254740993}',
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
        "native_extra": {"large": 2**100 + 17, "ordered": [None, False, 0, ""]},
    }


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_complete_native_flow_and_explicit_same_key_reconciliation(mode, isolated_http):
    summary = {
        "contract_revision": REVISION,
        "route_key": "fixture/route",
        "profile": "broker.ds-chat.v1",
        "model_version": "digest",
        "operation": "chat",
    }
    catalog = {"release": OPERATION, "items": [summary], "next_cursor": "opaque+cursor"}
    contract = {
        "contract_revision": REVISION,
        "artifact": {
            "schemas": {"input": {"type": "object"}},
            "native_extra": 2**100 + 1,
        },
        "execution_constraints": {"max_input_bytes": 262144},
    }
    isolated_http.get(ROOT + "catalog/").mock(return_value=response(catalog))
    isolated_http.get(ROOT + "contracts/" + REVISION + "/").mock(
        return_value=response(contract)
    )
    admission = isolated_http.post(ROOT + "operations/").mock(
        side_effect=[
            response({"operation": operation(), "created": True}, status=202),
            response({"operation": operation(), "created": False}),
        ]
    )
    read = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        side_effect=[
            response({"operation": operation(status="claimed")}),
            response(
                {"operation": operation(status="succeeded", output=native_output())}
            ),
        ]
    )
    inputs = {
        "messages": [{"role": "user", "content": "пример 🐈"}],
        "seed": 2**63 + 19,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "auto",
        "extra": [None, False, 0, ""],
    }
    with native_client(mode) as (client, call):
        # Any accidental use of the legacy retry/request path is a test failure.
        def legacy_forbidden(*args, **kwargs):
            raise AssertionError("native calls must not enter the legacy transport")

        client._request = legacy_forbidden
        client._http.cookies.set("ambient", "must-not-be-sent")
        client._http.headers["Authorization"] = "Bearer must-not-be-sent"
        page = call("catalog", limit=2, prefix="fixture/", cursor="earlier+cursor")
        assert page.model_dump() == catalog
        assert call("contract", REVISION).model_dump() == contract
        first = call("create", REVISION, inputs, idempotency_key=KEY)
        replay = call("create", REVISION, inputs, idempotency_key=KEY)
        assert first.created is True and replay.created is False
        assert (
            first.operation.operation_id == replay.operation.operation_id == OPERATION
        )
        result = call(
            "wait",
            OPERATION,
            contract_revision=REVISION,
            timeout=2,
            poll_interval=0.05,
            max_polls=2,
        )
    assert result.output == native_output()
    assert isinstance(result.output["native_extra"]["large"], int)
    assert admission.call_count == 2 and read.call_count == 2
    assert admission.calls[0].request.content == admission.calls[1].request.content
    assert json.loads(admission.calls[0].request.content) == {
        "contract_revision": REVISION,
        "inputs": inputs,
    }
    assert "Idempotency-Key" not in read.calls[0].request.headers
    for recorded in isolated_http.calls:
        assert recorded.request.headers["X-API-KEY"] == API_KEY
        assert recorded.request.headers["Accept-Encoding"] == "identity"
        assert "Authorization" not in recorded.request.headers
        assert "Cookie" not in recorded.request.headers
    assert dict(isolated_http.calls[0].request.url.params) == {
        "limit": "2",
        "prefix": "fixture/",
        "cursor": "earlier+cursor",
    }


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_cancel_is_one_explicit_request_not_a_stop_or_refund_claim(mode, isolated_http):
    route = isolated_http.post(ROOT + "operations/" + OPERATION + "/cancel/").mock(
        return_value=response(
            {"operation": operation(status="running", cancel_requested=True)}
        )
    )
    with native_client(mode) as (_, call):
        result = call("cancel", OPERATION, contract_revision=REVISION)
    assert result.cancel_requested and result.status == "running"
    assert route.call_count == 1
    assert route.calls[0].request.content == b"{}"
    assert "Idempotency-Key" not in route.calls[0].request.headers


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status", [201, 204, 301, 307, 401, 409, 429, 500])
def test_post_errors_never_retry_or_follow_redirect(mode, status, isolated_http):
    stream = Body(b"synthetic-private-error-body")
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=httpx.Response(
            status, stream=stream, headers={"Location": "https://redirect.invalid/"}
        )
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY)
    error = raised.value
    assert error.code == "http_status" and error.status_code == status
    assert error.outcome_unknown is True
    assert error.contract_revision == REVISION and error.idempotency_key == KEY
    assert route.call_count == 1 and len(isolated_http.calls) == 1
    assert stream.closed
    for private in (KEY, API_KEY, "synthetic-private-error-body", "redirect.invalid"):
        assert private not in repr(error) and private not in str(error)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError, OSError]
)
def test_lost_response_keeps_key_and_never_resubmits(mode, failure, isolated_http):
    route = isolated_http.post(ROOT + "operations/").mock(
        side_effect=failure("private transport data must not appear")
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY)
    assert raised.value.outcome_unknown is True
    assert raised.value.contract_revision == REVISION
    assert raised.value.idempotency_key == KEY
    assert raised.value.operation_id is None
    assert "private transport data" not in repr(raised.value)
    assert route.call_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status,created", [(200, True), (202, False), (202, 1)])
def test_admission_status_and_created_must_agree(mode, status, created, isolated_http):
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=response(
            {"operation": operation(), "created": created}, status=status
        )
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY)
    assert raised.value.code == "invalid_response"
    assert raised.value.outcome_unknown and raised.value.idempotency_key == KEY
    assert route.call_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "raw",
    [
        b'{"operation":',
        b'{"operation":null,"operation":null}',
        b'{"operation":{"unexpected":NaN}}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b"[]",
        b'{"x":"\xff"}',
    ],
)
def test_invalid_json_read_preserves_original_handle(mode, raw, isolated_http):
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        return_value=httpx.Response(200, stream=Body(raw))
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("get", OPERATION, contract_revision=REVISION)
    assert raised.value.code == "invalid_response"
    assert raised.value.operation_id == OPERATION
    assert raised.value.contract_revision == REVISION
    assert raised.value.outcome_unknown is False
    assert route.call_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "changes",
    [
        {"operation_id": "bbbbbbbb-1234-4321-abcd-123456789012"},
        {"prediction_id": "bbbbbbbb-1234-4321-abcd-123456789012"},
        {"contract_revision": "b" * 64},
        {"operation_id": "not-a-uuid"},
        {"cancel_requested": 1},
        {"status": "unexpected"},
        {"status": "succeeded"},
    ],
)
def test_invalid_metadata_cannot_replace_saved_handle(mode, changes, isolated_http):
    isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        return_value=response({"operation": operation(**changes)})
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("get", OPERATION, contract_revision=REVISION)
    assert raised.value.operation_id == OPERATION
    assert raised.value.contract_revision == REVISION
    assert not raised.value.outcome_unknown


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "case",
    [
        "missing-key",
        "edge-space",
        "non-ascii-key",
        "huge-key",
        "uppercase-revision",
        "invalid-uuid",
        "bool-timeout",
        "infinite-timeout",
        "negative-timeout",
        "oversized-utf8",
        "non-string-keys",
        "non-finite-input",
        "non-json-input",
        "circular-input",
        "catalog-limit",
        "catalog-cursor",
        "wait-polls",
        "wait-interval",
    ],
)
def test_invalid_inputs_refuse_before_http(mode, case, isolated_http):
    with native_client(mode) as (_, call):
        method, args, kwargs = "create", [REVISION, {}], {"idempotency_key": KEY}
        if case == "missing-key":
            kwargs = {}
        elif case == "edge-space":
            kwargs["idempotency_key"] = " trailing "
        elif case == "non-ascii-key":
            kwargs["idempotency_key"] = "ключ"
        elif case == "huge-key":
            kwargs["idempotency_key"] = "x" * 256
        elif case == "uppercase-revision":
            args[0] = REVISION.upper()
        elif case == "invalid-uuid":
            method, args, kwargs = (
                "get",
                ["../control"],
                {"contract_revision": REVISION},
            )
        elif case in {"bool-timeout", "infinite-timeout", "negative-timeout"}:
            kwargs["timeout"] = {
                "bool-timeout": True,
                "infinite-timeout": float("inf"),
                "negative-timeout": -1,
            }[case]
        elif case == "oversized-utf8":
            args[1] = {"text": "🐈" * 65536}
        elif case == "non-string-keys":
            args[1] = {1: "value"}
        elif case == "non-finite-input":
            args[1] = {"temperature": float("nan")}
        elif case == "non-json-input":
            args[1] = {"file": b"not-json"}
        elif case == "circular-input":
            args[1]["self"] = args[1]
        elif case == "catalog-limit":
            method, args, kwargs = "catalog", [], {"limit": True}
        elif case == "catalog-cursor":
            method, args, kwargs = "catalog", [], {"cursor": "x" * 2049}
        elif case.startswith("wait-"):
            method, args, kwargs = "wait", [OPERATION], {"contract_revision": REVISION}
            kwargs.update(
                {"max_polls": True} if case == "wait-polls" else {"poll_interval": 0}
            )
        with pytest.raises((ValueError, TypeError)):
            call(method, *args, **kwargs)
    assert len(isolated_http.calls) == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "case",
    ["client-closed", "invalid-timeout", "identity-changed", "base-url-userinfo"],
)
def test_transport_preflight_is_not_unknown(mode, case, isolated_http):
    if case == "base-url-userinfo":
        cls = AgentClient if mode == "sync" else AsyncAgentClient
        client = cls(
            api_key=API_KEY, base_url="https://unwanted:secret@sdk-native.invalid"
        )
        try:
            with pytest.raises(ModelProxyError) as raised:
                result = client.model_proxy.create(REVISION, {}, idempotency_key=KEY)
                if inspect.isawaitable(result):
                    asyncio.run(result)
        finally:
            client.close() if mode == "sync" else asyncio.run(client.close())
    else:
        with native_client(mode) as (client, call):
            if case == "client-closed":
                client.close() if mode == "sync" else asyncio.run(client.close())
            elif case == "invalid-timeout":
                client._http.timeout = httpx.Timeout(float("inf"))
            else:
                client._api_key = "different-caller"
            with pytest.raises(ModelProxyError) as raised:
                call("create", REVISION, {}, idempotency_key=KEY)
    assert not raised.value.outcome_unknown
    assert raised.value.contract_revision == REVISION
    assert raised.value.idempotency_key == KEY
    assert len(isolated_http.calls) == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "case", ["declared-large", "actual-large", "invalid-length", "encoded"]
)
def test_response_bounds_close_stream_and_keep_unknown_write(mode, case, isolated_http):
    headers = {}
    stream = Body(b"{}")
    if case == "declared-large":
        headers["Content-Length"] = str(2097153)
    elif case == "actual-large":
        stream = Body(b" " * 2097152, b" ")
    elif case == "invalid-length":
        headers["Content-Length"] = "-1"
    else:
        headers["Content-Encoding"] = "gzip"
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=httpx.Response(202, stream=stream, headers=headers)
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY)
    assert raised.value.code == (
        "encoded_response" if case == "encoded" else "response_too_large"
    )
    assert raised.value.outcome_unknown and raised.value.idempotency_key == KEY
    assert stream.closed and route.call_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", ["headers", "trickled-body"])
def test_whole_exchange_timeout_includes_headers_and_body(mode, case, isolated_http):
    stream = Body(*([b" "] * 8), delay=0.06)
    dispatched = []

    async def delayed_headers(request):
        dispatched.append(request)
        await asyncio.sleep(0.5)
        return response({"operation": operation(), "created": True}, status=202)

    route = isolated_http.post(ROOT + "operations/")
    if case == "headers":
        route.mock(side_effect=delayed_headers)
    else:
        route.mock(return_value=httpx.Response(202, stream=stream))
    with native_client(mode, timeout=0.2) as (_, call):
        start = time.monotonic()
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY, timeout=1)
        elapsed = time.monotonic() - start
    assert raised.value.code == "request_timeout" and raised.value.outcome_unknown
    assert 0.15 <= elapsed < 1.5
    assert (
        len(isolated_http.calls) <= 1
    )  # Interrupted headers may have no completed mock receipt.
    if case == "trickled-body":
        assert route.call_count == 1 and stream.closed
    else:
        assert len(dispatched) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_wrong_contract_revision_is_refused(mode, isolated_http):
    isolated_http.get(ROOT + "contracts/" + REVISION + "/").mock(
        return_value=response(
            {"contract_revision": "b" * 64, "artifact": {}, "execution_constraints": {}}
        )
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("contract", REVISION)
    assert raised.value.code == "identity_mismatch"
    assert raised.value.contract_revision == REVISION


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", ["duplicate-route", "too-many-items", "empty-release"])
def test_invalid_catalog_page_never_implies_a_published_model(
    mode, case, isolated_http
):
    summary = {
        "contract_revision": REVISION,
        "route_key": "fixture/route",
        "model_version": "digest",
        "operation": "chat",
        "profile": "broker.ds-chat.v1",
    }
    page = {"release": OPERATION, "items": [summary], "next_cursor": None}
    limit = 2
    if case == "duplicate-route":
        page["items"].append(dict(summary))
    elif case == "too-many-items":
        page["items"].append(dict(summary, route_key="fixture/other"))
        limit = 1
    else:
        page["release"] = None
    isolated_http.get(ROOT + "catalog/").mock(return_value=response(page))
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError, match="invalid_response"):
            call("catalog", limit=limit)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "case", ["truncated-json", "invalid-envelope", "invalid-status"]
)
def test_malformed_post_receipt_is_unknown_with_original_key(mode, case, isolated_http):
    body = response({"operation": None, "created": True}, status=202)
    if case == "truncated-json":
        body = httpx.Response(202, stream=Body(b'{"created":true,'))
    elif case == "invalid-status":
        body = response(
            {"operation": operation(status="future-state"), "created": True}, status=202
        )
    route = isolated_http.post(ROOT + "operations/").mock(return_value=body)
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY)
    assert raised.value.outcome_unknown and raised.value.idempotency_key == KEY
    assert raised.value.contract_revision == REVISION
    if case == "invalid-status":
        assert raised.value.operation_id == OPERATION
    assert route.call_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_wait_unknown_returns_once_without_any_post(mode, isolated_http):
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        return_value=response(
            {"operation": operation(status="unknown", error={"code": "lost_response"})}
        )
    )
    with native_client(mode) as (_, call):
        result = call("wait", OPERATION, contract_revision=REVISION)
    assert result.status == "unknown" and result.operation_id == OPERATION
    assert result.error == {"code": "lost_response"}
    assert route.call_count == 1 and len(isolated_http.calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", ["max-polls", "stopped", "deadline"])
def test_wait_ends_with_saved_handle_and_no_automatic_cancel(mode, case, isolated_http):
    if case != "stopped":
        route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
            side_effect=lambda request: response(
                {"operation": operation(status="running")}
            )
        )
    kwargs = {"max_polls": 2, "timeout": 2, "poll_interval": 0.05}
    if case == "stopped":
        kwargs["stopped"] = lambda: True
    elif case == "deadline":
        kwargs.update(timeout=0.2, poll_interval=1, max_polls=10)
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("wait", OPERATION, contract_revision=REVISION, **kwargs)
    assert (
        raised.value.code
        == {
            "max-polls": "poll_limit",
            "stopped": "wait_aborted",
            "deadline": "wait_timeout",
        }[case]
    )
    assert raised.value.operation_id == OPERATION
    assert raised.value.contract_revision == REVISION
    assert all(recorded.request.method == "GET" for recorded in isolated_http.calls)
    if case == "stopped":
        assert len(isolated_http.calls) == 0
    else:
        assert route.call_count == (2 if case == "max-polls" else 1)


@pytest.mark.asyncio
async def test_task_cancellation_closes_exchange_without_server_cancel(isolated_http):
    entered = asyncio.Event()
    stream = Body(b" ", delay=10)

    async def handler(request):
        entered.set()
        return httpx.Response(202, stream=stream)

    route = isolated_http.post(ROOT + "operations/").mock(side_effect=handler)
    async with AsyncAgentClient(api_key=API_KEY, base_url=ORIGIN) as client:
        task = asyncio.create_task(
            client.model_proxy.create(REVISION, {}, idempotency_key=KEY)
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert stream.closed and route.call_count == 1
    assert len(isolated_http.calls) == 1


@pytest.mark.asyncio
async def test_account_switch_during_admission_cannot_deliver_stale_result(
    isolated_http,
):
    async with AsyncAgentClient(api_key=API_KEY, base_url=ORIGIN) as client:

        async def switch_account(request):
            assert request.headers["X-API-KEY"] == API_KEY
            client._api_key = "new-caller"
            return response({"operation": operation(), "created": True}, status=202)

        route = isolated_http.post(ROOT + "operations/").mock(
            side_effect=switch_account
        )
        with pytest.raises(ModelProxyError) as raised:
            await client.model_proxy.create(REVISION, {}, idempotency_key=KEY)
        assert raised.value.code == "client_identity_changed"
        assert raised.value.outcome_unknown
        assert raised.value.idempotency_key == KEY
        with pytest.raises(ModelProxyError) as reread:
            await client.model_proxy.get(OPERATION, contract_revision=REVISION)
        assert reread.value.code == "client_identity_changed"
    assert route.call_count == 1 and len(isolated_http.calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", ["positive", "negative", "aggregate"])
def test_integer_input_bound_precedes_decimal_encoder(
    mode, case, monkeypatch, isolated_http
):
    inputs = {"counter": 10**4096}
    if case == "negative":
        inputs["counter"] = -inputs["counter"]
    elif case == "aggregate":
        # Individual integers fit, but their total decimal text exceeds the
        # input budget. Reject before building that JSON string.
        inputs = {"counters": [10**3000] * 100}
    with native_client(mode) as (_, call):

        def forbidden_encoder(*args, **kwargs):
            raise AssertionError("decimal JSON encoder reached before integer bound")

        monkeypatch.setattr(json, "dumps", forbidden_encoder)
        with pytest.raises(ValueError):
            call("create", REVISION, inputs, idempotency_key=KEY)
    assert len(isolated_http.calls) == 0


@pytest.mark.parametrize("method", ["get", "create"])
def test_integer_response_bound_without_interpreter_guard(method):
    # Python3.8 has no interpreter digit guard. Disable the newer guard only in
    # this bounded, isolated subprocess and feed a 5000-digit mock response.
    # No process-global setting in the SDK or parent test process is changed.
    program = r"""
import json
import sys
import httpx
import respx
from flymyai.agents import AgentClient, ModelProxyError

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
method = sys.argv[1]
revision = "a" * 64
operation_id = "aaaaaaaa-1234-4321-abcd-123456789012"
key = "original-integer-bound-intent"
op = {"operation_id": operation_id, "prediction_id": operation_id,
      "contract_revision": revision, "status": "succeeded", "stage": "submit",
      "cancel_requested": False, "output": {"counter": "INTEGER_MARKER"}}
payload = {"operation": op}
if method == "create":
    payload["created"] = True
raw = json.dumps(payload).replace('"INTEGER_MARKER"', "9" * 5000).encode()
assert len(raw) < 6000
class Body(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield raw

origin = "https://integer-bound.invalid"
suffix = "/api/v1/model-proxy/operations/" + (operation_id + "/" if method == "get" else "")
with respx.mock(assert_all_mocked=True, assert_all_called=True) as router:
    route = router.request("GET" if method == "get" else "POST", origin + suffix).mock(
        return_value=httpx.Response(200 if method == "get" else 202, stream=Body()))
    with AgentClient(api_key="synthetic-key", base_url=origin) as client:
        try:
            if method == "get":
                client.model_proxy.get(operation_id, contract_revision=revision)
            else:
                client.model_proxy.create(revision, {}, idempotency_key=key)
        except ModelProxyError as error:
            assert error.code == "invalid_response"
            assert error.contract_revision == revision
            assert error.outcome_unknown is (method == "create")
            if method == "get":
                assert error.operation_id == operation_id
            else:
                assert error.idempotency_key == key
        else:
            raise AssertionError("oversized integer accepted without explicit SDK digit bound")
    assert route.call_count == 1
if hasattr(sys, "get_int_max_str_digits"):
    assert sys.get_int_max_str_digits() == 0
print("bounded integer response refused once; recovery identity retained")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, method],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-2000:]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_integer_4096_digit_boundary_is_exact_both_directions(
    mode, sign, isolated_http
):
    # Keep the interpreter's own stricter setting, if configured, untouched.
    if hasattr(sys, "get_int_max_str_digits"):
        configured = sys.get_int_max_str_digits()
        if 0 < configured < 4096:
            pytest.skip(
                "interpreter digit setting is stricter than this boundary fixture"
            )
    value = sign * (10**4096 - 1)
    native = {"counter": value, "above_2_to_53": 2**100 + 17, "zero": 0}
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=response(
            {
                "operation": operation(status="succeeded", output=native),
                "created": True,
            },
            status=202,
        )
    )
    with native_client(mode) as (_, call):
        result = call("create", REVISION, native, idempotency_key=KEY)
    assert result.operation.output == native
    assert json.loads(route.calls[0].request.content)["inputs"] == native
    assert route.call_count == 1


def queued(code):
    return response({"operation": operation(status="queued", error={"code": code})})


COLD_STARTS = [
    ("h3_capacity_not_ready", "MiniMax-H3 is starting", "about 15 minutes"),
    ("h3_model_busy", "MiniMax-H3 is busy", "about 15 minutes"),
    ("ds_capacity_not_ready", "DeepSeek is starting", "up to 50 minutes"),
    ("ds_model_busy", "DeepSeek is busy", "up to 50 minutes"),
]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("code,state,cold_start", COLD_STARTS)
def test_queued_operation_says_plainly_why_it_waits(
    mode, code, state, cold_start, isolated_http
):
    isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        side_effect=lambda request: queued(code)
    )
    with native_client(mode) as (_, call):
        result = call("get", OPERATION, contract_revision=REVISION)
    assert result.status == "queued" and result.waiting_for_capacity is True
    assert result.error_code == code and result.error == {"code": code}
    assert state in result.reason and cold_start in result.reason
    assert "not a failure" in result.reason
    assert "never call create()" in result.reason


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("code,state,cold_start", COLD_STARTS)
@pytest.mark.parametrize("case", ["deadline", "max-polls", "stopped"])
def test_wait_ending_during_a_cold_start_keeps_the_operation_and_its_reason(
    mode, code, state, cold_start, case, isolated_http
):
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        side_effect=lambda request: queued(code)
    )
    seen = []
    kwargs = {"timeout": 5, "poll_interval": 0.05, "max_polls": 3}
    if case == "deadline":
        # Long enough for the first read on a busy runner; more polls than time.
        kwargs.update(timeout=1.5, poll_interval=0.2, max_polls=50)
    elif case == "stopped":
        kwargs["stopped"] = lambda: bool(seen)
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call(
                "wait",
                OPERATION,
                contract_revision=REVISION,
                on_update=seen.append,
                **kwargs,
            )
    error = raised.value
    assert (
        error.code
        == {
            "deadline": "wait_timeout",
            "max-polls": "poll_limit",
            "stopped": "wait_aborted",
        }[case]
    )
    assert error.operation_id == OPERATION and error.contract_revision == REVISION
    assert error.outcome_unknown is False
    assert error.last_status == "queued" and error.waiting_code == code
    assert state in error.reason and cold_start in error.reason
    assert state in str(error) and cold_start in str(error)
    assert OPERATION not in str(error) and OPERATION not in repr(error)
    assert route.call_count >= 1
    assert all(recorded.request.method == "GET" for recorded in isolated_http.calls)
    assert [item.error_code for item in seen] == [code], "reported once per change"


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_wait_reports_each_change_on_the_same_operation(mode, isolated_http):
    output = {"native": {"id": "job"}, "video": {"url": "https://storage.invalid/v"}}
    answers = [
        queued("h3_capacity_not_ready"),
        queued("h3_capacity_not_ready"),
        queued("h3_model_busy"),
        response({"operation": operation(status="running", stage="poll")}),
        response({"operation": operation(status="file_pending", stage="files")}),
        response({"operation": operation(status="succeeded", output=output)}),
    ]
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        side_effect=answers
    )
    seen = []
    with native_client(mode) as (_, call):
        result = call(
            "wait",
            OPERATION,
            contract_revision=REVISION,
            poll_interval=0.05,
            on_update=lambda item: seen.append((item.status, item.error_code)),
        )
    assert result.status == "succeeded" and result.output == output
    assert result.reason is None
    assert seen == [
        ("queued", "h3_capacity_not_ready"),
        ("queued", "h3_model_busy"),
        ("running", None),
        ("file_pending", None),
        ("succeeded", None),
    ]
    assert route.call_count == len(answers) and len(isolated_http.calls) == len(answers)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "status,code,words",
    [
        ("rejected", "h3_capacity_not_ready", "ran out"),
        ("rejected", "h3_model_busy", "ran out"),
        ("rejected", "ds_capacity_not_ready", "ran out"),
        ("rejected", "ds_model_busy", "ran out"),
        ("rejected", "proxy_deadline", "did not start in time"),
        ("failed", "model_lease_deleted", "no result and no charge"),
        ("failed", "model_http_500", "HTTP 500. There is no result and no charge"),
        ("unknown", "lost_response", "no automatic resubmission"),
    ],
)
def test_ends_of_a_wait_say_what_happened_to_the_result_and_the_money(
    mode, status, code, words, isolated_http
):
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        return_value=response(
            {"operation": operation(status=status, error={"code": code})}
        )
    )
    with native_client(mode) as (_, call):
        result = call("wait", OPERATION, contract_revision=REVISION)
    assert result.status == status and result.error_code == code
    assert result.waiting_for_capacity is False
    assert words in result.reason
    if status == "rejected":
        assert "nothing was charged" in result.reason
    assert route.call_count == 1 and len(isolated_http.calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "status,code",
    [
        ("failed", "other_reason"),
        ("failed", "proxy_deadline"),
        ("rejected", "other_reason"),
        ("cancelled", "h3_capacity_not_ready"),
    ],
)
def test_other_ends_claim_nothing_about_the_money(mode, status, code, isolated_http):
    isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        return_value=response(
            {"operation": operation(status=status, error={"code": code})}
        )
    )
    with native_client(mode) as (_, call):
        result = call("get", OPERATION, contract_revision=REVISION)
    assert result.reason is None and result.waiting_for_capacity is False


def test_plain_unfinished_operation_is_not_called_a_capacity_wait(isolated_http):
    isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        return_value=response({"operation": operation(status="queued")})
    )
    with native_client("sync") as (_, call):
        result = call("get", OPERATION, contract_revision=REVISION)
    assert result.waiting_for_capacity is False and result.error_code is None
    assert result.reason.startswith("Not finished yet.")


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(503, stream=Body(b"synthetic-private-error-body")),
        httpx.Response(429, stream=Body(b"{}")),
        httpx.ReadTimeout("private transport data"),
        httpx.ConnectError("private transport data"),
    ],
)
def test_one_lost_read_does_not_end_a_wait_once_a_state_is_known(
    mode, failure, isolated_http
):
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        side_effect=[
            queued("h3_capacity_not_ready"),
            failure,
            response({"operation": operation(status="unknown")}),
        ]
    )
    with native_client(mode) as (_, call):
        result = call("wait", OPERATION, contract_revision=REVISION, poll_interval=0.05)
    assert result.status == "unknown"
    assert route.call_count == 3 and len(isolated_http.calls) == 3
    assert all(recorded.request.method == "GET" for recorded in isolated_http.calls)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_only_lost_reads_end_a_wait_with_the_last_state(mode, isolated_http):
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        side_effect=[queued("h3_model_busy")]
        + [httpx.Response(503, stream=Body(b"{}")) for _ in range(5)]
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call(
                "wait",
                OPERATION,
                contract_revision=REVISION,
                poll_interval=0.05,
                max_polls=3,
            )
    assert raised.value.code == "poll_limit"
    assert raised.value.waiting_code == "h3_model_busy"
    assert route.call_count == 3


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "answers,code",
    [
        ([httpx.Response(503, stream=Body(b"{}"))], "http_status"),
        ([httpx.ReadTimeout("private transport data")], "transport_error"),
        (
            [
                queued("h3_model_busy"),
                httpx.Response(
                    404,
                    stream=Body(b'{"code":"proxy_operation_missing"}'),
                    headers={"Content-Type": "application/json"},
                ),
            ],
            "http_status",
        ),
        ([queued("h3_model_busy"), httpx.Response(200, stream=Body(b"[]"))], None),
    ],
)
def test_a_wait_stops_when_nothing_is_known_or_the_answer_is_final(
    mode, answers, code, isolated_http
):
    route = isolated_http.get(ROOT + "operations/" + OPERATION + "/").mock(
        side_effect=answers
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("wait", OPERATION, contract_revision=REVISION, poll_interval=0.05)
    error = raised.value
    if code is not None and isinstance(answers[0], httpx.ReadTimeout):
        assert error.code in {"transport_error", "request_timeout"}
    elif code is not None:
        assert error.code == code
    else:
        assert error.code == "invalid_response"
    assert error.operation_id == OPERATION and error.outcome_unknown is False
    assert route.call_count == len(answers)


def h3_reference(index, kind):
    return {
        "schema": "flymyai.artifact/v1",
        "id": "00000000-0000-4000-8000-" + str(index).zfill(12),
        "version": {
            "id": "00000000-0000-4000-9000-" + str(index).zfill(12),
            "revision": 1,
            "size": 2**53 + 1,
            "sha256": "d" * 64,
            "mime": "image/png" if kind == "image" else "audio/wav",
        },
        "kind": kind,
        "name": "ref " + str(index),
        "workspace": None,
        "relations": {},
        "legacy": {"agent_file_id": 2**53 + 1, "external_id": None},
    }


def h3_inputs(images, audio):
    kinds = ["image"] * images + ["audio"] * audio
    return {
        "prompt": "a chef plates a dish",
        "conditions": [
            {"type": kind, "artifact": h3_reference(index, kind)}
            for index, kind in enumerate(kinds)
        ],
        "duration_seconds": 5,
        "aspect_ratio": "21:9",
        "seed": 2147483647,
    }


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_the_largest_h3_input_is_sent_whole_and_in_order(mode, isolated_http):
    inputs = h3_inputs(9, 3)
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=response({"operation": operation(), "created": True}, status=202)
    )
    with native_client(mode) as (_, call):
        call("create", REVISION, inputs, idempotency_key=KEY)
    sent = json.loads(route.calls[0].request.content)["inputs"]
    assert sent == inputs and len(sent["conditions"]) == 12
    assert [item["type"] for item in sent["conditions"]] == ["image"] * 9 + [
        "audio"
    ] * 3


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("images,audio", [(10, 0), (0, 4), (12, 1)])
def test_bounds_stay_the_backends_and_its_refusal_is_named(
    mode, images, audio, isolated_http
):
    inputs = h3_inputs(images, audio)
    stream = Body(
        b'{"code":"h3_references","detail":"synthetic-private-error-body"}',
    )
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=httpx.Response(
            400, stream=stream, headers={"Content-Type": "application/json"}
        )
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, inputs, idempotency_key=KEY)
    error = raised.value
    assert error.code == "http_status" and error.status_code == 400
    assert error.refusal_code == "h3_references"
    assert error.outcome_unknown is False, "a named refusal started nothing"
    assert "h3_references" in str(error)
    assert error.contract_revision == REVISION and error.idempotency_key == KEY
    assert route.call_count == 1 and stream.closed
    assert json.loads(route.calls[0].request.content)["inputs"] == inputs
    for private in (KEY, API_KEY, "synthetic-private-error-body"):
        assert private not in repr(error) and private not in str(error)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "status,code",
    [
        (403, "h3_input_authority"),
        (403, "h3_team_unavailable"),
        (409, "idempotency_key_reused"),
        (429, "proxy_pending_limit"),
    ],
)
def test_named_refusals_of_an_admission_are_known(mode, status, code, isolated_http):
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=httpx.Response(
            status,
            stream=Body(json.dumps({"code": code}).encode()),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY)
    assert raised.value.refusal_code == code
    assert raised.value.outcome_unknown is False
    assert route.call_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "status,body,content_type",
    [
        (400, b'{"code":"Has Spaces And /private/path"}', "application/json"),
        (400, b'{"code":7}', "application/json"),
        (400, b"[]", "application/json"),
        (400, b"not json", "application/json"),
        (400, b'{"code":"h3_references"}', "text/plain"),
        (401, b'{"code":"token_not_valid"}', "application/json"),
        (408, b'{"code":"proxy_timeout"}', "application/json"),
        (503, b'{"code":"proxy_signing_key"}', "application/json"),
        (400, b" " * 4096 + b'{"code":"h3_references"}', "application/json"),
    ],
)
def test_an_answer_naming_no_fixed_code_stays_unknown(
    mode, status, body, content_type, isolated_http
):
    route = isolated_http.post(ROOT + "operations/").mock(
        return_value=httpx.Response(
            status, stream=Body(body), headers={"Content-Type": content_type}
        )
    )
    with native_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("create", REVISION, {}, idempotency_key=KEY)
    assert raised.value.code == "http_status"
    assert raised.value.refusal_code is None
    assert raised.value.outcome_unknown is True
    assert "private" not in str(raised.value) and route.call_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_catalog_selects_one_exact_model(mode, isolated_http):
    summary = {
        "contract_revision": REVISION,
        "route_key": "broker/h3",
        "model_version": "digest",
        "operation": "generate",
        "profile": "broker.h3-ref2va.v1",
        "project": {"id": "9007199254740993", "owner": "flymyai", "name": "h3"},
    }
    route = isolated_http.get(ROOT + "catalog/").mock(
        return_value=response({
            "release": OPERATION,
            "items": [summary],
            "next_cursor": None,
            "selection": {
                "project": summary["project"],
                "native_model_proxy": True,
            },
        })
    )
    with native_client(mode) as (_, call):
        page = call("catalog", limit=2, endpoint_id="flymyai/h3")
    assert dict(route.calls[0].request.url.params) == {
        "limit": "2",
        "prefix": "",
        "endpoint_id": "flymyai/h3",
    }
    assert page.items[0].contract_revision == REVISION
    assert page.items[0].profile == "broker.h3-ref2va.v1"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "endpoint_id", ["", "no-slash", "a/b/c", "/name", "owner/", 7, "o" * 151 + "/n"]
)
def test_catalog_refuses_a_bad_endpoint_before_http(mode, endpoint_id, isolated_http):
    with native_client(mode) as (_, call):
        with pytest.raises(ValueError):
            call("catalog", endpoint_id=endpoint_id)
    assert len(isolated_http.calls) == 0
