"""Finite SDK transport conformance, with no backend/provider lifecycle claim.

All requests use a real httpx client and MockTransport. Responses are lazy
streams so the SDK's bounded read path, including stream closure, is exercised.
Only explicit caller actions add scripted requests; unexpected HTTP fails at once.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import UUID

import httpx
import pytest

from flymyai.agents import (
    AppendMessageResponse,
    AsyncAgentClient,
    CodingAvailability,
    CodingContinuation,
    ExecutionStatus,
    FileMetadata,
    FilePage,
    FileRange,
    FilesContractUnsupportedError,
    FlyMyAIAgentError,
    RunDetail,
    RunLogPage,
    RunPresentationCursor,
    RunStatus,
    RunStep,
    RunTranscriptPage,
    RunObservationUnsupportedError,
    SyncAgentClient,
)

pytestmark = pytest.mark.asyncio

API_KEY = "sdk-controls-fixture-key"
BASE = "/api/v1/agents"
RUN_ID = "sdk-controls-run"
NEXT_RUN_ID = "sdk-controls-next"
AGENT_ID = "aaaaaaaa-0000-0000-0000-000000000001"
GROUP_ID = "bbbbbbbb-0000-0000-0000-000000000002"
WORKSPACE = "ws_01ARZ3NDEKTSV4RRFFQ69G5FAV"
PERSONAL = "ws_01ARZ3NDEKTSV4RRFFQ69G5FAW"
ARTIFACT = "af_01ARZ3NDEKTSV4RRFFQ69G5FAX"
NOW = "2026-09-09T12:00:00Z"
PAGE_BYTES = 512 * 1024
RANGE_BYTES = 65536


class _Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content
        self.closed = False
        self.delivered = 0

    def __iter__(self):
        for offset in range(0, len(self.content), 16384):
            chunk = self.content[offset:offset + 16384]
            self.delivered += len(chunk)
            yield chunk

    async def __aiter__(self):
        for chunk in self:
            yield chunk

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


@dataclass
class _Exchange:
    method: str
    path: str
    payload: object = None
    status: int = 200
    params: dict = field(default_factory=dict)
    request_json: object = None
    key: str | None = None
    request_headers: dict = field(default_factory=dict)
    response_headers: dict = field(default_factory=dict)
    raw: bytes | None = None
    timeout: bool = False


class _Script:
    def __init__(self, exchanges):
        assert 1 <= len(exchanges) <= 12
        self.exchanges = exchanges
        self.requests = []
        self.streams = []

    def __call__(self, request):
        index = len(self.requests)
        assert index < len(self.exchanges), "Unscripted HTTP or automatic retry"
        expected = self.exchanges[index]
        assert request.url.scheme == "https"
        assert request.url.host == "sdk.invalid"
        assert request.method == expected.method
        assert request.url.path == expected.path
        assert sorted(request.url.params.multi_items()) == sorted(
            (key, str(value)) for key, value in expected.params.items()
        )
        assert len(request.content) <= 2048
        body = json.loads(request.content) if request.content else None
        assert body == expected.request_json
        assert request.headers.get("X-API-KEY") == API_KEY
        assert request.headers.get("Authorization") is None
        assert request.headers.get("Idempotency-Key") == expected.key
        assert request.headers.get("Range") == expected.request_headers.get("Range")
        if request.method == "GET":
            assert request.headers["Accept-Encoding"] == "identity"
        for key, value in expected.request_headers.items():
            assert request.headers[key] == value
        self.requests.append({
            "method": request.method,
            "path": request.url.path,
            "params": dict(request.url.params),
            "body": request.content,
            "headers": dict(request.headers),
        })
        if expected.timeout:
            raise httpx.ReadTimeout("Scripted transport outcome unknown", request=request)
        content = expected.raw
        if content is None:
            content = b"" if expected.payload is None else json.dumps(
                expected.payload, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
        assert len(content) <= PAGE_BYTES + 1
        stream = _Body(content)
        self.streams.append(stream)
        return httpx.Response(
            expected.status,
            headers={"Content-Type": "application/json", **expected.response_headers},
            stream=stream,
            request=request,
        )


async def _call(method, *args, **kwargs):
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.fixture(params=("sync", "async"), ids=("sync", "async"))
def sdk_mode(request):
    return request.param


@pytest.fixture
def open_client(monkeypatch, sdk_mode):
    @asynccontextmanager
    async def open_script(exchanges):
        script = _Script(exchanges)
        http_name = "Client" if sdk_mode == "sync" else "AsyncClient"
        native_http = getattr(httpx, http_name)

        def with_transport(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(script)
            return native_http(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(httpx, http_name, with_transport)
            client_type = SyncAgentClient if sdk_mode == "sync" else AsyncAgentClient
            client = client_type(
                api_key=API_KEY, base_url="https://sdk.invalid", timeout=1.0,
                max_retries=7,
            )
            try:
                yield client, script
                assert len(script.requests) == len(script.exchanges)
            finally:
                await _call(client.close)
                assert all(stream.closed for stream in script.streams)

    return open_script


def _ack(run_id=RUN_ID):
    # Exact compact CodingRunResponseSerializer fields, without detail/history.
    return {
        "id": run_id, "status": "pending", "effort": "max", "model": "pinned-model",
        "run_seq": 0, "user_agent_task": 7, "user_agent_task_uuid": AGENT_ID,
        "created_at": NOW, "updated_at": NOW,
    }


def _status(*, run_id=RUN_ID, status="running", run_seq=3, since=0,
            step_ids=(), has_more=False, as_of_seq=100, goal=None,
            result=None, error=None, runtime_admission=None):
    terminal = status in ("completed", "failed", "cancelled", "archived")
    steps = []
    for step_id in step_ids:
        message = f"Step {step_id}"
        steps.append({
            "id": step_id, "type": "info", "message": message,
            "message_size_bytes": len(message.encode("utf-8")), "message_truncated": False,
            "label": message, "label_size_bytes": len(message.encode("utf-8")),
            "label_truncated": False,
        })
    return {
        "view": "bounded_v1", "id": run_id, "status": status, "run_seq": run_seq,
        "updated_at": NOW, "is_settled": terminal, "step_count": len(steps),
        "tool_step_count": 0, "last_step_id": steps[-1]["id"] if steps else None,
        "new_steps": steps, "agent_surface_revision": 9, "page_size": 20,
        "has_more": has_more, "next_since": steps[-1]["id"] if steps else since,
        "poll_complete": terminal and not has_more, "step_count_has_more": False,
        "presentation_cursor_v1": {
            "version": "presentation_cursor_v1", "run_seq": run_seq, "as_of_seq": as_of_seq,
        },
        "goal": goal, "runtime_admission": runtime_admission, "result": result, "error": error,
    }


def _status_get(payload, since=0, page_size=20):
    return _Exchange(
        "GET", f"{BASE}/executions/{payload['id']}/status/", payload,
        params={"view": "bounded_v1", "since": since, "page_size": page_size},
    )


def _resource(content, *, kind="agent_result", truncated=False):
    digest = hashlib.sha256(content).hexdigest()
    inline = content[:24] if truncated else content
    is_json = kind == "agent_result" and not truncated
    return {
        "version": "execution_resource_v1",
        "ref": f"execution-resource:v1:{RUN_ID}:3:{kind}:sha256:{digest}:hmac-sha256:" + "a" * 64,
        "kind": kind, "media_type": "application/json" if kind == "agent_result" else "text/plain",
        "encoding": "utf-8", "size_bytes": len(content), "sha256": digest,
        "inline": {
            "format": "json" if is_json else "text",
            "value": json.loads(inline) if is_json else inline.decode("utf-8"),
            "size_bytes": len(inline), "truncated": truncated,
        },
        "truncated": truncated,
        "retrieval": {
            "href": "https://untrusted.invalid/whole-history",
            "accept_ranges": "bytes", "max_range_bytes": RANGE_BYTES,
        },
        "receipt": {"kind": "chunked_postgres_v1", "chunk_bytes": RANGE_BYTES},
    }


async def test_owner_coding_entry_keeps_omissions_and_exact_workspace(open_client):
    variants = [
        ({}, {}),
        ({"coding": True, "workspace": WORKSPACE, "variables": {"text": "Bounded input"}},
         {"coding": True, "workspace": WORKSPACE, "variables": {"text": "Bounded input"}}),
        ({"coding": False}, {"coding": False}),
        ({"workspace": WORKSPACE}, {"workspace": WORKSPACE}),
    ]
    exchanges = []
    for surface, path in (
        ("agent", f"{BASE}/tasks/{AGENT_ID}/run-loop/"),
        ("frozen", f"{BASE}/compilations/17/run-instruction/"),
    ):
        for index, (_, fields) in enumerate(variants):
            body = {"variables": {}, **fields} if surface == "agent" else fields or None
            response = _ack()
            if fields.get("coding") is False:
                response.update(messages=[], logs=[], variables={}, original_prompt="Legacy input",
                                runtime_admission={"runtime": "legacy", "admission": "legacy",
                                                   "continuation": {"append": True, "fork": True,
                                                                    "resume": False, "reason": None}})
            exchanges.append(_Exchange(
                "POST", path, response, status=201, request_json=body,
                key=f"{surface}-intent-{index}",
            ))
    async with open_client(exchanges) as (client, script):
        for surface, method, target in (
            ("agent", client.agents.run, AGENT_ID),
            ("frozen", client.compilations.run_instruction, 17),
        ):
            for index, (kwargs, _) in enumerate(variants):
                ack = await _call(method, target, idempotency_key=f"{surface}-intent-{index}", **kwargs)
                assert isinstance(ack, RunDetail)
                assert (ack.id, ack.run_seq, ack.status) == (RUN_ID, 0, ExecutionStatus.PENDING)
                assert (ack.user_agent_task, ack.user_agent_task_uuid) == (7, AGENT_ID)
                assert ack.created_at.isoformat() == "2026-09-09T12:00:00+00:00"
                assert ack.messages == [] and ack.logs == []
            with pytest.raises(ValueError):
                await _call(method, target, idempotency_key="invalid-selection",
                            coding=False, workspace=WORKSPACE)
        assert len(script.requests) == 8


async def test_control_ack_and_owner_status_parse_typed_projections(open_client):
    admission = {
        "runtime": "coding", "admission": "pinned",
        "continuation": {"append": False, "fork": False, "resume": True,
                         "reason": "coding_continue_required"},
    }
    page = _status(status="completed", since=11, step_ids=(12,),
                   runtime_admission=admission, result=_resource(b'{"ok":true}'))
    page["page_size"] = 100
    exchanges = [
        _Exchange("GET", f"{BASE}/tasks/{AGENT_ID}/coding-availability/",
                  {"default": "coding", "available": True, "code": None, "detail": None}),
        _Exchange("POST", f"{BASE}/executions/legacy-run/append-message/",
                  {"id": "legacy-run", "status": "running", "effort": "medium",
                   "model": "legacy-model", "run_seq": 4},
                  request_json={"text": "Continue legacy work", "effort": "medium"}),
        _status_get(page, since=11, page_size=100),
    ]
    async with open_client(exchanges) as (client, script):
        availability = await _call(client.agents.coding_availability, AGENT_ID)
        assert isinstance(availability, CodingAvailability)
        assert availability.default == "coding" and availability.available
        assert availability.code is None and availability.detail is None
        ack = await _call(client.runs.append_message, "legacy-run",
                          text="Continue legacy work", effort="medium")
        assert isinstance(ack, AppendMessageResponse)
        assert (ack.id, ack.status, ack.run_seq) == ("legacy-run", ExecutionStatus.RUNNING, 4)
        assert not ack.is_terminal
        observed = await _call(client.runs.status, RUN_ID, since=11, page_size=100)
        assert isinstance(observed, RunStatus)
        assert observed.is_terminal and observed.poll_complete
        assert observed.output == {"ok": True}
        assert observed.runtime_admission.runtime == "coding"
        capabilities = observed.runtime_admission.continuation
        assert (capabilities.append, capabilities.fork, capabilities.resume) == (False, False, True)
        assert capabilities.reason == "coding_continue_required"
        assert isinstance(observed.new_steps[0], RunStep)
        assert observed.new_steps[0].observed_run_seq == 3
        assert len(script.requests) == 3


async def test_continuation_requires_explicit_identical_replay(open_client):
    path = f"{BASE}/executions/{RUN_ID}/continue-coding/"
    key = "continue-original-intent"
    body = {"text": "Continue the saved goal"}
    accepted = {"id": NEXT_RUN_ID, "previous_execution": RUN_ID, "status": "pending",
                "effort": "max", "model": "pinned-model", "run_seq": 0}
    unknown = {"code": "tool_outcome_unknown", "detail": "Operation outcome is unknown."}
    unknown_source = "sdk-controls-unknown"
    unknown_path = f"{BASE}/executions/{unknown_source}/continue-coding/"
    unknown_key = "continue-unknown-intent"
    exchanges = [
        _Exchange("POST", path, {"detail": "Temporarily unavailable"}, status=503,
                  request_json=body, key=key),
        _Exchange("POST", path, request_json=body, key=key, timeout=True),
        _Exchange("POST", path, accepted, status=201, request_json=body, key=key),
        _Exchange("POST", path, accepted, status=201, request_json=body, key=key),
        _status_get(_status(run_id=NEXT_RUN_ID, status="completed", run_seq=0)),
        _Exchange("POST", unknown_path, unknown, status=409, request_json=body, key=unknown_key),
        _Exchange("POST", unknown_path, unknown, status=409, request_json=body, key=unknown_key),
    ]
    async with open_client(exchanges) as (client, script):
        with pytest.raises(FlyMyAIAgentError) as transient:
            await _call(client.runs.continue_coding, RUN_ID, text=body["text"], idempotency_key=key)
        assert transient.value.status_code == 503
        assert len(script.requests) == 1
        with pytest.raises(httpx.ReadTimeout):
            await _call(client.runs.continue_coding, RUN_ID, text=body["text"], idempotency_key=key)
        assert len(script.requests) == 2
        # These are explicit caller replays of one request, not SDK retries.
        first = await _call(client.runs.continue_coding, RUN_ID, text=body["text"], idempotency_key=key)
        replay = await _call(client.runs.continue_coding, RUN_ID, text=body["text"], idempotency_key=key)
        assert isinstance(first, CodingContinuation)
        assert first == replay
        assert (first.id, first.previous_execution, first.run_seq) == (NEXT_RUN_ID, RUN_ID, 0)
        observed = await _call(client.runs.wait, first.id, run_seq=first.run_seq,
                               timeout=5, poll_interval=0)
        assert observed.id == NEXT_RUN_ID and observed.status == "completed"
        writes = script.requests[:4]
        assert all(request["body"] == writes[0]["body"] for request in writes)
        assert all(request["headers"]["idempotency-key"] == key for request in writes)
        # A durable unknown receipt on another source stays unknown on explicit
        # replay. The fixture never turns that terminal refusal into success.
        for expected_count in (6, 7):
            with pytest.raises(FlyMyAIAgentError) as uncertain:
                await _call(client.runs.continue_coding, unknown_source,
                            text=body["text"], idempotency_key=unknown_key)
            assert uncertain.value.status_code == 409
            assert uncertain.value.response_body == unknown
            assert len(script.requests) == expected_count
        assert script.requests[5]["body"] == script.requests[6]["body"]


async def test_cancel_ack_or_uncertainty_keeps_terminal_observation(open_client):
    goal = {
        "schema": "flymyai.goal-progress/v1", "goal_id": GROUP_ID, "status": "verified",
        "review_count": 1, "reason": "verified", "terminal": False, "run_seq": 8, "revision": 2,
    }
    exchanges = [
        _Exchange("POST", f"{BASE}/executions/{RUN_ID}/cancel/", status=204),
        _status_get(_status(run_seq=3)),
        _status_get(_status(status="completed", run_seq=3)),
        _Exchange("POST", f"{BASE}/executions/{NEXT_RUN_ID}/cancel/", timeout=True),
        _status_get(_status(run_id=NEXT_RUN_ID, run_seq=8, since=40, goal=goal), since=40),
        _status_get(_status(run_id=NEXT_RUN_ID, status="cancelled", run_seq=9, since=40), since=40),
    ]
    async with open_client(exchanges) as (client, script):
        assert await _call(client.runs.cancel, RUN_ID) is None
        assert len(script.requests) == 1
        completed = await _call(client.runs.wait, RUN_ID, run_seq=3, timeout=5, poll_interval=0)
        assert completed.status == "completed" and completed.goal is None
        assert completed.poll_complete
        assert len(script.requests) == 3
        with pytest.raises(httpx.ReadTimeout):
            await _call(client.runs.cancel, NEXT_RUN_ID)
        assert len(script.requests) == 4
        cancelled = await _call(client.runs.wait, NEXT_RUN_ID, since=40, run_seq=8,
                                timeout=5, poll_interval=0)
        assert cancelled.status == "cancelled" and cancelled.run_seq == 9
        assert cancelled.poll_complete
        assert [row["method"] for row in script.requests] == ["POST", "GET", "GET", "POST", "GET", "GET"]


async def test_wait_and_events_drain_terminal_pages_and_fence_stale_observations(open_client):
    pages = [
        _status(status="completed", run_seq=7, since=10),
        _status(status="completed", run_seq=8, since=10, as_of_seq=99),
        _status(run_seq=8, since=10, step_ids=range(11, 31), has_more=True, as_of_seq=101),
        _status(status="completed", run_seq=8, since=30, step_ids=range(31, 51),
                has_more=True, as_of_seq=102),
        _status(status="completed", run_seq=8, since=50, step_ids=(51,), as_of_seq=103),
    ]
    for page in pages[3:]:
        page.update(step_count=41, last_step_id=51)
    queries = [10, 10, 10, 30, 50]
    exchanges = [_status_get(page, since=since) for page, since in zip(pages, queries)]
    exchanges += [_status_get(page, since=since) for page, since in zip(pages, queries)]
    cursor = RunPresentationCursor(version="presentation_cursor_v1", run_seq=8, as_of_seq=100)
    async with open_client(exchanges) as (client, script):
        result = await _call(client.runs.wait, RUN_ID, since=10, run_seq=8,
                             presentation_cursor=cursor, timeout=5, poll_interval=0)
        assert result.poll_complete and result.status == "completed"
        assert (result.next_since, result.run_seq, result.presentation_cursor_v1.as_of_seq) == (51, 8, 103)
        assert [step.id for step in result.new_steps] == [51]
        assert len(script.requests) == 5
        stream = client.runs.stream_events(
            RUN_ID, since=10, run_seq=8, presentation_cursor=cursor, timeout=5, poll_interval=0,
        )
        delivered = []
        if hasattr(stream, "__aiter__"):
            async for step in stream:
                assert isinstance(step, RunStep)
                delivered.append((step.id, step.observed_run_seq))
                assert len(delivered) <= 41
        else:
            for step in stream:
                assert isinstance(step, RunStep)
                delivered.append((step.id, step.observed_run_seq))
                assert len(delivered) <= 41
        assert delivered == [(step_id, 8) for step_id in range(11, 52)]
        assert [row["params"]["since"] for row in script.requests] == [str(n) for n in queries * 2]


def _execution_range(resource, content, offset, limit):
    end = min(len(content), offset + limit) - 1
    return _Exchange(
        "GET", f"{BASE}/executions/{RUN_ID}/resource/", status=206,
        params={"ref": resource["ref"]}, request_headers={"Range": f"bytes={offset}-{end}"},
        response_headers={
            "Content-Range": f"bytes {offset}-{end}/{len(content)}",
            "X-Execution-Resource-Ref": resource["ref"], "X-Content-SHA256": resource["sha256"],
        },
        raw=content[offset:end + 1],
    )


async def test_observation_ranges_and_history_are_explicit_and_fail_closed(open_client):
    content = b'{"payload":"' + b"x" * RANGE_BYTES + b'"}'
    error_content = b"Failure detail"
    resource = _resource(content, truncated=True)
    error = _resource(error_content, kind="error")
    cursor = "t" * 512
    transcript = {"messages": [{"role": "assistant", "content": "Recent message"}],
                  "has_more": True, "next_cursor": cursor, "page_size": 1,
                  "response_bytes_limit": PAGE_BYTES, "receipt": {}}
    older = {**transcript, "messages": [{"role": "user", "content": "Older message"}],
             "has_more": False, "next_cursor": None}
    exchanges = [
        _status_get(_status(status="failed", result=resource, error=error)),
        _execution_range(resource, content, 0, RANGE_BYTES),
        _execution_range(resource, content, RANGE_BYTES, RANGE_BYTES),
        _execution_range(error, error_content, 0, RANGE_BYTES),
        _Exchange("GET", f"{BASE}/executions/{RUN_ID}/transcript/", transcript,
                  params={"page_size": 1}),
        _Exchange("GET", f"{BASE}/executions/{RUN_ID}/transcript/", older,
                  params={"page_size": 1, "cursor": cursor}),
        _Exchange("GET", f"{BASE}/executions/{RUN_ID}/logs/",
                  {"logs": [{"id": 1, "type": "info", "message": "Projected log"}],
                   "has_more": True, "next_cursor": "next-log-page", "page_size": 1,
                   "response_bytes_limit": PAGE_BYTES, "receipt": {}}, params={"page_size": 1}),
        _Exchange("GET", f"{BASE}/executions/{RUN_ID}/status/", {"id": RUN_ID, "status": "completed"},
                  params={"view": "bounded_v1", "since": 0, "page_size": 20}),
        _Exchange("GET", f"{BASE}/executions/{RUN_ID}/status/", raw=b"x" * (PAGE_BYTES + 1),
                  params={"view": "bounded_v1", "since": 0, "page_size": 20}),
        _Exchange("GET", f"{BASE}/executions/{RUN_ID}/status/", raw=b"encoded response",
                  response_headers={"Content-Encoding": "gzip"},
                  params={"view": "bounded_v1", "since": 0, "page_size": 20}),
        _Exchange("GET", f"{BASE}/executions/{RUN_ID}/resource/",
                  {"detail": "Execution resource generation changed."}, status=409,
                  params={"ref": resource["ref"]}, request_headers={"Range": "bytes=0-0"}),
    ]
    async with open_client(exchanges) as (client, script):
        status = await _call(client.runs.status, RUN_ID)
        assert status.status == "failed" and status.poll_complete
        assert status.result.truncated and status.error.kind == "error"
        with pytest.raises(ValueError):
            _ = status.output
        with pytest.raises(ValueError):
            await _call(client.runs.read_resource, RUN_ID, status.result, limit=RANGE_BYTES + 1)
        assert len(script.requests) == 1
        first = await _call(client.runs.read_resource, RUN_ID, status.result, offset=0, limit=RANGE_BYTES)
        assert first == content[:RANGE_BYTES]
        assert len(script.requests) == 2
        tail = await _call(client.runs.read_resource, RUN_ID, status.result,
                           offset=RANGE_BYTES, limit=RANGE_BYTES)
        assert tail == content[RANGE_BYTES:]
        error_bytes = await _call(client.runs.read_resource, RUN_ID, status.error, limit=RANGE_BYTES)
        assert error_bytes == error_content
        recent = await _call(client.runs.transcript, RUN_ID, page_size=1)
        assert isinstance(recent, RunTranscriptPage)
        assert recent.next_cursor == cursor and recent.has_more
        assert len(script.requests) == 5
        history = await _call(client.runs.transcript, RUN_ID, page_size=1, cursor=recent.next_cursor)
        assert history.messages == older["messages"] and not history.has_more
        logs = await _call(client.runs.logs, RUN_ID, page_size=1)
        assert isinstance(logs, RunLogPage)
        assert logs.next_cursor == "next-log-page" and logs.has_more
        assert len(script.requests) == 7
        for expected_count in (8, 9, 10):
            with pytest.raises(RunObservationUnsupportedError):
                await _call(client.runs.wait, RUN_ID, timeout=5, poll_interval=0)
            assert len(script.requests) == expected_count
        assert script.streams[8].delivered == PAGE_BYTES + 1
        assert script.streams[9].delivered == 0
        with pytest.raises(FlyMyAIAgentError) as conflict:
            await _call(client.runs.read_resource, RUN_ID, status.result, limit=1)
        assert conflict.value.status_code == 409
        assert status.poll_complete and status.status == "failed"


def _file_metadata(content, *, version=7, artifact=ARTIFACT):
    return {
        "id": artifact, "name": "reports/result.bin", "version": version, "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(), "content_type": "application/octet-stream",
        "executable": False, "hidden": False, "deleted": False,
        "created_at": NOW, "updated_at": NOW, "origin": {},
    }


async def test_files_discovery_and_pages_preserve_exact_owner_scope(open_client):
    library = "ws_01ARZ3NDEKTSV4RRFFQ69G5FA0"
    cursor = "f" * 128
    first_meta = _file_metadata(b"first")
    second_meta = _file_metadata(b"second", artifact="af_01ARZ3NDEKTSV4RRFFQ69G5FAZ")
    first_page = {"workspace": WORKSPACE, "revision": 12, "total": 2,
                  "files": [first_meta], "next_cursor": cursor}
    second_page = {**first_page, "revision": 13, "files": [second_meta], "next_cursor": None}
    exchanges = [
        _Exchange("GET", f"{BASE}/files/workspaces",
                  {"workspaces": [{"workspace": library, "kind": "library", "revision": 1,
                                   "agent_uuid": None, "name": "Personal library", "role": "admin"}],
                   "next_cursor": library}, params={"limit": 1}),
        _Exchange("GET", f"{BASE}/files/workspaces",
                  {"workspaces": [{"workspace": WORKSPACE, "kind": "task", "revision": 12,
                                   "agent_uuid": GROUP_ID, "name": "Shared project", "role": "admin"}],
                   "next_cursor": None}, params={"limit": 1, "cursor": library}),
        _Exchange("GET", f"{BASE}/files/page",
                  {"workspace": library, "revision": 1, "total": 0, "files": [], "next_cursor": None},
                  params={"limit": 20}),
        _Exchange("GET", f"{BASE}/files/page",
                  {"workspace": PERSONAL, "revision": 2, "total": 0, "files": [], "next_cursor": None},
                  params={"limit": 20, "agent_uuid": AGENT_ID}),
        _Exchange("GET", f"{BASE}/files/page", first_page,
                  params={"limit": 1, "workspace": WORKSPACE, "prefix": "reports/"}),
        _Exchange("GET", f"{BASE}/files/page", second_page,
                  params={"limit": 1, "workspace": WORKSPACE, "prefix": "reports/", "cursor": cursor}),
        _Exchange("GET", f"{BASE}/files/workspace-subjects",
                  {"subjects": [{"kind": "task", "id": AGENT_ID, "name": "Owner agent"}],
                   "next_cursor": None}, params={"kind": "task", "limit": 20}),
        _Exchange("GET", f"{BASE}/files/workspace-subjects",
                  {"subjects": [{"kind": "group", "id": GROUP_ID, "name": "Owner group"}],
                   "next_cursor": GROUP_ID}, params={"kind": "group", "limit": 1}),
        _Exchange("GET", f"{BASE}/files/workspace-subjects", {"subjects": [], "next_cursor": None},
                  params={"kind": "group", "limit": 1, "cursor": GROUP_ID}),
    ]
    async with open_client(exchanges) as (client, script):
        workspaces = await _call(client.files.workspaces, limit=1)
        assert workspaces.next_cursor == library
        assert len(script.requests) == 1
        shared = await _call(client.files.workspaces, limit=1, cursor=workspaces.next_cursor)
        assert shared.workspaces[0].workspace == WORKSPACE
        assert shared.workspaces[0].role == "admin"
        default = await _call(client.files.list)
        personal = await _call(client.files.list, agent_uuid=AGENT_ID)
        assert default.workspace == library and personal.workspace == PERSONAL
        selected = await _call(client.files.list, workspace=WORKSPACE, prefix="reports/", limit=1)
        assert isinstance(selected, FilePage)
        assert selected.workspace == WORKSPACE and selected.next_cursor == cursor
        assert len(script.requests) == 5
        next_page = await _call(client.files.list, workspace=selected.workspace, prefix="reports/",
                                limit=1, cursor=selected.next_cursor)
        assert [item.id for item in next_page.files] == [second_meta["id"]]
        assert next_page.revision == 13 and next_page.next_cursor is None
        tasks = await _call(client.files.subjects, "task")
        groups = await _call(client.files.subjects, "group", limit=1)
        assert tasks.subjects[0].id == UUID(AGENT_ID)
        assert groups.subjects[0].id == UUID(GROUP_ID)
        assert groups.subjects[0].kind == "group"
        assert len(script.requests) == 8
        last = await _call(client.files.subjects, "group", limit=1, cursor=str(groups.next_cursor))
        assert last.subjects == [] and last.next_cursor is None
        for invalid in ({"workspace": WORKSPACE, "agent_uuid": AGENT_ID},
                        {"limit": 51}, {"cursor": "f" * 129}):
            with pytest.raises(ValueError):
                await _call(client.files.list, **invalid)
        assert len(script.requests) == 9


def _file_range(meta, content, offset, limit, *, response_ref=None):
    ref = f"{meta['id']}@v{meta['version']}"
    end = min(meta["size"], offset + limit) - 1
    return _Exchange(
        "GET", f"{BASE}/files/{ref}/content", status=206,
        params={"view": "bounded_v1", "workspace": WORKSPACE},
        request_headers={"Range": f"bytes={offset}-{end}"},
        response_headers={"Content-Range": f"bytes {offset}-{end}/{meta['size']}",
                          "X-Artifact-Ref": response_ref or ref, "X-Content-SHA256": meta["sha256"]},
        raw=content[offset:end + 1],
    )


async def test_files_pin_versions_bound_ranges_and_expose_conflicts(open_client):
    content = b"0123456789abcdef" * 4096 + b"end"
    metadata = _file_metadata(content)
    ref = f"{ARTIFACT}@v7"
    stale = {"code": "FILE_STALE_VERSION", "detail": "The selected version is stale."}
    exchanges = [
        _Exchange("GET", f"{BASE}/files/{ARTIFACT}/meta", metadata, params={"workspace": WORKSPACE}),
        _file_range(metadata, content, 1, RANGE_BYTES),
        _file_range(metadata, content, RANGE_BYTES + 1, RANGE_BYTES),
        _Exchange("GET", f"{BASE}/files/{ref}/meta", metadata, params={"workspace": WORKSPACE}),
        _file_range(metadata, content, 0, 3),
        _Exchange("GET", f"{BASE}/files/{ref}/content", stale, status=409,
                  params={"view": "bounded_v1", "workspace": WORKSPACE},
                  request_headers={"Range": "bytes=0-2"}),
        _file_range(metadata, content, 0, 3, response_ref=f"{ARTIFACT}@v8"),
        _Exchange("GET", f"{BASE}/files/page", {"tree": []}, params={"limit": 20, "workspace": WORKSPACE}),
    ]
    async with open_client(exchanges) as (client, script):
        meta = await _call(client.files.stat, ARTIFACT, workspace=WORKSPACE)
        assert isinstance(meta, FileMetadata)
        assert (meta.ref, meta.size, meta.version) == (ref, RANGE_BYTES + 3, 7)
        window = await _call(client.files.read, meta, offset=1, limit=RANGE_BYTES, workspace=WORKSPACE)
        assert isinstance(window, FileRange)
        assert window.ref == ref and window.offset == 1 and window.total_bytes == len(content)
        assert window.content == content[1:RANGE_BYTES + 1]
        assert len(script.requests) == 2
        tail = await _call(client.files.read, meta, offset=RANGE_BYTES + 1,
                           limit=RANGE_BYTES, workspace=WORKSPACE)
        assert tail.content == content[RANGE_BYTES + 1:]
        assert tail.ref == ref
        for offset, limit in ((0, RANGE_BYTES + 1), (meta.size, 1), (-1, 1)):
            with pytest.raises(ValueError):
                await _call(client.files.read, meta, offset=offset, limit=limit, workspace=WORKSPACE)
        with pytest.raises(ValueError):
            await _call(client.files.read, "7", offset=0, limit=1, workspace=WORKSPACE)
        assert len(script.requests) == 3
        pinned = await _call(client.files.read, ref, offset=0, limit=3, workspace=WORKSPACE)
        assert pinned.ref == ref and pinned.content == content[:3]
        assert len(script.requests) == 5
        with pytest.raises(FlyMyAIAgentError) as conflict:
            await _call(client.files.read, meta, offset=0, limit=3, workspace=WORKSPACE)
        assert conflict.value.status_code == 409
        assert conflict.value.response_body == stale
        assert len(script.requests) == 6
        with pytest.raises(FilesContractUnsupportedError):
            await _call(client.files.read, meta, offset=0, limit=3, workspace=WORKSPACE)
        assert meta.ref == ref and meta.version == 7
        with pytest.raises(FilesContractUnsupportedError):
            await _call(client.files.list, workspace=WORKSPACE)
