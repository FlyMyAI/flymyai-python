"""A user's own file to an H3 result, over offline HTTP fixtures only.

This proves what the SDK sends and returns, not that a backend or a model accepted it.
"""

import asyncio
import builtins
import inspect
import json
from contextlib import contextmanager

import httpx
import pytest
import respx

from flymyai.agents import (
    AgentClient,
    AsyncAgentClient,
    ModelProxyError,
    ModelProxyFile,
)
from flymyai.agents._client import _model_proxy_budget

ORIGIN = "https://sdk-files.invalid"
FILES = ORIGIN + "/api/v1/agents/files/"
PROXY = ORIGIN + "/api/v1/model-proxy/"
REVISION = "a" * 64
OPERATION = "aaaaaaaa-1234-4321-abcd-123456789012"
API_KEY = "synthetic-caller-key"
KEY = "photo-upload-1"
PNG = b"\x89PNG\r\n\x1a\n" + b"\0\xff\x80" * 4
SHA = "c" * 64


def artifact():
    return {
        "schema": "flymyai.artifact/v1",
        "id": "11111111-2222-4333-8444-555555555555",
        "version": {
            "id": "66666666-7777-4888-8999-000000000000",
            "revision": 1,
            "size": len(PNG),
            "sha256": SHA,
            "mime": "image/png",
        },
        "kind": "image",
        "name": "my photo.png",
        "workspace": None,
        "relations": {},
        "legacy": None,
    }


def stored(**changes):
    value = {
        "id": "af_01M6CT3V35WF1KZYEHX2V27G3J",
        "name": "my photo.png",
        "version": 1,
        "created": True,
        "bytes_written": len(PNG),
        "sha256": SHA,
        "verified": True,
        "artifact": artifact(),
    }
    value.update(changes)
    return value


class Body(httpx.AsyncByteStream):
    def __init__(self, raw):
        self.raw = raw
        self.closed = False

    async def __aiter__(self):
        yield self.raw

    async def aclose(self):
        self.closed = True


def answer(value, status=200, headers=None):
    raw = json.dumps(value, separators=(",", ":")).encode()
    return httpx.Response(status, stream=Body(raw), headers=headers)


def refusal(value, status):
    return answer(value, status, {"Content-Type": "application/json"})


@pytest.fixture(autouse=True)
def http():
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        yield router


@contextmanager
def proxy_client(mode, **options):
    cls = AgentClient if mode == "sync" else AsyncAgentClient
    client = cls(api_key=API_KEY, base_url=ORIGIN, **options)

    # Any accidental use of the legacy retry/request path is a test failure.
    def legacy_forbidden(*args, **kwargs):
        raise AssertionError("a file must not enter the legacy transport")

    client._request = legacy_forbidden

    def call(method, *args, **kwargs):
        result = getattr(client.model_proxy, method)(*args, **kwargs)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    try:
        yield client, call
    finally:
        client.close() if mode == "sync" else asyncio.run(client.close())


def form_of(request):
    """The multipart fields of a request, by name: (headers, bytes)."""
    boundary = request.headers["Content-Type"].split("boundary=")[1].encode()
    fields = {}
    for piece in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = piece[2:-2].partition(b"\r\n\r\n")
        lines = head.decode().split("\r\n")
        name = lines[0].split('name="')[1].split('"')[0]
        fields[name] = (lines, body)
    return fields


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("source", ["str-path", "path", "bytes", "stream"])
def test_upload_stores_on_the_file_route_and_returns_the_reference(
    mode, source, http, tmp_path
):
    route = http.post(FILES).mock(return_value=answer(stored(), status=201))
    local = tmp_path / "my photo.png"
    local.write_bytes(PNG)
    options = {"idempotency_key": KEY}
    if source == "str-path":
        file = str(local)
    elif source == "path":
        file = local
    elif source == "bytes":
        file = PNG
        options["filename"] = "my photo.png"
    else:
        file = local.open("rb")
    try:
        with proxy_client(mode) as (_, call):
            uploaded = call("upload_file", file, **options)
    finally:
        if source == "stream":
            file.close()
    assert isinstance(uploaded, ModelProxyFile)
    assert uploaded.artifact == artifact()
    assert uploaded.id == stored()["id"] and uploaded.created is True
    assert route.call_count == 1 and len(http.calls) == 1
    request = route.calls[0].request
    assert request.headers["X-API-KEY"] == API_KEY
    assert "Authorization" not in request.headers
    assert "Cookie" not in request.headers
    assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert int(request.headers["Content-Length"]) > len(PNG)
    fields = form_of(request)
    assert sorted(fields) == ["file", "idempotency_key"]
    assert fields["idempotency_key"][1] == KEY.encode()
    head, body = fields["file"]
    assert 'filename="my photo.png"' in head[0]
    assert head[1] == "Content-Type: image/png"
    assert body == PNG


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_upload_takes_the_callers_name_and_type(mode, http):
    route = http.post(FILES).mock(return_value=answer(stored(), status=201))
    with proxy_client(mode) as (_, call):
        call(
            "upload_file",
            b"RIFF",
            filename="voice.bin",
            media_type="audio/wav",
            idempotency_key=KEY,
        )
    head, body = form_of(route.calls[0].request)["file"]
    assert 'filename="voice.bin"' in head[0]
    assert head[1] == "Content-Type: audio/wav" and body == b"RIFF"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "name,expected",
    [("voice.wav", "audio/"), ("voice.mp3", "audio/"), ("photo.jpg", "image/jpeg")],
)
def test_upload_takes_the_type_from_the_filename(mode, name, expected, http):
    route = http.post(FILES).mock(return_value=answer(stored(), status=201))
    with proxy_client(mode) as (_, call):
        call("upload_file", b"x", filename=name, idempotency_key=KEY)
    head, _ = form_of(route.calls[0].request)["file"]
    assert head[1].startswith("Content-Type: " + expected)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "file,options",
    [
        (b"x", {"idempotency_key": KEY}),
        (b"x", {"idempotency_key": KEY, "filename": "  "}),
        (b"x", {"idempotency_key": KEY, "filename": "unknown.bin"}),
        (b"x", {"idempotency_key": KEY, "filename": "no-extension"}),
        (b"x", {"idempotency_key": " spaced ", "filename": "a.png"}),
        (b"x", {"idempotency_key": "", "filename": "a.png"}),
        (b"x", {"idempotency_key": "ключ", "filename": "a.png"}),
        (b"x", {"idempotency_key": "k" * 201, "filename": "a.png"}),
        (b"x", {"idempotency_key": 7, "filename": "a.png"}),
        (b"x", {"idempotency_key": KEY, "filename": "a.png", "timeout": 601}),
        (b"x", {"idempotency_key": KEY, "filename": "a.png", "timeout": True}),
    ],
)
def test_upload_refuses_before_http(mode, file, options, http):
    with proxy_client(mode) as (_, call):
        with pytest.raises(ValueError):
            call("upload_file", file, **options)
    assert len(http.calls) == 0


def test_the_longest_key_of_a_file_is_sent(http):
    route = http.post(FILES).mock(return_value=answer(stored(), status=201))
    with proxy_client("sync") as (_, call):
        call("upload_file", b"x", filename="a.png", idempotency_key="k" * 200)
    assert form_of(route.calls[0].request)["idempotency_key"][1] == b"k" * 200


def test_a_string_is_a_path_and_never_the_files_content(http):
    with proxy_client("sync") as (_, call):
        with pytest.raises(FileNotFoundError):
            call("upload_file", "not a real path.png", idempotency_key=KEY)
    assert len(http.calls) == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status", [201, 413, 500])
def test_a_path_is_closed_after_every_answer(mode, status, http, tmp_path, monkeypatch):
    opened = []
    real_open = builtins.open

    def tracked(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    local = tmp_path / "a.png"
    local.write_bytes(PNG)
    http.post(FILES).mock(return_value=answer(stored(), status=status))
    with proxy_client(mode) as (_, call):
        monkeypatch.setattr(builtins, "open", tracked)
        if status == 201:
            call("upload_file", local, idempotency_key=KEY)
        else:
            with pytest.raises(ModelProxyError) as raised:
                call("upload_file", local, idempotency_key=KEY)
            assert raised.value.status_code == status
        monkeypatch.setattr(builtins, "open", real_open)
    assert opened and all(handle.closed for handle in opened)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_repeated_upload_answers_the_same_reference(mode, http):
    route = http.post(FILES).mock(
        side_effect=[
            answer(stored(), status=201),
            answer(stored(created=False, bytes_written=0, idempotent_replay=True)),
        ]
    )
    with proxy_client(mode) as (_, call):
        first = call("upload_file", PNG, filename="a.png", idempotency_key=KEY)
        again = call("upload_file", PNG, filename="a.png", idempotency_key=KEY)
    assert again.artifact == first.artifact and again.created is False
    assert route.call_count == 2
    assert (
        form_of(route.calls[0].request)["idempotency_key"]
        == form_of(route.calls[1].request)["idempotency_key"]
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("case", ["null", "absent"])
def test_a_file_with_no_reference_says_none(mode, case, http):
    value = stored(artifact=None)
    if case == "absent":
        del value["artifact"]
    http.post(FILES).mock(return_value=answer(value, status=201))
    with proxy_client(mode) as (_, call):
        uploaded = call("upload_file", PNG, filename="a.png", idempotency_key=KEY)
    assert uploaded.artifact is None and uploaded.id == value["id"]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "status,value,code",
    [
        (
            413,
            {
                "error": {
                    "name": "FileError",
                    "code": "FILE_TOO_LARGE",
                    "message": "synthetic-private-error-body",
                }
            },
            "FILE_TOO_LARGE",
        ),
        (
            400,
            {
                "error": {
                    "code": "FILE_NAME_INVALID",
                    "data": {"name": "synthetic-private-error-body"},
                }
            },
            "FILE_NAME_INVALID",
        ),
        (403, {"code": "h3_team_unavailable"}, "h3_team_unavailable"),
    ],
)
def test_a_refused_file_is_named(mode, status, value, code, http):
    route = http.post(FILES).mock(return_value=refusal(value, status))
    with proxy_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("upload_file", PNG, filename="a.png", idempotency_key=KEY)
    error = raised.value
    assert error.code == "http_status" and error.status_code == status
    assert error.refusal_code == code and error.outcome_unknown is False
    assert error.idempotency_key == KEY
    assert route.call_count == 1 and code in str(error)
    assert "synthetic-private-error-body" not in repr(error) + str(error)
    assert KEY not in repr(error) + str(error)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "lost",
    [
        answer({"error": {"code": "FILE_VERIFY_FAILED"}}, 500),
        answer(stored(), 202),
        answer({}, 201),
        answer(stored(sha256="not-a-digest"), 201),
        answer(stored(version="1"), 201),
        answer(stored(artifact="text"), 201),
        httpx.Response(201, stream=Body(b"not json")),
        refusal({"detail": "no fixed code"}, 413),
        refusal({"error": "text"}, 400),
        refusal({"error": None}, 400),
        refusal({"error": {"code": "Has Spaces"}}, 400),
        httpx.ReadTimeout("private transport data"),
        httpx.ConnectError("private transport data"),
    ],
)
def test_a_lost_upload_answer_keeps_the_key_and_never_resends(mode, lost, http):
    route = http.post(FILES).mock(side_effect=[lost])
    with proxy_client(mode) as (_, call):
        with pytest.raises(ModelProxyError) as raised:
            call("upload_file", PNG, filename="a.png", idempotency_key=KEY)
    error = raised.value
    assert error.outcome_unknown is True and error.idempotency_key == KEY
    assert error.refusal_code is None
    assert route.call_count == 1
    assert "private transport data" not in repr(error) + str(error)


def test_only_a_file_gets_the_longer_budget():
    with AgentClient(api_key=API_KEY, base_url=ORIGIN) as client:
        assert _model_proxy_budget(client, 120.0) == 10.0
        assert _model_proxy_budget(client, 120.0, 600.0) == 60.0, "the client's timeout"
    with AgentClient(api_key=API_KEY, base_url=ORIGIN, timeout=900.0) as client:
        assert _model_proxy_budget(client, 120.0, 600.0) == 120.0


def test_a_changed_caller_gets_no_upload(http):
    with proxy_client("sync") as (client, call):
        client._api_key = "different-caller"
        with pytest.raises(ModelProxyError) as raised:
            call("upload_file", PNG, filename="a.png", idempotency_key=KEY)
    assert raised.value.code == "client_identity_changed"
    assert raised.value.outcome_unknown is False
    assert len(http.calls) == 0


def operation(**changes):
    value = {
        "operation_id": OPERATION,
        "prediction_id": OPERATION,
        "contract_revision": REVISION,
        "status": "queued",
        "stage": "submit",
        "cancel_requested": False,
    }
    value.update(changes)
    return value


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_local_file_becomes_an_h3_result(mode, http, tmp_path):
    video = "https://storage.invalid/proxy-operations/result.mp4?signature=fresh"
    http.post(FILES).mock(return_value=answer(stored(), status=201))
    http.get(PROXY + "catalog/").mock(
        return_value=answer({
            "release": OPERATION,
            "items": [{
                "contract_revision": REVISION,
                "route_key": "broker/h3",
                "model_version": "digest",
                "operation": "generate",
                "profile": "broker.h3-ref2va.v1",
            }],
            "next_cursor": None,
        })
    )
    admission = http.post(PROXY + "operations/").mock(
        return_value=answer({"operation": operation(), "created": True}, status=202)
    )
    read = http.get(PROXY + "operations/" + OPERATION + "/").mock(
        side_effect=[
            answer({"operation": operation(error={"code": "h3_capacity_not_ready"})}),
            answer({"operation": operation(status="running", stage="poll")}),
            answer({
                "operation": operation(
                    status="succeeded",
                    stage="files",
                    output={
                        "native": {"id": "job-1", "status": "completed"},
                        "video": {"url": video},
                    },
                )
            }),
        ]
    )
    local = tmp_path / "my photo.png"
    local.write_bytes(PNG)
    seen = []
    with proxy_client(mode) as (_, call):
        page = call("catalog", limit=2, endpoint_id="flymyai/h3")
        revision = page.items[0].contract_revision
        uploaded = call("upload_file", local, idempotency_key=KEY)
        inputs = {
            "prompt": "the photo comes alive",
            "conditions": [{"type": "image", "artifact": uploaded.artifact}],
            "duration_seconds": 5,
            "aspect_ratio": "9:16",
            "seed": 42,
        }
        created = call("create", revision, inputs, idempotency_key="video-1")
        result = call(
            "wait",
            created.operation.operation_id,
            contract_revision=revision,
            timeout=5,
            poll_interval=0.05,
            on_update=seen.append,
        )
    assert result.status == "succeeded" and result.output["video"]["url"] == video
    assert admission.call_count == 1, "never submitted again"
    sent = json.loads(admission.calls[0].request.content)
    assert sent == {"contract_revision": REVISION, "inputs": inputs}
    assert sent["inputs"]["conditions"][0]["artifact"] == artifact()
    assert admission.calls[0].request.headers["Idempotency-Key"] == "video-1"
    assert read.call_count == 3
    assert seen[0].waiting_for_capacity is True
    assert "about 15 minutes" in seen[0].reason
    assert [item.status for item in seen] == ["queued", "running", "succeeded"]
