"""A file over 8 MB goes to storage before the prediction request.

2026-11-24: a 126 MB video for the Video Upscaler (it takes 500 MB) was refused with
413 by the api.flymy.ai ingress, which takes at most 100 MB per request. The SDK now
uploads such a file through the model's upload_url (FlyMyAI/gateway#306) and sends
the link its input takes; without that route a file of up to 100 MB stays inline.
"""

import io
import pathlib
from urllib.parse import parse_qs

import httpx
import pytest

from flymyai import async_client, client as sync_client
from flymyai.multipart.large_files import (
    INLINE_FILE_MAX_BYTES,
    LargeFileError,
    aupload_large_files,
    upload_large_files,
)

MB = 1024 * 1024
PUT = "https://storage.googleapis.com/prediction-result-prod/prediction/1935/7/a1?X-Goog-Signature=put"
GET = "https://storage.googleapis.com/prediction-result-prod/prediction/1935/7/a1?X-Goog-Signature=get"
UPLOAD_URL = "/api/v1/flymyai/bytedance-video-upscaler/upload_url"
KEY = {"X-API-KEY": "fixture-key"}


class Gateway:
    """The gateway's upload_url and storage's PUT, recorded."""

    def __init__(self, status=200, body=None, put_status=200):
        self.status = status
        self.body = (
            body
            if body is not None
            else {"url": PUT, "method": "PUT", "file_input": None, "download_url": GET}
        )
        self.put_status = put_status
        self.api = []
        self.puts = []

    def api_handler(self, request):
        self.api.append(request)
        return httpx.Response(self.status, json=self.body)

    def storage_handler(self, request):
        self.puts.append((request, request.read()))
        return httpx.Response(self.put_status)

    def clients(self):
        return (
            httpx.Client(
                base_url="https://api.test",
                transport=httpx.MockTransport(self.api_handler),
            ),
            httpx.Client(transport=httpx.MockTransport(self.storage_handler)),
        )

    def async_clients(self):
        return (
            httpx.AsyncClient(
                base_url="https://api.test",
                transport=httpx.MockTransport(self.api_handler),
            ),
            httpx.AsyncClient(transport=httpx.MockTransport(self.storage_handler)),
        )


def big(size=132487294):
    return b"\0" * size


def form_field(body: bytes, name: str):
    """A field of the submitted form: with no file left the SDK sends it urlencoded."""
    return parse_qs(body.decode())[name]


def test_a_large_file_goes_to_storage_and_the_request_carries_its_link(tmp_path):
    gateway = Gateway()
    api, storage = gateway.clients()
    video = tmp_path / "clip.mp4"
    video.write_bytes(big(9 * MB))

    payload = upload_large_files(
        {"video": video, "resolution": "4k"}, api, UPLOAD_URL, KEY, storage
    )

    assert payload == {"video": GET, "resolution": "4k"}
    assert [r.url.path for r in gateway.api] == [UPLOAD_URL]
    assert gateway.api[0].headers["X-API-KEY"] == "fixture-key"
    request, body = gateway.puts[0]
    assert str(request.url) == PUT and request.method == "PUT"
    # the key goes to the gateway only: storage gets the signed link and the bytes
    assert "X-API-KEY" not in request.headers
    assert request.headers["Content-Type"] == "video/mp4"
    assert request.headers["Content-Length"] == str(9 * MB)
    assert body == video.read_bytes()


def test_small_files_and_other_values_stay_as_they_are():
    gateway = Gateway()
    api, storage = gateway.clients()
    payload = {"image": b"\x89PNG" * 10, "prompt": "a cat", "seed": 7}

    assert upload_large_files(payload, api, UPLOAD_URL, KEY, storage) is payload
    assert upload_large_files(
        {"image": big(INLINE_FILE_MAX_BYTES)}, api, UPLOAD_URL, KEY, storage
    ) == {"image": big(INLINE_FILE_MAX_BYTES)}
    assert gateway.api == [] and gateway.puts == []


def test_a_list_with_a_large_file_goes_up_whole():
    gateway = Gateway()
    api, storage = gateway.clients()

    payload = upload_large_files(
        {"videos": [big(9 * MB), b"small"]}, api, UPLOAD_URL, KEY, storage
    )

    assert payload == {"videos": [GET, GET]}
    assert len(gateway.puts) == 2


def test_a_stream_is_sent_from_its_start_and_rewound_when_kept():
    gateway = Gateway()
    api, storage = gateway.clients()
    stream = io.BytesIO(big(9 * MB))
    stream.seek(100)

    upload_large_files({"video": stream}, api, UPLOAD_URL, KEY, storage)
    assert len(gateway.puts[0][1]) == 9 * MB

    refused = Gateway(
        status=403,
        body={
            "detail": "Failed to generate the url: No File URL field found on model!"
        },
    )
    api, storage = refused.clients()
    stream.seek(500)
    kept = upload_large_files({"video": stream}, api, UPLOAD_URL, KEY, storage)
    assert kept["video"] is stream and stream.tell() == 0


@pytest.mark.parametrize(
    "gateway",
    [
        Gateway(
            status=403,
            body={
                "detail": "Failed to generate the url: No File URL field found on model!"
            },
        ),
        # a gateway before download_url: only a file link field's PUT link
        Gateway(body={"url": PUT, "method": "PUT", "file_input": None}),
        Gateway(put_status=503),
    ],
)
def test_without_the_upload_a_file_of_up_to_100_mb_stays_in_the_request(gateway):
    api, storage = gateway.clients()
    data = big(60 * MB)

    assert upload_large_files({"video": data}, api, UPLOAD_URL, KEY, storage) == {
        "video": data
    }


def test_without_the_upload_a_file_over_100_mb_is_refused_before_sending():
    gateway = Gateway(
        status=403,
        body={
            "detail": "Failed to generate the url: No File URL field found on model!"
        },
    )
    api, storage = gateway.clients()

    with pytest.raises(
        LargeFileError, match="larger than the 100 MB a request carries"
    ):
        upload_large_files({"video": big()}, api, UPLOAD_URL, KEY, storage)


@pytest.mark.asyncio
async def test_async_uploads_a_file_by_chunks(tmp_path):
    gateway = Gateway()
    api, storage = gateway.async_clients()
    video = tmp_path / "clip.mov"
    video.write_bytes(big(9 * MB + 5))

    payload = await aupload_large_files({"video": video}, api, UPLOAD_URL, KEY, storage)

    assert payload == {"video": GET}
    request, body = gateway.puts[0]
    assert "X-API-KEY" not in request.headers
    assert request.headers["Content-Type"] == "video/quicktime"
    assert body == video.read_bytes()


@pytest.mark.asyncio
async def test_async_refuses_a_file_over_100_mb_without_the_upload():
    gateway = Gateway(status=404, body={})
    api, storage = gateway.async_clients()

    with pytest.raises(LargeFileError):
        await aupload_large_files({"video": big()}, api, UPLOAD_URL, KEY, storage)


def test_the_sync_client_submits_the_link(monkeypatch):
    seen = {}

    def fake_upload(payload, client, upload_url, headers):
        seen["upload_url"] = str(upload_url)
        seen["headers"] = headers
        return {**payload, "video": GET}

    def handle(request):
        seen["form"] = request.read()
        return httpx.Response(202, json={"prediction_id": "p-1"})

    monkeypatch.setattr(
        "flymyai.core.clients.SyncClient.upload_large_files", fake_upload
    )
    with sync_client(
        "fixture-key", model="flymyai/bytedance-video-upscaler", max_retries=1
    ) as sdk:
        sdk._client.close()
        sdk._client = httpx.Client(
            base_url="https://api.test", transport=httpx.MockTransport(handle)
        )
        task = sdk.predict_async_task({"video": big(9 * MB), "resolution": "4k"})

    assert task.prediction_id == "p-1"
    assert seen["upload_url"] == UPLOAD_URL
    assert seen["headers"] == KEY
    assert form_field(seen["form"], "video") == [GET]
    assert form_field(seen["form"], "resolution") == ["4k"]


@pytest.mark.asyncio
async def test_the_async_client_submits_the_link(monkeypatch):
    seen = {}

    async def fake_upload(payload, client, upload_url, headers):
        seen["upload_url"] = str(upload_url)
        return {**payload, "video": GET}

    def handle(request):
        seen["form"] = request.read()
        return httpx.Response(202, json={"prediction_id": "p-2"})

    monkeypatch.setattr(
        "flymyai.core.clients.AsyncClient.aupload_large_files", fake_upload
    )
    async with async_client(
        "fixture-key", model="flymyai/bytedance-video-upscaler", max_retries=1
    ) as sdk:
        await sdk._client.aclose()
        sdk._client = httpx.AsyncClient(
            base_url="https://api.test", transport=httpx.MockTransport(handle)
        )
        task = await sdk.predict_async_task({"video": pathlib.Path(__file__)})

    assert task.prediction_id == "p-2"
    assert seen["upload_url"] == UPLOAD_URL
    assert form_field(seen["form"], "video") == [GET]
