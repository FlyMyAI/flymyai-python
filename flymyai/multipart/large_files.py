"""Large files go to storage before the prediction request.

A request carries at most 100 MB: the api.flymy.ai ingress answers 413 above it,
before the gateway sees the request. Some models take larger files, videos of up to
500 MB (2026-11-24: a 126 MB video for the Video Upscaler was refused that way). A file
over 8 MB is uploaded through the model's ``upload_url`` first (FlyMyAI/gateway#306)
and the request carries the link its input takes (``download_url``).

A gateway without that route, or a model it refuses (403), keeps a file of up to
100 MB in the request, as before; a larger one raises ``LargeFileError`` before
anything is sent. The API key goes only to the gateway: the storage link carries its
own signature, and the upload is sent by a client without the key.
"""

from __future__ import annotations

import io
import mimetypes
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import httpx

INLINE_FILE_MAX_BYTES = 8 * 1024 * 1024
REQUEST_MAX_BYTES = 100 * 1024 * 1024
_BINARY_TYPES = (pathlib.Path, io.BufferedIOBase, bytes)
_CHUNK = 1024 * 1024
_UPLOAD_TIMEOUT = httpx.Timeout(60.0, write=None, read=600.0)


class LargeFileError(ValueError):
    """A file of the input that cannot be sent: over 100 MB, and the model's
    direct upload is not available or failed."""


class _Unavailable(Exception):
    """The gateway gives no direct upload for this model."""


def _size(value: Any) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, pathlib.Path):
        return value.stat().st_size
    position = value.tell()
    value.seek(0, io.SEEK_END)
    size = value.tell()
    value.seek(position)
    return size


def _name(value: Any) -> str:
    if isinstance(value, pathlib.Path):
        return value.name
    return pathlib.Path(str(getattr(value, "name", "") or "")).name or "file"


def _content_type(value: Any) -> str:
    guessed, _encoding = mimetypes.guess_type(_name(value), strict=False)
    return guessed or "application/octet-stream"


def plan(payload: dict) -> Dict[str, List[Tuple[Optional[int], Any]]]:
    """The fields to upload: a file over 8 MB, or every file of a list that holds one
    (a list goes either all as links or all as files: the form takes no mix)."""
    fields: Dict[str, List[Tuple[Optional[int], Any]]] = {}
    for key, value in payload.items():
        if isinstance(value, _BINARY_TYPES):
            if _size(value) > INLINE_FILE_MAX_BYTES:
                fields[key] = [(None, value)]
        elif isinstance(value, (list, tuple)) and value:
            if all(isinstance(item, _BINARY_TYPES) for item in value) and any(
                _size(item) > INLINE_FILE_MAX_BYTES for item in value
            ):
                fields[key] = list(enumerate(value))
    return fields


def _links(response: httpx.Response) -> Tuple[str, str]:
    """The PUT link and the reading link of one upload_url answer."""
    if response.status_code in (403, 404, 405):
        raise _Unavailable()
    response.raise_for_status()
    body = response.json()
    put = body.get("url") if isinstance(body, dict) else None
    download = body.get("download_url") if isinstance(body, dict) else None
    # a gateway before download_url signs only a file link field's PUT link
    if (
        not isinstance(put, str)
        or not isinstance(download, str)
        or not put.startswith("https://")
        or not download.startswith("https://")
        or body.get("method") != "PUT"
    ):
        raise _Unavailable()
    return put, download


def _headers(value: Any) -> Dict[str, str]:
    return {"Content-Type": _content_type(value), "Content-Length": str(_size(value))}


def _keep_or_refuse(key: str, items: List[Tuple[Optional[int], Any]], why: str):
    """A field that could not go up: kept in the request when it fits, else refused."""
    for _index, value in items:
        if isinstance(value, io.BufferedIOBase):
            value.seek(0)
    if sum(_size(value) for _index, value in items) <= REQUEST_MAX_BYTES:
        return None
    largest = max((value for _index, value in items), key=_size)
    raise LargeFileError(
        f"{_name(largest)} ({key}) is larger than the 100 MB a request carries, and "
        f"it could not be uploaded first: {why}"
    )


def _apply(payload: dict, uploaded: Dict[str, List[Tuple[Optional[int], str]]]) -> dict:
    out = dict(payload)
    for key, links in uploaded.items():
        if len(links) == 1 and links[0][0] is None:
            out[key] = links[0][1]
        else:
            out[key] = [
                link for _index, link in sorted(links, key=lambda pair: pair[0])
            ]
    return out


def upload_large_files(
    payload: dict,
    client: httpx.Client,
    upload_url: Any,
    headers: Optional[Dict[str, str]] = None,
    storage: Optional[httpx.Client] = None,
) -> dict:
    """The payload with its large files in storage, for a sync client. ``client`` is
    the SDK's own (it carries the key); ``storage`` sends the bytes without it."""
    fields = plan(payload)
    if not fields:
        return payload
    own = storage is None
    storage = storage or httpx.Client(timeout=_UPLOAD_TIMEOUT)
    uploaded: Dict[str, List[Tuple[Optional[int], str]]] = {}
    try:
        for key, items in fields.items():
            links: List[Tuple[Optional[int], str]] = []
            try:
                for index, value in items:
                    put, download = _links(client.get(upload_url, headers=headers))
                    if isinstance(value, io.BufferedIOBase):
                        value.seek(0)
                    if isinstance(value, pathlib.Path):
                        with value.open("rb") as handle:
                            answer = storage.put(
                                put, content=handle, headers=_headers(value)
                            )
                    else:
                        answer = storage.put(
                            put, content=value, headers=_headers(value)
                        )
                    answer.raise_for_status()
                    links.append((index, download))
            except _Unavailable:
                _keep_or_refuse(
                    key, items, "this model or gateway has no direct upload"
                )
                continue
            except httpx.HTTPError as exc:
                _keep_or_refuse(key, items, f"{type(exc).__name__}: {exc}")
                continue
            uploaded[key] = links
    finally:
        if own:
            storage.close()
    return _apply(payload, uploaded)


async def _chunks(path: pathlib.Path):
    import anyio

    with path.open("rb") as handle:
        while True:
            chunk = await anyio.to_thread.run_sync(handle.read, _CHUNK)
            if not chunk:
                return
            yield chunk


async def aupload_large_files(
    payload: dict,
    client: httpx.AsyncClient,
    upload_url: Any,
    headers: Optional[Dict[str, str]] = None,
    storage: Optional[httpx.AsyncClient] = None,
) -> dict:
    """upload_large_files for an async client."""
    fields = plan(payload)
    if not fields:
        return payload
    own = storage is None
    storage = storage or httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT)
    uploaded: Dict[str, List[Tuple[Optional[int], str]]] = {}
    try:
        for key, items in fields.items():
            links: List[Tuple[Optional[int], str]] = []
            try:
                for index, value in items:
                    put, download = _links(
                        await client.get(upload_url, headers=headers)
                    )
                    if isinstance(value, pathlib.Path):
                        content: Any = _chunks(value)
                    elif isinstance(value, io.BufferedIOBase):
                        value.seek(0)
                        content = value.read()
                    else:
                        content = value
                    answer = await storage.put(
                        put, content=content, headers=_headers(value)
                    )
                    answer.raise_for_status()
                    links.append((index, download))
            except _Unavailable:
                _keep_or_refuse(
                    key, items, "this model or gateway has no direct upload"
                )
                continue
            except httpx.HTTPError as exc:
                _keep_or_refuse(key, items, f"{type(exc).__name__}: {exc}")
                continue
            uploaded[key] = links
    finally:
        if own:
            await storage.aclose()
    return _apply(payload, uploaded)
