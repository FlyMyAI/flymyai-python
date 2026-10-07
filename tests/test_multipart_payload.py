"""Nested request fields reach the gateway intact (2026-10-06, Alexander's VLM carousels).

httpx sent a dict as its Python repr in a urlencoded body and refused it in a
multipart one, so ``response_schema`` never arrived as JSON; a list goes out as one
field per item, which is how the gateway reads list inputs.
"""

from __future__ import annotations

import io
import json
from urllib.parse import parse_qs

import httpx
import pytest

from flymyai.multipart import MultipartPayload

SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}}
IMAGES = ["https://example.com/1.png", "https://example.com/2.png"]


def _request(payload: dict) -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://api.example.com/predict",
        **MultipartPayload(payload).serialize()
    )


def _form(payload: dict) -> dict:
    return parse_qs(_request(payload).read().decode())


def _parts(payload: dict) -> list[tuple[str, bytes]]:
    request = _request(payload)
    body = request.read()
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    parts = []
    for chunk in body.split(b"--" + boundary)[1:-1]:
        head, _, value = chunk.strip(b"\r\n").partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        parts.append((name, value))
    return parts


def test_dict_goes_out_as_json_text_without_files():
    form = _form({"prompt": "count", "response_schema": SCHEMA, "image": IMAGES})

    assert json.loads(form["response_schema"][0]) == SCHEMA
    assert form["image"] == IMAGES
    assert form["prompt"] == ["count"]


def test_dict_and_list_go_out_with_a_file():
    parts = _parts({
        "video": io.BytesIO(b"\x00\x01"),
        "response_schema": SCHEMA,
        "media_url": IMAGES,
        "prompt": "count",
    })
    values = {}
    for name, value in parts:
        values.setdefault(name, []).append(value)

    assert json.loads(values["response_schema"][0]) == SCHEMA
    assert [item.decode() for item in values["media_url"]] == IMAGES
    assert values["video"] == [b"\x00\x01"]
    assert values["prompt"] == [b"count"]


def test_list_items_that_are_objects_are_json():
    form = _form({"messages": [{"role": "user", "content": "hi"}, ["a", 1]]})

    assert [json.loads(item) for item in form["messages"]] == [
        {"role": "user", "content": "hi"},
        ["a", 1],
    ]


def test_list_of_files_is_one_part_per_file():
    parts = _parts({"image": [b"first", b"second"], "prompt": "compare"})

    assert [value for name, value in parts if name == "image"] == [b"first", b"second"]


def test_files_mixed_with_values_are_refused():
    with pytest.raises(TypeError):
        MultipartPayload({"image": [b"bytes", "https://example.com/1.png"]}).serialize()


def test_primitives_are_unchanged():
    serialized = MultipartPayload(
        {"prompt": "hi", "max_tokens": 5, "temperature": 0.2, "stream": True}
    ).serialize()

    assert serialized == {
        "files": {},
        "data": {"prompt": "hi", "max_tokens": 5, "temperature": 0.2, "stream": True},
    }
