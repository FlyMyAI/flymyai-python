import json
import os
import typing
from dataclasses import dataclass

import httpx


class FlyMyAIResponse(httpx.Response):
    is_event: bool = False

    @classmethod
    def from_httpx(cls, response: httpx.Response):
        return cls(
            status_code=response.status_code,
            content=response.content,
            request=response.request,
            headers=response.headers,
        )

    def json(self, **kwargs) -> typing.Any:
        content = self.content.lstrip()
        if content.startswith(b"data") or content.startswith(b"event"):
            # A body in SSE form ("data: {...}", maybe after "event: error"), as the
            # gateway sends its errors on the SSE routes: the JSON of the first data line.
            try:
                return _sse_data_json(content)
            except ValueError:
                pass
            # The bare form "data{...}" that older code paths produced.
            for field in (b"data", b"event"):
                if content.startswith(field):
                    return json.loads(content[len(field) :])
        return super().json(**kwargs)


def _sse_data_json(content: bytes) -> typing.Any:
    for raw in content.splitlines():
        line = raw.strip()
        if not line.startswith(b"data"):
            continue
        value = line[len(b"data") :]
        if value[:1] == b":":
            value = value[1:]
        value = value.strip()
        if value:
            return json.loads(value)
    raise ValueError("no data line")


@dataclass
class ChatResponseData:
    text: typing.Optional[str]
    tool_used: typing.Optional[str]
    file_url: typing.Optional[str]

    @classmethod
    def from_dict(
        cls, data: typing.Optional[dict]
    ) -> typing.Optional["ChatResponseData"]:
        if not data:
            return None
        return cls(
            text=data.get("text"),
            tool_used=data.get("tool_used"),
            file_url=(
                "".join(
                    [
                        os.getenv("FLYMYAI_M1_DSN", "https://api.chat.flymy.ai/"),
                        data.get("file_url"),
                    ]
                )
                if data.get("file_url")
                else None
            ),
        )


@dataclass
class FlyMyAIM1Response:
    success: bool
    error: typing.Optional[str]
    data: typing.Optional[ChatResponseData]

    @classmethod
    def from_httpx(cls, response):
        json_data = response.json()
        return cls(
            success=json_data.get("success", False),
            error=json_data.get("error"),
            data=ChatResponseData.from_dict(json_data.get("data")),
        )
