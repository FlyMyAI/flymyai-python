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
        if self.content.startswith(b"data"):
            return json.loads(self.content.removeprefix(b"data"))
        elif self.content.startswith(b"event"):
            return json.loads(self.content.removeprefix(b"event"))
        else:
            return super().json(**kwargs)


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
