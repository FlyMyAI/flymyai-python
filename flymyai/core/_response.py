import json
import typing

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
