import json
import typing

import httpx


class FlyMyAIResponse(httpx.Response):
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
            trail_content = self.content[: len(b"data")]
            return json.loads(trail_content)
        else:
            return super().json(**kwargs)
