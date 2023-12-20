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
