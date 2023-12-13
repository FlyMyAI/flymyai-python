import dataclasses
import json

import httpx


@dataclasses.dataclass
class Base4xxResponse:
    status_code: int
    url: httpx.URL
    content: bytes

    requires_retry: bool = False

    def to_msg(self):
        return f"""
            BAD REQUEST DETECTED ({self.status_code}):
            REQUEST URL: {self.url};
        """

    @classmethod
    def from_response(cls, response: httpx.Response):
        return cls(response.status_code, response.url, response.content)


@dataclasses.dataclass
class FlyMyAI400Response(Base4xxResponse):
    requires_retry: bool = False

    def to_msg(self):
        return f"""
            Bad request happened: {self.content.decode()}
        """


@dataclasses.dataclass
class FlyMyAI401Response(Base4xxResponse):
    requires_retry: bool = False

    def to_msg(self):
        return f"""
            Authentication error: verify your credentials!
        """


@dataclasses.dataclass
class FlyMyAI422Response(Base4xxResponse):
    requires_retry: bool = False

    def to_msg(self):
        jsoned = json.loads(self.content)
        msg = super().to_msg()
        if detail := jsoned.get("detail"):
            msg += f"Details: {detail}"
        return msg


@dataclasses.dataclass
class PredictionResponse:
    exc_history: list | None

    output_data: dict


@dataclasses.dataclass
class OpenAPISchemaResponse:
    exc_history: list | None
    schema: dict
