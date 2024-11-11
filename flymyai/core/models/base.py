import dataclasses

import httpx


@dataclasses.dataclass
class ResponseLike:
    status_code: int
    url: httpx.URL
    content: bytes

    def to_msg(self):
        return f"""
            BAD REQUEST DETECTED ({self.status_code}):
            REQUEST URL: {self.url};
        """
