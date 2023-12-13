import pytest
from flymyai.multipart import MultipartPayload


@pytest.fixture
def multiparts():
    return [
        ({"i123": 123, "i456": 456}, False),
        (
            {
                "i123": 123,
                "i456": "/home/oleg/Downloads/Telegram Desktop/image_2023-12-07_23-42-17.png",
            },
            True,
        ),
        ({"i123": [4, 5, 6], "i456": {"test": "value"}}, False),
    ]


def test_multipart_payload(multiparts):
    for payload in multiparts:
        files = MultipartPayload(payload[0]).serialize().get("files")
        assert bool(files) == payload[1]
