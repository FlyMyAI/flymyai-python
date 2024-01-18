import pytest
from flymyai.multipart import MultipartPayload

from .FixtureFactory import FixtureFactory

factory = FixtureFactory(__file__)


@pytest.fixture
def multiparts():
    return factory("multiparts")


def test_multipart_payload(multiparts):
    for payload in multiparts:
        files = MultipartPayload(payload[0]).serialize().get("files")
        assert bool(files) == payload[1]
