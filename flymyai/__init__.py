import httpx

from .core.authorizations import ClientInfoFactory
from .core.client import FlyMyAI, AsyncFlyMyAI
from .core.exceptions import FlyMyAIPredictException, FlyMyAIExceptionGroup


__all__ = [
    "run",
    "httpx",
    "async_run",
    "FlyMyAI",
    "AsyncFlyMyAI",
    "ClientInfoFactory",
    "FlyMyAIExceptionGroup",
    "FlyMyAIPredictException",
]


client = FlyMyAI
async_client = AsyncFlyMyAI
run = client.run_predict
async_run = async_client.arun_predict
