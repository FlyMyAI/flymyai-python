import httpx

from flymyai.core.client import FlyMyAI, AsyncFlyMyAI
from flymyai.core.exceptions import FlyMyAIPredictException, FlyMyAIExceptionGroup


__all__ = [
    "run",
    "httpx",
    "async_run",
    "FlyMyAI",
    "AsyncFlyMyAI",
    "FlyMyAIExceptionGroup",
    "FlyMyAIPredictException",
]


client = FlyMyAI
async_client = AsyncFlyMyAI
run = client.run_predict
async_run = async_client.arun_predict
