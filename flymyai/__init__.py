import httpx

from flymyai.core.client import FlyMyAI, AsyncFlyMyAI, FlyMyAIM1, AsyncFlymyAIM1
from flymyai.core.exceptions import FlyMyAIPredictException, FlyMyAIExceptionGroup
from flymyai.agents import (
    AgentClient,
    AsyncAgentClient,
    FlyMyAIAgentError,
    SchemaSuggestion,
    SuggestSchemaError,
    VariablesValidationError,
)

__all__ = [
    # Prediction clients
    "run",
    "httpx",
    "async_run",
    "FlyMyAI",
    "AsyncFlyMyAI",
    "FlyMyAIExceptionGroup",
    "FlyMyAIPredictException",
    # Agent clients
    "AgentClient",
    "AsyncAgentClient",
    "FlyMyAIAgentError",
    "VariablesValidationError",
    "SuggestSchemaError",
    "SchemaSuggestion",
]


client = FlyMyAI
async_client = AsyncFlyMyAI
run = client.run_predict
async_run = async_client.arun_predict

m1_client = FlyMyAIM1
async_m1_client = AsyncFlymyAIM1
