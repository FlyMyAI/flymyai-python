import httpx

from flymyai.core.client import FlyMyAI, AsyncFlyMyAI, FlyMyAIM1, AsyncFlymyAIM1
from flymyai.core.exceptions import FlyMyAIPredictException, FlyMyAIExceptionGroup
from flymyai.agents import (
    AgentGroup,
    AgentClient,
    AsyncAgentClient,
    ConnectionBinding,
    ExternalPrincipal,
    ExternalPrincipalStatus,
    FlyMyAIAgentError,
    IntegrationConnection,
    IntegrationConnectionStatus,
    McpAccessMode,
    McpResourceSet,
    McpResourceSetAuthorityType,
    McpResourceSetManagementMode,
    McpResourceSetMember,
    McpResourceSetMemberInput,
    McpResourceSetStatus,
    McpResourceSetSummary,
    McpResourceSetStaleRevisionError,
    McpResourceType,
    RuntimeConnections,
    SchemaSuggestion,
    SuggestSchemaError,
    VariablesValidationError,
    WorkspaceGrantStaleRevisionError,
    WorkspaceGrant,
    WorkspaceGrantMutation,
    WorkspaceGrantPage,
    WorkspaceGrantRole,
    WorkspaceGrantSubject,
    WorkspaceGrantSubjectKind,
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
    "ConnectionBinding",
    "ExternalPrincipal",
    "ExternalPrincipalStatus",
    "FlyMyAIAgentError",
    "IntegrationConnection",
    "IntegrationConnectionStatus",
    "McpAccessMode",
    "McpResourceSetManagementMode",
    "McpResourceSetAuthorityType",
    "McpResourceSetStaleRevisionError",
    "VariablesValidationError",
    "WorkspaceGrantStaleRevisionError",
    "SuggestSchemaError",
    "SchemaSuggestion",
    "AgentGroup",
    "McpResourceSet",
    "McpResourceSetMember",
    "McpResourceSetMemberInput",
    "McpResourceSetStatus",
    "McpResourceSetSummary",
    "McpResourceType",
    "RuntimeConnections",
    "WorkspaceGrant",
    "WorkspaceGrantMutation",
    "WorkspaceGrantPage",
    "WorkspaceGrantRole",
    "WorkspaceGrantSubject",
    "WorkspaceGrantSubjectKind",
]


client = FlyMyAI
async_client = AsyncFlyMyAI
run = client.run_predict
async_run = async_client.arun_predict

m1_client = FlyMyAIM1
async_m1_client = AsyncFlymyAIM1
