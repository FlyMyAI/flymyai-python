from flymyai.agents._client import (
    AsyncAgentClient,
    FlyMyAIAgentError,
    SuggestSchemaError,
    SyncAgentClient,
    VariablesValidationError,
)
from flymyai.agents._types import (
    Agent,
    AgentDetail,
    AgentStatus,
    AvailableTool,
    Compilation,
    CompilationStatus,
    ConfigurationStep,
    ExecutionLog,
    ExecutionLogType,
    ExecutionStatus,
    Run,
    RunDetail,
    SchemaSuggestion,
    Tool,
)

# Convenience alias — docs use `AgentClient`
AgentClient = SyncAgentClient

__all__ = [
    # Clients
    "AgentClient",
    "SyncAgentClient",
    "AsyncAgentClient",
    # Exceptions
    "FlyMyAIAgentError",
    "VariablesValidationError",
    "SuggestSchemaError",
    # Models
    "Agent",
    "AgentDetail",
    "AgentStatus",
    "AvailableTool",
    "Compilation",
    "CompilationStatus",
    "ConfigurationStep",
    "ExecutionLog",
    "ExecutionLogType",
    "ExecutionStatus",
    "Run",
    "RunDetail",
    "SchemaSuggestion",
    "Tool",
]
