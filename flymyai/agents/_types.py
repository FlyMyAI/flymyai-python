from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    DRAFT = "draft"
    INITIALIZATION_REQUIRED = "initialization_required"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CompilationStatus(str, Enum):
    PENDING = "pending"
    COMPILING = "compiling"
    COMPILED = "compiled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionLogType(str, Enum):
    DECLARED_FUNCTIONS = "declared_functions"
    TOOL_CALLED = "tool_called"
    TOOL_CALL_EXCEPTION = "tool_call_exception"
    TASK_CANCELLED = "task_cancelled"


# ── Agent (backend: UserAgentTask) ───────────────────────────────────────────


class Agent(BaseModel):
    """An agent task — the top-level configuration for an autonomous agent."""

    uuid: str
    name: str
    user_prompt: str
    available_tools: Any = Field(default_factory=list)
    available_custom_mcp_servers: List[int] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    all_tools_configured: bool = False
    tools_need_to_configure: List[int] = Field(default_factory=list)
    generated_pipeline: Dict[str, Any] = Field(default_factory=dict)
    status: AgentStatus = AgentStatus.DRAFT
    cron_schedule: str = ""
    webhook_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @property
    def id(self) -> str:
        return self.uuid

    @property
    def goal(self) -> str:
        return self.user_prompt


class AgentDetail(Agent):
    """Agent with nested tool objects instead of IDs."""

    available_tools: List[Dict[str, Any]] = Field(default_factory=list)


# ── Execution / Run (backend: UserAgentExecution) ───────────────────────────


class ExecutionLog(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    type: ExecutionLogType
    message: str
    data: Any = Field(default_factory=dict)


class Run(BaseModel):
    """A single agent execution (run)."""

    id: int
    user_agent_task: int
    previous_execution: Optional[int] = None
    original_prompt: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    run_seq: int = 0
    error: Optional[str] = None
    agent_result: Optional[Dict[str, Any]] = None

    @property
    def output(self) -> Optional[Dict[str, Any]]:
        return self.agent_result

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        )


class RunDetail(Run):
    """Run with execution logs attached."""

    logs: List[ExecutionLog] = Field(default_factory=list)
    user_agent_task_uuid: Optional[str] = None


# ── Tool (backend: UserMcpTool) ─────────────────────────────────────────────


class ConfigurationStep(BaseModel):
    description: str
    step_type: str
    vars_from_user_schema: Optional[Any] = None
    configuration_schema: Optional[Any] = None
    config: Optional[Any] = None
    execution_command: Optional[str] = None


class Tool(BaseModel):
    """A configured MCP tool belonging to the user."""

    id: int
    mcp_tool: str
    user_config: Dict[str, Any] = Field(default_factory=dict)
    is_configured: bool = False
    is_active: bool = True
    unsafe_methods: List[str] = Field(default_factory=list)
    required_configuration_steps: List[ConfigurationStep] = Field(default_factory=list)
    finished_configuration_steps: List[Dict[str, Any]] = Field(default_factory=list)
    next_configuration_step: Optional[Dict[str, Any]] = None
    redirect_url: str = ""
    response: str = ""
    created_at: datetime
    updated_at: datetime

    @property
    def name(self) -> str:
        return self.mcp_tool


class AvailableTool(BaseModel):
    """An MCP tool from the catalog that can be added to a user's account."""

    name: str
    type: str
    title: str = ""
    description: str = ""
    detail: str = ""
    href: str = ""
    categories: List[str] = Field(default_factory=list)
    instruction: Optional[str] = None
    custom_class: Optional[str] = None
    github_link: Optional[str] = None
    configuration_steps: List[Dict[str, Any]] = Field(default_factory=list)


# ── Script Compilation ──────────────────────────────────────────────────────


class Compilation(BaseModel):
    """A compiled/frozen script or instruction derived from an agent execution."""

    id: int
    execution: int
    status: CompilationStatus = CompilationStatus.PENDING
    script_code: str = ""
    instruction_md: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @property
    def is_ready(self) -> bool:
        """True once the compilation/instruction can be executed."""
        return self.status in (
            CompilationStatus.COMPILED,
            CompilationStatus.COMPLETED,
            CompilationStatus.RUNNING,
        )


# ── Schema suggestion ───────────────────────────────────────────────────────


class SchemaSuggestion(BaseModel):
    """Result of a ``suggest_schema`` call.

    ``input_description`` / ``output_description`` are only populated when
    the request asked the server to draft them.
    """

    reasoning: str = ""
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    input_description: Optional[str] = None
    output_description: Optional[str] = None
