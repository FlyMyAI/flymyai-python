from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


ResourceID = Union[str, int]


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
    USER_ADDED_MESSAGE = "user_added_message"

    @classmethod
    def _missing_(cls, value: object):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._name_ = str(value).upper()
        return obj


class Agent(BaseModel):
    """An agent task — the top-level configuration for an autonomous agent."""

    uuid: str
    name: str
    user_prompt: str
    available_tools: Any = Field(default_factory=list)
    all_tools_configured: bool = False
    tools_need_to_configure: List[int] = Field(default_factory=list)
    generated_pipeline: Dict[str, Any] = Field(default_factory=dict)
    status: AgentStatus = AgentStatus.DRAFT
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


class ExecutionLog(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    type: ExecutionLogType
    message: str
    data: Any = Field(default_factory=dict)


class Run(BaseModel):
    """A single agent execution (run)."""

    id: ResourceID
    user_agent_task: int
    previous_execution: Optional[ResourceID] = None
    original_prompt: str
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


class Compilation(BaseModel):
    """A compiled script derived from an agent execution."""

    id: ResourceID
    execution: ResourceID
    status: CompilationStatus = CompilationStatus.PENDING
    script_code: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
