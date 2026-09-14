from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union, cast

from pydantic import BaseModel, ConfigDict, Field

ResourceID = Union[str, int]
RuntimeConnections = Mapping[str, Union[str, Sequence[str]]]


class AgentStatus(str, Enum):
    DRAFT = "draft"
    INITIALIZATION_REQUIRED = "initialization_required"
    ACTIVE = "active"
    ARCHIVED = "archived"


class McpAccessMode(str, Enum):
    """How an agent resolves connector authority at runtime."""

    LEGACY = "legacy"
    SCOPED = "scoped"


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


class McpResourceType(str, Enum):
    """Exact resource kinds accepted by an MCP resource set."""

    USER_MCP_TOOL = "user_mcp_tool"
    CUSTOM_MCP_SERVER = "custom_mcp_server"
    INTEGRATION_CONNECTION = "integration_connection"


class McpResourceSetStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"


class McpResourceSetManagementMode(str, Enum):
    FLYMYAI = "flymyai"
    CUSTOMER = "customer"


class McpResourceSetAuthorityType(str, Enum):
    OWNER = "owner"
    EXTERNAL_PRINCIPAL = "external_principal"


class ExternalPrincipalStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"


class IntegrationConnectionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ERROR = "error"
    DISABLED = "disabled"


class WorkspaceGrantSubjectKind(str, Enum):
    """Stable owner resource that may receive workspace access."""

    TASK = "task"
    GROUP = "group"


class WorkspaceGrantRole(str, Enum):
    """Access level granted to an owner task or flat agent group."""

    READ = "read"
    WRITE = "write"


class ExecutionLogType(str, Enum):
    DECLARED_FUNCTIONS = "declared_functions"
    TOOL_CALLED = "tool_called"
    TOOL_CALL_EXCEPTION = "tool_call_exception"
    TASK_CANCELLED = "task_cancelled"
    USER_ADDED_MESSAGE = "user_added_message"

    @classmethod
    def _missing_(cls, value: object):
        obj = str.__new__(cls, value)
        obj._value_ = cast(str, value)
        obj._name_ = str(value).upper()
        return obj


class Agent(BaseModel):
    """An agent task - the top-level configuration for an autonomous agent."""

    uuid: str
    name: str
    # The bounded list contract (AgentTaskList) deliberately omits user_prompt
    # (it dominates list bytes); only the detail row carries it. A default keeps
    # one Agent model valid for both row shapes.
    user_prompt: str = ""
    available_tools: Any = Field(default_factory=list)
    available_custom_mcp_servers: List[int] = Field(default_factory=list)
    mcp_resource_set_ids: List[str] = Field(default_factory=list)
    mcp_access_mode: McpAccessMode = McpAccessMode.LEGACY
    input_schema: Optional[Dict[str, Any]] = None
    input_description: str = ""
    output_schema: Optional[Dict[str, Any]] = None
    output_description: str = ""
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


class ExecutionLog(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime
    type: ExecutionLogType
    message: str
    data: Any = Field(default_factory=dict)


class RunTaskSummary(BaseModel):
    """Bounded task metadata embedded in an execution-list row."""

    model_config = ConfigDict(extra="ignore")

    uuid: str
    name: str
    status: AgentStatus
    is_public: bool = False
    available_tool_labels: List[str] = Field(default_factory=list)
    available_tool_labels_complete: bool = False
    has_input_schema: bool = False


class RunCompilationSummary(BaseModel):
    """Bounded latest-compilation metadata embedded in an execution row."""

    model_config = ConfigDict(extra="ignore")

    id: ResourceID
    status: CompilationStatus
    created_at: datetime
    cron_schedule: str = ""
    timezone: str = ""


class Run(BaseModel):
    """A single agent execution (run)."""

    id: ResourceID
    user_agent_task: int
    previous_execution: Optional[ResourceID] = None
    original_prompt: str = ""
    variables: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    run_seq: int = 0
    effort: str = ""
    model: str = ""
    is_favorite: bool = False
    task_summary: Optional[RunTaskSummary] = None
    latest_compilation: Optional[RunCompilationSummary] = None
    total_price: Any = None
    total_price_complete: Optional[bool] = None
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


class RunGoalProgress(BaseModel):
    """Optional owner-only goal progress; verified is still pending completion."""

    model_config = ConfigDict(extra="ignore")

    # Read models preserve future server vocabulary without promoting it to
    # execution authority. Run.is_terminal still uses the canonical run status.
    schema_version: str = Field(alias="schema", min_length=1, max_length=128)
    goal_id: Optional[str] = Field(
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    )
    status: Optional[str] = Field(min_length=1, max_length=128)
    review_count: Optional[int] = Field(ge=0, le=64)
    reason: str = Field(min_length=1, max_length=128)
    terminal: Optional[bool]
    run_seq: int = Field(ge=0)
    revision: Optional[int] = Field(ge=0)


class CodingAvailability(BaseModel):
    """Configured owner admission capability, not a live worker/provider probe."""

    default: Literal["coding"]
    available: bool
    code: Optional[str]
    detail: Optional[str]


class RunContinuationCapabilities(BaseModel):
    append: bool
    fork: bool
    resume: bool
    reason: Optional[str]


class RunRuntimeAdmission(BaseModel):
    runtime: Literal["coding", "legacy", "unavailable"]
    admission: Literal["pinned", "legacy", "unknown"]
    continuation: RunContinuationCapabilities


class CodingContinuation(BaseModel):
    """New execution linked to its completed source; observe the returned id."""

    id: ResourceID
    previous_execution: ResourceID
    status: ExecutionStatus
    effort: str
    model: str
    run_seq: int = Field(ge=0)


class RunDetail(Run):
    """Run with execution logs attached."""

    logs: List[ExecutionLog] = Field(default_factory=list)
    user_agent_task_uuid: Optional[str] = None
    goal: Optional[RunGoalProgress] = None
    runtime_admission: Optional[RunRuntimeAdmission] = None


class RunResourceInline(BaseModel):
    format: Literal["json", "text"]
    value: Any
    size_bytes: int = Field(ge=0, le=4096)
    truncated: bool


class RunResourceRetrieval(BaseModel):
    href: str
    accept_ranges: Literal["bytes"]
    max_range_bytes: Literal[65536]


class RunResourceReceipt(BaseModel):
    kind: Literal["chunked_postgres_v1"]
    chunk_bytes: Literal[65536]


class RunResource(BaseModel):
    """Owner-authorized result/error reference, bound to run and content digest."""

    version: Literal["execution_resource_v1"]
    ref: str = Field(max_length=512)
    kind: Literal["agent_result", "error"]
    media_type: Literal["application/json", "text/plain"]
    encoding: Literal["utf-8"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inline: RunResourceInline
    truncated: bool
    retrieval: RunResourceRetrieval
    receipt: RunResourceReceipt


class RunPresentationCursor(BaseModel):
    version: Literal["presentation_cursor_v1"]
    run_seq: int = Field(ge=0)
    as_of_seq: int = Field(ge=0)


class RunStep(BaseModel):
    """A compact status event. Full log data requires an explicit history read."""

    id: int = Field(ge=1)
    type: str
    message: str
    message_size_bytes: int = Field(ge=0)
    message_truncated: bool
    label: str
    label_size_bytes: int = Field(ge=0)
    label_truncated: bool
    # Observation generation, not the generation in which a historical log was written.
    observed_run_seq: Optional[int] = None


class RunStatus(BaseModel):
    """One bounded observation page, not a fully hydrated execution."""

    view: Literal["bounded_v1"]
    id: ResourceID
    status: Literal["pending", "running", "completed", "failed", "cancelled", "archived"]
    run_seq: int = Field(ge=0)
    updated_at: datetime
    is_settled: bool
    step_count: int = Field(ge=0, le=10000)
    tool_step_count: int = Field(ge=0, le=10000)
    last_step_id: Optional[int] = Field(ge=1)
    new_steps: List[RunStep] = Field(max_length=100)
    agent_surface_revision: int = Field(ge=0)
    page_size: int = Field(ge=1, le=100)
    has_more: bool
    next_since: int = Field(ge=0)
    poll_complete: bool
    step_count_has_more: bool
    presentation_cursor_v1: RunPresentationCursor
    goal: Optional[RunGoalProgress] = None
    runtime_admission: Optional[RunRuntimeAdmission] = None
    chat_files_revision: Optional[str] = None
    result: Optional[RunResource] = None
    error: Optional[RunResource] = None

    @property
    def is_terminal(self) -> bool:
        return self.is_settled and self.status in {
            "completed", "failed", "cancelled", "archived",
        }

    @property
    def output(self) -> Any:
        """Return a complete inline result; large results require read_resource."""
        if self.result is None:
            return None
        if self.result.truncated or self.result.inline.truncated:
            raise ValueError("Result is a resource; use runs.read_resource() with explicit ranges.")
        return self.result.inline.value


class RunTranscriptPage(BaseModel):
    """Recent messages in chronological order; next_cursor selects older messages."""

    messages: List[Dict[str, Any]] = Field(max_length=50)
    has_more: bool
    next_cursor: Optional[str] = Field(max_length=512)
    page_size: int = Field(ge=1, le=50)
    response_bytes_limit: int
    receipt: Dict[str, Any]


class RunLogPage(BaseModel):
    logs: List[Dict[str, Any]] = Field(max_length=100)
    has_more: bool
    next_cursor: Optional[str] = Field(max_length=512)
    page_size: int = Field(ge=1, le=100)
    response_bytes_limit: int
    receipt: Dict[str, Any]


class AppendMessageResponse(BaseModel):
    """Bounded acknowledgement returned after appending to a run."""

    id: ResourceID
    status: ExecutionStatus = ExecutionStatus.PENDING
    effort: str = ""
    model: str = ""
    run_seq: int = 0
    error: Optional[str] = None
    agent_result: Optional[Dict[str, Any]] = None
    chat_files: List[Dict[str, Any]] = Field(default_factory=list)

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


class ConfigurationStep(BaseModel):
    description: str
    step_type: str
    vars_from_user_schema: Optional[Any] = None
    configuration_schema: Optional[Any] = None
    config: Optional[Any] = None
    execution_command: Optional[str] = None


class BrowserUseProfileStatus(str, Enum):
    UNBOUND = "unbound"
    BOUND = "bound"
    UNKNOWN = "unknown"


class BrowserUseProfile(BaseModel):
    """Non-secret saved browser metadata; cookie presence is not login proof."""

    profile_id: str
    name: str
    cookie_domains: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    note: str = ""


class BrowserUseProfileBinding(BaseModel):
    connection_id: str
    status: BrowserUseProfileStatus
    profile: Optional[BrowserUseProfile] = None
    code: Optional[str] = None
    detail: Optional[str] = None
    recovery_intent_id: Optional[str] = None


class Tool(BaseModel):
    """A configured MCP tool belonging to the user."""

    id: int
    public_id: Optional[str] = None
    mcp_tool: str
    alias: str = "default"
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


class McpResourceSetMemberInput(BaseModel):
    """One exact resource in one logical slot for an atomic membership update."""

    model_config = ConfigDict(extra="forbid")

    resource_type: McpResourceType
    resource_id: str
    slot: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    allowed_actions: List[str] = Field(default_factory=list)
    position: Optional[int] = Field(default=None, ge=0)


class McpResourceSetMember(BaseModel):
    """Secret-free projection of one exact MCP resource-set member."""

    model_config = ConfigDict(extra="ignore")

    public_id: str
    resource_type: McpResourceType
    resource_id: str
    toolkit_slug: str
    alias: str
    display_name: str
    slot: str
    allowed_actions: List[str] = Field(default_factory=list)
    position: int
    created_at: datetime
    updated_at: datetime


class McpResourceSetSummary(BaseModel):
    """Bounded collection projection without nested membership payloads."""

    model_config = ConfigDict(extra="ignore")

    public_id: str
    name: str
    description: str = ""
    status: McpResourceSetStatus = McpResourceSetStatus.ACTIVE
    management_mode: McpResourceSetManagementMode
    authority_type: McpResourceSetAuthorityType
    principal_id: Optional[str] = None
    revision: int = Field(ge=1, strict=True)
    member_count: int = Field(ge=0, strict=True)
    created_at: datetime
    updated_at: datetime

    @property
    def id(self) -> str:
        return self.public_id


class McpResourceSet(McpResourceSetSummary):
    """Full resource-set detail with all bounded members included."""

    members: List[McpResourceSetMember]


class AgentGroup(BaseModel):
    """Flat owner-scoped group with atomic agent and resource-set assignments."""

    model_config = ConfigDict(extra="ignore")

    public_id: str
    name: str
    description: str = ""
    is_active: bool = True
    agent_ids: List[str] = Field(default_factory=list)
    resource_set_ids: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @property
    def id(self) -> str:
        return self.public_id


class WorkspaceGrantSubject(BaseModel):
    """Stable task or group identity returned by the workspace grant API."""

    model_config = ConfigDict(extra="ignore")

    kind: WorkspaceGrantSubjectKind
    id: str


class WorkspaceGrant(BaseModel):
    """One active owner workspace grant."""

    model_config = ConfigDict(extra="ignore")

    grant_id: str
    subject: WorkspaceGrantSubject
    role: WorkspaceGrantRole
    created_at: datetime

    @property
    def id(self) -> str:
        return self.grant_id


class WorkspaceGrantPage(BaseModel):
    """One keyset-paginated workspace grant page, capped by the backend."""

    model_config = ConfigDict(extra="ignore")

    workspace: str
    revision: int = Field(ge=0, strict=True)
    grants: List[WorkspaceGrant] = Field(default_factory=list)
    next_cursor: Optional[str] = None


class WorkspaceGrantMutation(BaseModel):
    """Compact acknowledgement for a grant upsert or revocation."""

    model_config = ConfigDict(extra="ignore")

    grant_id: str
    revision: int = Field(ge=0, strict=True)
    no_change: bool = False


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
    """A compiled/frozen script or instruction derived from an agent execution."""

    id: ResourceID
    execution: ResourceID
    status: CompilationStatus = CompilationStatus.PENDING
    script_code: str = ""
    instruction_md: str = ""
    cron_schedule: str = ""
    timezone: str = "UTC"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @property
    def is_ready(self) -> bool:
        """Match the backend's runnable compilation and instruction states."""
        if self.status in (
            CompilationStatus.COMPILED,
            CompilationStatus.COMPLETED,
        ):
            return True
        return bool(self.instruction_md) and self.status in (
            CompilationStatus.RUNNING,
            CompilationStatus.FAILED,
        )


class AgentVersion(BaseModel):
    """Immutable frozen runtime version of an agent."""

    public_id: str
    agent_task: str
    source_compilation: Optional[int] = None
    version_number: int
    instruction_md: str = ""
    runtime_manifest: Dict[str, Any] = Field(default_factory=dict)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    llm_model: str = ""
    effort: str = ""
    created_at: datetime

    @property
    def id(self) -> str:
        return self.public_id


class AgentAccessRequirement(BaseModel):
    """One logical MCP access slot declared by an immutable version."""

    public_id: str
    agent_version: str
    slot: str
    toolkit_slug: str
    adapter_provider: Optional[str] = None
    connection_required: bool = True
    hosted_setup_supported: bool = False
    cardinality: str
    exact_actions: List[str] = Field(default_factory=list)
    inferred_actions: List[str] = Field(default_factory=list)
    risk_class: str = "read"
    policy: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class AgentDeployment(BaseModel):
    """Stable endpoint that publishes one immutable agent version."""

    public_id: str
    agent_task: str
    active_version: Optional[str] = None
    candidate_version: Optional[str] = None
    name: str
    status: str
    publish_mode: str
    created_at: datetime
    updated_at: datetime

    @property
    def id(self) -> str:
        return self.public_id


class ExternalPrincipal(BaseModel):
    """Typed, secret-free identity for one product customer and deployment."""

    model_config = ConfigDict(extra="ignore")

    public_id: str
    deployment: str
    external_user_id: str
    display_name: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: ExternalPrincipalStatus = ExternalPrincipalStatus.ACTIVE
    created_at: datetime
    updated_at: datetime

    @property
    def id(self) -> str:
        return self.public_id


class IntegrationConnection(BaseModel):
    """Secret-free projection of one exact external connector account."""

    model_config = ConfigDict(extra="ignore")

    public_id: str
    principal: str
    toolkit_slug: str
    alias: str = ""
    status: IntegrationConnectionStatus = IntegrationConnectionStatus.PENDING
    provider: str = ""
    provider_connection_id: str = ""
    provider_account_id: str = ""
    granted_scopes: List[str] = Field(default_factory=list)
    legacy_user_mcp_tool: Optional[int] = None
    credentials_configured: bool = False
    credential_revision: int = Field(default=1, ge=1, strict=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_error: str = ""
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    @property
    def id(self) -> str:
        return self.public_id


class ConnectionBinding(BaseModel):
    """Exact logical slot to connection UUID assignment for one principal."""

    model_config = ConfigDict(extra="ignore")

    public_id: str
    principal: str
    slot: str
    connections: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @property
    def id(self) -> str:
        return self.public_id


class AgentDeploymentAccess(BaseModel):
    """Publish manifest plus optional exact-customer connection metadata."""

    deployment: AgentDeployment
    active_version: Optional[AgentVersion] = None
    candidate_version: Optional[AgentVersion] = None
    requirements: List[AgentAccessRequirement] = Field(default_factory=list)
    principals: List[ExternalPrincipal] = Field(default_factory=list)
    connections: List[IntegrationConnection] = Field(default_factory=list)
    bindings: List[ConnectionBinding] = Field(default_factory=list)

    def principal_for_external_user(
        self,
        external_user_id: str,
    ) -> ExternalPrincipal:
        """Return one exact principal or fail closed on missing/ambiguous data."""
        if not isinstance(external_user_id, str) or not external_user_id.strip():
            raise ValueError("external_user_id must be a non-blank string.")
        matches = [
            principal
            for principal in self.principals
            if principal.external_user_id == external_user_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one external principal for external_user_id "
                f"{external_user_id!r}; found {len(matches)}."
            )
        return matches[0]

    def ready_connection_ids_for_slot(
        self,
        *,
        principal_id: str,
        slot: str,
    ) -> List[str]:
        """Return exact active connection UUIDs or fail closed until ready."""
        if not isinstance(principal_id, str) or not principal_id.strip():
            raise ValueError("principal_id must be a non-blank string.")
        if not isinstance(slot, str) or not slot.strip():
            raise ValueError("slot must be a non-blank string.")
        bindings = [
            binding
            for binding in self.bindings
            if binding.principal == principal_id and binding.slot == slot
        ]
        if len(bindings) != 1:
            raise ValueError(
                "Expected exactly one connection binding for principal_id "
                f"{principal_id!r} and slot {slot!r}; found {len(bindings)}."
            )
        connection_ids = bindings[0].connections
        if not connection_ids:
            raise ValueError(f"Connection slot {slot!r} is not ready.")
        connections_by_id = {
            connection.id: connection for connection in self.connections
        }
        if len(connections_by_id) != len(self.connections):
            raise ValueError("Deployment access contains duplicate connection UUIDs.")
        for connection_id in connection_ids:
            connection = connections_by_id.get(connection_id)
            if connection is None or connection.principal != principal_id:
                raise ValueError(
                    f"Connection slot {slot!r} contains an invalid connection UUID."
                )
            if (
                connection.status is not IntegrationConnectionStatus.ACTIVE
                or not connection.credentials_configured
            ):
                raise ValueError(f"Connection slot {slot!r} is not ready.")
        return list(connection_ids)


class AgentConnectionSession(BaseModel):
    """Short-lived hosted authorization link for one customer access slot."""

    redirect_url: str
    expires_at: datetime
    provider: str


class AgentDeploymentPreflight(BaseModel):
    """Authoritative server-side validation for a staged deployment release."""

    ready: bool


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
