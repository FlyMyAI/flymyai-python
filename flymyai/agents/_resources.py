from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from flymyai.agents._types import (
    Agent,
    AgentGroup,
    AgentConnectionSession,
    AgentDeployment,
    AgentDeploymentAccess,
    AgentDeploymentPreflight,
    AgentDetail,
    AgentVersion,
    AppendMessageResponse,
    AvailableTool,
    Compilation,
    CompilationStatus,
    McpAccessMode,
    McpResourceSet,
    McpResourceSetAuthorityType,
    McpResourceSetMember,
    McpResourceSetMemberInput,
    McpResourceSetManagementMode,
    McpResourceSetSummary,
    McpResourceSetStatus,
    ResourceID,
    RuntimeConnections,
    Run,
    RunDetail,
    RunLogPage,
    RunProgressStatus,
    RunProgressStep,
    RunResource,
    RunResourceRange,
    RunTranscriptPage,
    SchemaSuggestion,
    Tool,
)
from flymyai.core.idempotency import idempotency_headers as _idempotency_headers

if TYPE_CHECKING:
    from flymyai.agents._client import AsyncAgentClient, SyncAgentClient


_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_MAX_CURSOR_PAGES = 100
_MAX_CURSOR_ROWS = 10_000
_MAX_CURSOR_PAGE_ROWS = 100
_MAX_CURSOR_CHARS = 1024
_DEFAULT_CURSOR_PAGE_SIZE = 100
_RUN_STATUS_MAX_RESPONSE_BYTES = 1024 * 1024
_RUN_PAGE_MAX_RESPONSE_BYTES = 512 * 1024
_RUN_RESOURCE_MAX_RANGE_BYTES = 65_536
_RUN_RESOURCE_MAX_BYTES = 64 * 1024 * 1024
_RUN_CURSOR_MAX_CHARS = 512
_RUN_WAIT_MAX_REQUESTS = 1_000
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PaginationKey = Tuple[Tuple[str, str], ...]
_McpResourceSetMemberLike = Union[
    McpResourceSetMemberInput,
    Mapping[str, Any],
]
_McpResourceSetStatusLike = Union[McpResourceSetStatus, str]
_McpResourceSetManagementModeLike = Union[McpResourceSetManagementMode, str]
_McpResourceSetAuthorityTypeLike = Union[McpResourceSetAuthorityType, str]
_McpAccessModeLike = Union[McpAccessMode, str]
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_DEPRECATED_COMPILATION_RUN_MESSAGE = (
    "Compilations.run() is disabled because the legacy endpoint has no "
    "caller-owned replay contract. Use Compilations.run_instruction(..., "
    "idempotency_key=...) for an owner run or Deployments.run(..., "
    "idempotency_key=...) for a published deployment."
)


def _resource_set_member_payloads(
    members: Sequence[_McpResourceSetMemberLike],
) -> List[Dict[str, Any]]:
    """Validate and serialize only the public membership contract."""
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
        raise ValueError("members must be a sequence of resource-set members.")
    if len(members) > 100:
        raise ValueError("members must contain at most 100 entries.")
    payloads: List[Dict[str, Any]] = []
    seen_members: set[tuple[str, str, str]] = set()
    slot_action_ceilings: Dict[str, tuple[str, ...]] = {}
    slot_counts: Dict[str, int] = {}
    for member in members:
        validated = (
            member
            if isinstance(member, McpResourceSetMemberInput)
            else McpResourceSetMemberInput(**dict(member))
        )
        member_key = (
            validated.resource_type.value,
            _validate_public_uuid(
                validated.resource_id,
                field_name="resource_id",
            ),
            validated.slot,
        )
        if member_key in seen_members:
            raise ValueError(
                "members contains a duplicate (resource_type, resource_id, slot)"
            )
        seen_members.add(member_key)

        action_ceiling = tuple(sorted(validated.allowed_actions))
        if (
            validated.slot in slot_action_ceilings
            and slot_action_ceilings[validated.slot] != action_ceiling
        ):
            raise ValueError(
                f"slot {validated.slot!r} cannot mix different action ceilings"
            )
        slot_action_ceilings[validated.slot] = action_ceiling
        slot_counts[validated.slot] = slot_counts.get(validated.slot, 0) + 1
        if slot_counts[validated.slot] > 25:
            raise ValueError(
                f"slot {validated.slot!r} must contain at most 25 resources"
            )
        payloads.append(validated.model_dump(mode="json", exclude_none=True))
    return payloads


def _string_enum_value(value: Any) -> str:
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def _validate_slug(value: str, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    if not 1 <= len(value) <= max_length:
        raise ValueError(
            f"{field_name} must contain between 1 and {max_length} characters."
        )
    if _SLUG_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} may contain only ASCII letters, digits, underscores, "
            "and hyphens."
        )
    return value


def _validate_alias(alias: str) -> str:
    return _validate_slug(alias, field_name="alias", max_length=64)


def _validate_slot(slot: str) -> str:
    return _validate_slug(slot, field_name="slot", max_length=128)


def _validate_public_uuid(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank UUID string.")
    try:
        UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID string.") from exc
    return value


def _mcp_access_mode_value(value: _McpAccessModeLike) -> str:
    try:
        return McpAccessMode(_string_enum_value(value)).value
    except ValueError as exc:
        raise ValueError("mcp_access_mode must be 'legacy' or 'scoped'.") from exc


def _mcp_resource_set_authority_type_value(
    value: _McpResourceSetAuthorityTypeLike,
) -> str:
    try:
        return McpResourceSetAuthorityType(_string_enum_value(value)).value
    except ValueError as exc:
        raise ValueError(
            "authority_type must be 'owner' or 'external_principal'."
        ) from exc


def _validate_collection_query(query: Optional[str]) -> Optional[str]:
    if query is None:
        return None
    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    if len(query) > 256:
        raise ValueError("query must contain at most 256 characters.")
    normalized = query.strip()
    if normalized and not normalized.isprintable():
        raise ValueError("query must not contain control characters.")
    return normalized or None


def _validate_principal_id_filter(principal_id: Optional[str]) -> Optional[str]:
    if principal_id is None:
        return None
    if not isinstance(principal_id, str) or not principal_id.strip():
        raise ValueError("principal_id must be a non-blank string.")
    return principal_id


def _validate_expected_revision(expected_revision: int) -> int:
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 1
    ):
        raise ValueError("expected_revision must be an integer of at least 1.")
    return expected_revision


def _validate_external_customer_pair(
    *,
    external_user_id: Optional[str],
    deployment_id: Optional[str],
) -> None:
    if (external_user_id is None) != (deployment_id is None):
        raise ValueError(
            "external_user_id and deployment_id must be supplied together."
        )


def _runtime_connections_payload(
    connections: RuntimeConnections,
) -> Dict[str, Union[str, List[str]]]:
    if not isinstance(connections, Mapping):
        raise ValueError("connections must be a mapping from slot to connection IDs.")
    if len(connections) > 100:
        raise ValueError("connections must contain at most 100 slots.")

    payload: Dict[str, Union[str, List[str]]] = {}
    for raw_slot, raw_selection in connections.items():
        slot = _validate_slot(raw_slot)
        if isinstance(raw_selection, str):
            payload[slot] = _validate_public_uuid(
                raw_selection,
                field_name=f"connections[{slot!r}]",
            )
            continue

        if not isinstance(raw_selection, Sequence) or isinstance(
            raw_selection, (str, bytes)
        ):
            raise ValueError(
                f"connections[{slot!r}] must be one connection ID or a sequence "
                "of connection IDs."
            )
        if not 1 <= len(raw_selection) <= 25:
            raise ValueError(
                f"connections[{slot!r}] must contain between 1 and 25 IDs."
            )

        connection_ids: List[str] = []
        for connection_id in raw_selection:
            connection_ids.append(
                _validate_public_uuid(
                    connection_id,
                    field_name=f"connections[{slot!r}] item",
                )
            )
        if len(set(connection_ids)) != len(connection_ids):
            raise ValueError(f"connections[{slot!r}] must not contain duplicate IDs.")
        payload[slot] = connection_ids
    return payload


def _resource_set_create_payload(
    *,
    name: str,
    description: Optional[str],
    status: Optional[_McpResourceSetStatusLike],
    management_mode: _McpResourceSetManagementModeLike,
    principal_id: Optional[str],
) -> Dict[str, Any]:
    try:
        mode = McpResourceSetManagementMode(_string_enum_value(management_mode))
    except ValueError as exc:
        raise ValueError("management_mode must be 'flymyai' or 'customer'.") from exc

    if principal_id is None and mode is not McpResourceSetManagementMode.FLYMYAI:
        raise ValueError("management_mode='customer' requires the exact principal_id.")
    if principal_id is not None and mode is not McpResourceSetManagementMode.CUSTOMER:
        raise ValueError(
            "principal_id requires management_mode='customer'; owner sets use "
            "management_mode='flymyai' without a principal_id."
        )

    body: Dict[str, Any] = {"name": name, "management_mode": mode.value}
    if description is not None:
        body["description"] = description
    if status is not None:
        body["status"] = _string_enum_value(status)
    if principal_id is not None:
        body["principal_id"] = principal_id
    return body


def _resource_set_metadata_payload(
    *,
    expected_revision: int,
    name: Optional[str],
    description: Optional[str],
    status: Optional[_McpResourceSetStatusLike],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "expected_revision": _validate_expected_revision(expected_revision)
    }
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if status is not None:
        body["status"] = _string_enum_value(status)
    return body


def _validate_runtime_resource_selection(
    *,
    connections: Optional[RuntimeConnections],
    resource_set_id: Optional[str],
    resource_set_revision: Optional[int],
) -> None:
    if connections is not None and resource_set_id is not None:
        raise ValueError("Supply either resource_set_id or connections, not both.")
    if resource_set_revision is not None and resource_set_id is None:
        raise ValueError("resource_set_revision requires resource_set_id.")
    if resource_set_revision is not None:
        if (
            isinstance(resource_set_revision, bool)
            or not isinstance(resource_set_revision, int)
            or resource_set_revision < 1
        ):
            raise ValueError("resource_set_revision must be an integer of at least 1.")
    if connections is not None:
        _runtime_connections_payload(connections)


def _validate_cursor_page_size(page_size: int) -> int:
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= 100
    ):
        raise ValueError("page_size must be an integer between 1 and 100.")
    return page_size


def _validate_max_items(max_items: Optional[int]) -> Optional[int]:
    if max_items is None:
        return None
    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or not 1 <= max_items <= _MAX_CURSOR_ROWS
    ):
        raise ValueError(
            f"max_items must be an integer between 1 and {_MAX_CURSOR_ROWS}."
        )
    return max_items


def _validate_status_since(since: int) -> int:
    if (
        isinstance(since, bool)
        or not isinstance(since, int)
        or not 0 <= since <= 9_223_372_036_854_775_807
    ):
        raise ValueError("since must be an integer between 0 and 2^63 - 1.")
    return since


def _validate_run_page_size(page_size: int, *, maximum: int) -> int:
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= maximum
    ):
        raise ValueError(f"page_size must be an integer between 1 and {maximum}.")
    return page_size


def _validate_run_cursor(cursor: Optional[str]) -> Optional[str]:
    if cursor is None:
        return None
    if (
        not isinstance(cursor, str)
        or not cursor
        or len(cursor) > _RUN_CURSOR_MAX_CHARS
        or not cursor.isprintable()
    ):
        raise ValueError(
            "cursor must be a non-blank printable string of at most 512 characters."
        )
    return cursor


def _bounded_response_json(response: Any, *, resource_name: str, max_bytes: int) -> Any:
    if len(response.content) > max_bytes:
        raise RuntimeError(f"{resource_name} response exceeded {max_bytes} bytes.")
    try:
        return response.json()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{resource_name} returned malformed JSON.") from exc


def _validate_run_resource(resource: RunResource) -> None:
    if resource.version != "execution_resource_v1":
        raise RuntimeError("Run resource returned an unexpected version.")
    if resource.kind not in {"agent_result", "error"}:
        raise RuntimeError("Run resource returned an unexpected kind.")
    expected_media_type = (
        "application/json" if resource.kind == "agent_result" else "text/plain"
    )
    if resource.media_type != expected_media_type or resource.encoding != "utf-8":
        raise RuntimeError("Run resource returned unexpected encoding metadata.")
    if not 0 <= resource.size_bytes <= _RUN_RESOURCE_MAX_BYTES:
        raise RuntimeError("Run resource size is outside the public contract.")
    if _SHA256_PATTERN.fullmatch(resource.sha256) is None:
        raise RuntimeError("Run resource returned an invalid SHA-256 digest.")
    if (
        not resource.ref
        or len(resource.ref) > _RUN_CURSOR_MAX_CHARS
        or not resource.ref.isprintable()
    ):
        raise RuntimeError("Run resource returned an invalid opaque ref.")
    if resource.inline.format not in {"json", "text"}:
        raise RuntimeError("Run resource returned an invalid inline format.")
    if not 0 <= resource.inline.size_bytes <= 4_096:
        raise RuntimeError("Run resource returned an oversized inline projection.")
    if resource.inline.truncated != resource.truncated:
        raise RuntimeError("Run resource returned inconsistent truncation metadata.")
    if not resource.truncated and resource.inline.size_bytes != resource.size_bytes:
        raise RuntimeError("Run resource returned inconsistent inline byte metadata.")
    if resource.retrieval.accept_ranges != "bytes" or not (
        1
        <= resource.retrieval.max_range_bytes
        <= _RUN_RESOURCE_MAX_RANGE_BYTES
    ):
        raise RuntimeError("Run resource returned invalid range metadata.")
    if (
        resource.receipt.kind != "chunked_postgres_v1"
        or resource.receipt.chunk_bytes != _RUN_RESOURCE_MAX_RANGE_BYTES
    ):
        raise RuntimeError("Run resource returned invalid receipt metadata.")


def _validate_transcript_page(
    page: RunTranscriptPage,
    *,
    cursor: Optional[str],
    page_size: int,
) -> None:
    if page.page_size != page_size or len(page.messages) > page_size:
        raise RuntimeError("Run transcript returned an unexpected page size.")
    if page.response_bytes_limit != _RUN_PAGE_MAX_RESPONSE_BYTES:
        raise RuntimeError("Run transcript returned an unexpected response limit.")
    next_cursor = _validate_run_cursor(page.next_cursor)
    if page.has_more != (next_cursor is not None):
        raise RuntimeError("Run transcript returned inconsistent cursor metadata.")
    if next_cursor is not None and next_cursor == cursor:
        raise RuntimeError("Run transcript cursor did not advance.")
    receipt = page.receipt
    if (
        receipt.version != "execution_transcript_receipt_v1"
        or receipt.source not in {"display_messages", "messages"}
        or receipt.presentation_seq < 0
        or receipt.total_messages < 0
        or receipt.signature_algorithm != "hmac-sha256"
        or _SHA256_PATTERN.fullmatch(receipt.signature) is None
    ):
        raise RuntimeError("Run transcript returned invalid receipt metadata.")
    seen_ids: Set[str] = set()
    for message in page.messages:
        returned_bytes = len(message.content.encode("utf-8"))
        if message.role not in {"user", "assistant"}:
            raise RuntimeError("Run transcript returned an invalid message role.")
        if returned_bytes > 8_192:
            raise RuntimeError("Run transcript returned oversized message content.")
        if message.content_size_bytes < returned_bytes:
            raise RuntimeError("Run transcript returned invalid content size metadata.")
        if message.content_size_exact and (
            message.content_truncated
            != (returned_bytes < message.content_size_bytes)
        ):
            raise RuntimeError("Run transcript returned invalid truncation metadata.")
        if (
            not message.message_id
            or len(message.message_id) > 255
            or not message.message_id.isprintable()
            or message.message_id in seen_ids
        ):
            raise RuntimeError("Run transcript returned an invalid message identity.")
        seen_ids.add(message.message_id)


def _validate_log_page(
    page: RunLogPage,
    *,
    cursor: Optional[str],
    page_size: int,
) -> None:
    if page.page_size != page_size or len(page.logs) > page_size:
        raise RuntimeError("Run logs returned an unexpected page size.")
    if page.response_bytes_limit != _RUN_PAGE_MAX_RESPONSE_BYTES:
        raise RuntimeError("Run logs returned an unexpected response limit.")
    next_cursor = _validate_run_cursor(page.next_cursor)
    if page.has_more != (next_cursor is not None):
        raise RuntimeError("Run logs returned inconsistent cursor metadata.")
    if next_cursor is not None and next_cursor == cursor:
        raise RuntimeError("Run log cursor did not advance.")
    receipt = page.receipt
    if (
        receipt.version != "execution_log_receipt_v1"
        or receipt.snapshot_max_id < 0
        or receipt.ordering != "id"
        or receipt.signature_algorithm != "hmac-sha256"
        or _SHA256_PATTERN.fullmatch(receipt.signature) is None
    ):
        raise RuntimeError("Run logs returned invalid receipt metadata.")
    previous_id = 0
    for entry in page.logs:
        if entry.id <= previous_id or entry.id > receipt.snapshot_max_id:
            raise RuntimeError("Run log IDs must increase inside one snapshot page.")
        previous_id = entry.id
        if entry.data_hidden and (
            entry.data != {}
            or entry.data_size_exact
            or not entry.data_has_more
        ):
            raise RuntimeError("Run logs returned unsafe hidden-data metadata.")
        if entry.data_size_exact:
            if not 0 <= entry.data_size_bytes <= 4_096:
                raise RuntimeError("Run logs returned oversized exact data.")
            try:
                rendered = json.dumps(
                    entry.data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Run logs returned non-JSON data.") from exc
            if len(rendered) > 4_096:
                raise RuntimeError("Run logs returned oversized exact data.")


def _validate_wait_options(
    *, timeout: float, poll_interval: float, max_requests: int
) -> None:
    for field_name, value in (("timeout", timeout), ("poll_interval", poll_interval)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{field_name} must be a finite non-negative number.")
    if (
        isinstance(max_requests, bool)
        or not isinstance(max_requests, int)
        or not 1 <= max_requests <= _RUN_WAIT_MAX_REQUESTS
    ):
        raise ValueError(
            f"max_requests must be an integer between 1 and {_RUN_WAIT_MAX_REQUESTS}."
        )


def _validate_progress_page(
    result: RunProgressStatus,
    *,
    since: int,
    page_size: int,
) -> None:
    """Reject a malformed response before a poller can spin or grow state."""
    if result.view != "bounded_v1":
        raise RuntimeError("Run status returned an unexpected view.")
    if result.page_size != page_size:
        raise RuntimeError("Run status returned an unexpected page size.")
    if (
        result.step_count < 0
        or result.step_count > 10_000
        or result.tool_step_count < 0
        or result.tool_step_count > result.step_count
    ):
        raise RuntimeError("Run status returned invalid bounded totals.")
    if len(result.new_steps) > page_size:
        raise RuntimeError("Run status returned more steps than requested.")
    previous_id = since
    for step in result.new_steps:
        if step.id <= previous_id:
            raise RuntimeError("Run status step IDs must increase after since.")
        if len(step.message.encode("utf-8")) > 512:
            raise RuntimeError("Run status returned an oversized step message.")
        if len(step.label.encode("utf-8")) > 256:
            raise RuntimeError("Run status returned an oversized step label.")
        message_bytes = len(step.message.encode("utf-8"))
        label_bytes = len(step.label.encode("utf-8"))
        if (
            step.message_size_bytes < message_bytes
            or step.message_truncated
            != (message_bytes < step.message_size_bytes)
            or step.label_size_bytes < label_bytes
            or step.label_truncated != (label_bytes < step.label_size_bytes)
        ):
            raise RuntimeError("Run status returned invalid step size metadata.")
        previous_id = step.id
    if result.next_since != previous_id:
        raise RuntimeError("Run status returned an inconsistent next_since cursor.")
    if result.has_more and not result.new_steps:
        raise RuntimeError("Run status cannot advance a non-empty page cursor.")
    if result.poll_complete != (result.is_settled and not result.has_more):
        raise RuntimeError("Run status returned an inconsistent terminal page.")
    if not result.poll_complete and (result.result is not None or result.error is not None):
        raise RuntimeError("Run status exposed resources before polling completed.")
    for resource in (result.result, result.error):
        if resource is not None:
            _validate_run_resource(resource)


def _resource_range_from_response(
    response: Any,
    *,
    resource: RunResource,
    offset: int,
    length: int,
) -> RunResourceRange:
    if response.status_code != 206:
        raise RuntimeError("Run resource range did not return HTTP 206.")
    data = bytes(response.content)
    if len(data) != length or len(data) > _RUN_RESOURCE_MAX_RANGE_BYTES:
        raise RuntimeError("Run resource returned an unexpected byte count.")
    end = offset + len(data) - 1
    expected_content_range = f"bytes {offset}-{end}/{resource.size_bytes}"
    headers = response.headers
    if headers.get("Content-Length") != str(len(data)):
        raise RuntimeError("Run resource returned an invalid Content-Length.")
    if headers.get("Content-Range") != expected_content_range:
        raise RuntimeError("Run resource returned an invalid Content-Range.")
    if headers.get("X-Content-SHA256") != resource.sha256:
        raise RuntimeError("Run resource returned a different digest.")
    if headers.get("ETag") != f'"sha256:{resource.sha256}"':
        raise RuntimeError("Run resource returned an invalid ETag.")
    if headers.get("X-Execution-Resource-Ref") != resource.ref:
        raise RuntimeError("Run resource returned a different opaque ref.")
    content_type = headers.get("Content-Type", "").partition(";")[0].strip().lower()
    if content_type != resource.media_type:
        raise RuntimeError("Run resource returned an invalid media type.")
    if offset == 0 and len(data) == resource.size_bytes:
        if hashlib.sha256(data).hexdigest() != resource.sha256:
            raise RuntimeError("Run resource bytes failed digest verification.")
    return RunResourceRange(
        data=data,
        start=offset,
        end=end,
        total_bytes=resource.size_bytes,
        sha256=resource.sha256,
        ref=resource.ref,
        media_type=resource.media_type,
    )


def _compact_cursor_page(
    data: Any,
    *,
    resource_name: str,
) -> Tuple[List[Any], Optional[str]]:
    """Parse the compact cursor envelope or one bounded legacy array."""
    if isinstance(data, list):
        if len(data) > _MAX_CURSOR_PAGE_ROWS:
            raise RuntimeError(
                f"{resource_name} legacy response exceeded "
                f"{_MAX_CURSOR_PAGE_ROWS} rows."
            )
        return data, None
    if not isinstance(data, dict):
        raise RuntimeError(f"{resource_name} returned a malformed list response.")

    required_fields = {"next_cursor", "previous_cursor", "results"}
    missing_fields = sorted(required_fields.difference(data))
    if missing_fields:
        raise RuntimeError(
            f"{resource_name} cursor envelope is missing: {', '.join(missing_fields)}."
        )
    extra_fields = sorted(set(data).difference(required_fields))
    if extra_fields:
        raise RuntimeError(
            f"{resource_name} cursor envelope has unexpected fields: "
            f"{', '.join(extra_fields)}."
        )
    results = data["results"]
    if not isinstance(results, list):
        raise RuntimeError(f"{resource_name} cursor results must be a list.")
    if len(results) > _MAX_CURSOR_PAGE_ROWS:
        raise RuntimeError(
            f"{resource_name} cursor page exceeded {_MAX_CURSOR_PAGE_ROWS} rows."
        )

    for field_name in ("next_cursor", "previous_cursor"):
        cursor = data[field_name]
        if cursor is not None and (
            not isinstance(cursor, str)
            or not cursor.strip()
            or len(cursor) > _MAX_CURSOR_CHARS
            or not cursor.isprintable()
        ):
            raise RuntimeError(
                f"{resource_name} {field_name} must be null or a non-blank printable "
                f"string of at most {_MAX_CURSOR_CHARS} characters."
            )
    return results, data["next_cursor"]


def _next_compact_cursor_params(
    next_cursor: Optional[str],
    *,
    page_size: int,
    visited: Set[str],
    page_count: int,
    row_count: int,
    resource_name: str,
    base_params: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, Union[str, int]]]:
    if row_count > _MAX_CURSOR_ROWS:
        raise RuntimeError(
            f"{resource_name} pagination exceeded {_MAX_CURSOR_ROWS} rows."
        )
    if next_cursor is None:
        return None
    if next_cursor in visited:
        raise RuntimeError(
            f"{resource_name} pagination repeated a cursor; refusing to continue."
        )
    if page_count >= _MAX_CURSOR_PAGES:
        raise RuntimeError(
            f"{resource_name} pagination exceeded {_MAX_CURSOR_PAGES} pages; "
            "refusing to continue."
        )
    visited.add(next_cursor)
    params: Dict[str, Union[str, int]] = dict(base_params or {})
    params.update({"page_size": page_size, "cursor": next_cursor})
    return params


def _list_results(data: Any) -> List[Any]:
    """Accept both paginated DRF responses and legacy plain lists."""
    if isinstance(data, dict):
        results = data.get("results", [])
        return results if isinstance(results, list) else []
    return data if isinstance(data, list) else []


def _next_list_params(data: Any) -> Optional[Dict[str, str]]:
    if not isinstance(data, dict):
        return None
    next_url = data.get("next")
    if not isinstance(next_url, str) or not next_url:
        return None
    return dict(parse_qsl(urlsplit(next_url).query, keep_blank_values=True))


def _guarded_next_list_params(
    data: Any,
    *,
    base_params: Mapping[str, str],
    visited: Set[_PaginationKey],
    page_count: int,
    resource_name: str,
) -> Optional[Dict[str, str]]:
    parsed = _next_list_params(data)
    if parsed is None:
        return None
    params = {**parsed, **base_params}
    key: _PaginationKey
    if "cursor" in parsed:
        key = (("cursor", parsed["cursor"]),)
    else:
        key = tuple(sorted(parsed.items()))
    if key in visited:
        raise RuntimeError(
            f"{resource_name} pagination repeated a cursor; refusing to continue."
        )
    if page_count >= _MAX_CURSOR_PAGES:
        raise RuntimeError(
            f"{resource_name} pagination exceeded {_MAX_CURSOR_PAGES} pages; "
            "refusing to continue."
        )
    visited.add(key)
    return params


class Agents:
    """CRUD for agents. Maps to ``/api/v1/agents/tasks/``."""

    def __init__(self, client: "SyncAgentClient") -> None:
        self._c = client

    def create(
        self,
        *,
        name: str,
        goal: str,
        tools: Optional[List[int]] = None,
        mcp_servers: Optional[List[int]] = None,
        mcp_resource_set_ids: Optional[List[str]] = None,
        mcp_access_mode: Optional[_McpAccessModeLike] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        input_description: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        output_description: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Agent:
        """Create a new agent.

        Parameters
        ----------
        name:
            Human-readable agent name.
        goal:
            The agent's prompt / instructions (stored as ``user_prompt``).
            May contain Jinja2 placeholders like ``{{ topic }}`` - provide
            matching ``input_schema`` so the backend can validate runtime
            ``variables``.
        tools:
            List of ``UserMcpTool`` IDs to attach.
        mcp_servers:
            List of custom MCP server IDs to attach.
        mcp_resource_set_ids:
            Stable public UUIDs of owner MCP resource sets to grant directly.
        mcp_access_mode:
            ``legacy`` keeps direct personal connection behavior. ``scoped``
            enables the explicit resource-set allowlist, including an empty
            allowlist.
        input_schema:
            JSON Schema describing runtime variables accepted by ``run``.
            Required whenever ``goal`` contains Jinja2 placeholders.
        input_description:
            Plain-text description of what the agent expects as input.
            Used together with ``input_schema`` at freeze time as the
            authoritative scope: chat steps outside the input→output
            canonical flow are dropped from the frozen instruction.
        output_schema:
            JSON Schema the agent must produce as its final result.
        output_description:
            Plain-text description of what the agent returns. Paired
            with ``output_schema`` to bound the canonical pipeline.
        status:
            Initial status (default ``draft``).
        """
        body: Dict[str, Any] = {"name": name, "user_prompt": goal}
        if tools is not None:
            body["available_tools"] = tools
        if mcp_servers is not None:
            body["available_custom_mcp_servers"] = mcp_servers
        if mcp_resource_set_ids is not None:
            body["mcp_resource_set_ids"] = mcp_resource_set_ids
        if mcp_access_mode is not None:
            body["mcp_access_mode"] = _mcp_access_mode_value(mcp_access_mode)
        if input_schema is not None:
            body["input_schema"] = input_schema
        if input_description is not None:
            body["input_description"] = input_description
        if output_schema is not None:
            body["output_schema"] = output_schema
        if output_description is not None:
            body["output_description"] = output_description
        if status is not None:
            body["status"] = status
        data = self._c._request("POST", "/api/v1/agents/tasks/", json=body)
        return Agent(**data)

    def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
    ) -> List[Agent]:
        """Return agents through bounded cursor pages."""
        page_size = _validate_cursor_page_size(page_size)
        base_params = {"pagination": "cursor"}
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                "/api/v1/agents/tasks/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="Agents",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="Agents",
                base_params=base_params,
            )
            if next_params is None:
                return [Agent(**item) for item in results]
            params = next_params

    def get(self, agent_id: str) -> AgentDetail:
        """Get agent by UUID (returns full detail with nested tools)."""
        data = self._c._request("GET", f"/api/v1/agents/tasks/{agent_id}/")
        return AgentDetail(**data)

    def update(
        self,
        agent_id: str,
        *,
        mcp_access_mode: Optional[_McpAccessModeLike] = None,
        **kwargs: Any,
    ) -> Agent:
        """Partial update (PATCH).

        Use ``goal=`` to update ``user_prompt``.
        """
        if "goal" in kwargs:
            kwargs["user_prompt"] = kwargs.pop("goal")
        if mcp_access_mode is not None:
            kwargs["mcp_access_mode"] = _mcp_access_mode_value(mcp_access_mode)
        data = self._c._request(
            "PATCH", f"/api/v1/agents/tasks/{agent_id}/", json=kwargs
        )
        return Agent(**data)

    def delete(self, agent_id: str) -> None:
        """Soft-delete (archive) an agent."""
        self._c._request("DELETE", f"/api/v1/agents/tasks/{agent_id}/")

    # -- run -------------------------------------------------------------------

    def run(
        self,
        agent_id: str,
        *,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Create an execution and start the agent loop.

        Parameters
        ----------
        agent_id:
            Agent UUID.
        idempotency_key:
            Caller-owned durable operation key. Reuse it only for an identical
            retry of this logical run.
        variables:
            Runtime values to substitute into the ``goal`` Jinja2 template.
            Must match the agent's ``input_schema`` when one is set.
        Returns
        -------
        RunDetail
            Newly created run (status will be ``pending``/``running``).
        """
        body: Dict[str, Any] = {"variables": variables or {}}
        data = self._c._request(
            "POST",
            f"/api/v1/agents/tasks/{agent_id}/run-loop/",
            json=body,
            headers=_idempotency_headers(idempotency_key),
        )
        return RunDetail(**data)

    # -- freeze ----------------------------------------------------------------

    def freeze(self, run_id: int) -> Compilation:
        """Freeze a completed run into a reusable Markdown instruction.

        Distills the full chat + tool trace into an ``instruction_md`` that
        captures *what the agent does*. Re-run it later with
        :meth:`Compilations.run_instruction`.
        """
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{run_id}/"
        )
        return Compilation(**data)

    # -- suggest_schema --------------------------------------------------------

    def suggest_schema(
        self,
        *,
        user_prompt: str,
        input_hint: Optional[str] = None,
        output_hint: Optional[str] = None,
        generate_descriptions: bool = False,
    ) -> SchemaSuggestion:
        """Draft ``input_schema`` / ``output_schema`` from a prompt.

        Calls ``POST /api/v1/agents/tasks/suggest-schema/``. Server uses
        Anthropic to infer schemas from ``{{ placeholder }}`` references in
        the prompt and the optional natural-language hints. Does **not**
        persist anything - you have to PATCH the agent yourself.

        Parameters
        ----------
        user_prompt:
            The agent prompt. May contain ``{{ var }}`` placeholders.
        input_hint:
            One-sentence description of expected inputs. If empty and
            ``generate_descriptions`` is ``True``, the server drafts one.
        output_hint:
            Same for outputs.
        generate_descriptions:
            When ``True`` the response includes
            ``input_description`` / ``output_description``.
        """
        body: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "generate_descriptions": generate_descriptions,
        }
        if input_hint is not None:
            body["input_hint"] = input_hint
        if output_hint is not None:
            body["output_hint"] = output_hint
        data = self._c._request(
            "POST", "/api/v1/agents/tasks/suggest-schema/", json=body
        )
        return SchemaSuggestion(**data)

    # -- compile_from_run ------------------------------------------------------

    def compile_from_run(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Freeze a run and wait until the compilation is ready.

        Convenience wrapper: :meth:`freeze` + :meth:`Compilations.wait`.
        Raises :class:`FlyMyAIAgentError` if the compilation ends in
        ``FAILED`` state.
        """
        comp = self.freeze(run_id)
        comp = self._c.compilations.wait(
            comp.id, timeout=timeout, poll_interval=poll_interval
        )
        if comp.status == CompilationStatus.FAILED:
            from flymyai.agents._client import FlyMyAIAgentError

            raise FlyMyAIAgentError(
                f"Compilation {comp.id} failed: {comp.error or '(no error)'}",
                status_code=0,
                response_body=comp.model_dump(),
            )
        return comp


class Runs:
    """Manage agent executions (runs). Maps to ``/api/v1/agents/executions/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def create(
        self,
        *,
        agent_id: str,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Create a new run for the given agent.

        Convenience alias for ``client.agents.run(agent_id, variables=...)``.
        """
        return self._c.agents.run(
            agent_id,
            idempotency_key=idempotency_key,
            variables=variables,
        )

    def list(self) -> List[Run]:
        """List all executions for the current user (newest first)."""
        params: Optional[Dict[str, str]] = None
        results: List[Any] = []
        visited: Set[_PaginationKey] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                "/api/v1/agents/executions/",
                params=params,
            )
            results.extend(_list_results(data))
            params = _guarded_next_list_params(
                data,
                base_params={},
                visited=visited,
                page_count=page_count,
                resource_name="Agent runs",
            )
            if params is None:
                return [Run(**item) for item in results]

    def get(self, run_id: ResourceID) -> RunDetail:
        """Get a single execution with logs."""
        data = self._c._request("GET", f"/api/v1/agents/executions/{run_id}/")
        return RunDetail(**data)

    def status(
        self,
        run_id: ResourceID,
        *,
        since: int = 0,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
    ) -> RunProgressStatus:
        """Get one bounded page of execution progress without heavy run fields."""
        since = _validate_status_since(since)
        page_size = _validate_cursor_page_size(page_size)
        response = self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/status/",
            params={
                "view": "bounded_v1",
                "since": since,
                "page_size": page_size,
            },
        )
        data = _bounded_response_json(
            response,
            resource_name="Run status",
            max_bytes=_RUN_STATUS_MAX_RESPONSE_BYTES,
        )
        result = RunProgressStatus(**data)
        _validate_progress_page(result, since=since, page_size=page_size)
        return result

    def transcript(
        self,
        run_id: ResourceID,
        *,
        cursor: Optional[str] = None,
        page_size: int = 20,
    ) -> RunTranscriptPage:
        """Return one bounded transcript page, latest messages first."""
        cursor = _validate_run_cursor(cursor)
        page_size = _validate_run_page_size(page_size, maximum=50)
        params: Dict[str, Union[str, int]] = {"page_size": page_size}
        if cursor is not None:
            params["cursor"] = cursor
        response = self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/transcript/",
            params=params,
        )
        data = _bounded_response_json(
            response,
            resource_name="Run transcript",
            max_bytes=_RUN_PAGE_MAX_RESPONSE_BYTES,
        )
        page = RunTranscriptPage(**data)
        _validate_transcript_page(page, cursor=cursor, page_size=page_size)
        return page

    def logs(
        self,
        run_id: ResourceID,
        *,
        cursor: Optional[str] = None,
        page_size: int = 50,
    ) -> RunLogPage:
        """Return one bounded page from a fixed execution-log snapshot."""
        cursor = _validate_run_cursor(cursor)
        page_size = _validate_run_page_size(page_size, maximum=100)
        params: Dict[str, Union[str, int]] = {"page_size": page_size}
        if cursor is not None:
            params["cursor"] = cursor
        response = self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/logs/",
            params=params,
        )
        data = _bounded_response_json(
            response,
            resource_name="Run logs",
            max_bytes=_RUN_PAGE_MAX_RESPONSE_BYTES,
        )
        page = RunLogPage(**data)
        _validate_log_page(page, cursor=cursor, page_size=page_size)
        return page

    def resource_range(
        self,
        run_id: ResourceID,
        resource: RunResource,
        *,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> RunResourceRange:
        """Read and verify exactly one range without following server-provided URLs."""
        if not isinstance(resource, RunResource):
            raise ValueError("resource must be a RunResource.")
        _validate_run_resource(resource)
        if resource.size_bytes == 0:
            raise ValueError("An empty resource is complete in its inline projection.")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset < resource.size_bytes
        ):
            raise ValueError("offset must identify a byte inside the resource.")
        remaining = resource.size_bytes - offset
        if length is None:
            requested_length = min(resource.retrieval.max_range_bytes, remaining)
        else:
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or not 1 <= length <= resource.retrieval.max_range_bytes
                or length > remaining
            ):
                raise ValueError(
                    "length must fit the remaining resource and advertised range cap."
                )
            requested_length = length
        end = offset + requested_length - 1
        response = self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/resource/",
            params={"ref": resource.ref},
            headers={"Range": f"bytes={offset}-{end}"},
        )
        return _resource_range_from_response(
            response,
            resource=resource,
            offset=offset,
            length=requested_length,
        )

    def cancel(self, run_id: ResourceID) -> None:
        """Cancel a running execution."""
        self._c._request("POST", f"/api/v1/agents/executions/{run_id}/cancel/")

    def append_message(
        self,
        run_id: ResourceID,
        *,
        text: str,
        effort: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AppendMessageResponse:
        """Append a message and return the backend's bounded acknowledgement."""
        body: Dict[str, Any] = {"text": text}
        if effort is not None:
            body["effort"] = effort
        if model is not None:
            body["model"] = model
        data = self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/append-message/",
            json=body,
        )
        return AppendMessageResponse(**data)

    def suggest_schema(
        self,
        run_id: int,
        *,
        inputs_prompt: Optional[str] = None,
        outputs_prompt: Optional[str] = None,
    ) -> SchemaSuggestion:
        """Draft ``input_schema`` / ``output_schema`` from a finished run.

        Calls ``POST /api/v1/agents/executions/{id}/suggest-schema/``.
        The server uses the execution's chat history + tool trace to infer
        schemas.

        .. warning::
            **Side effect**: the server also saves the resulting schemas
            onto the agent (``user_agent_task.input_schema`` /
            ``output_schema``). Fetch the agent again with
            :meth:`Agents.get` to read them back.

        Parameters
        ----------
        inputs_prompt:
            Optional natural-language hint guiding the input schema draft.
        outputs_prompt:
            Same for outputs.
        """
        body: Dict[str, Any] = {}
        if inputs_prompt is not None:
            body["inputs_prompt"] = inputs_prompt
        if outputs_prompt is not None:
            body["outputs_prompt"] = outputs_prompt
        data = self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/suggest-schema/",
            json=body or None,
        )
        return SchemaSuggestion(**data)

    def wait(
        self,
        run_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Poll until the run reaches a terminal status.

        Parameters
        ----------
        run_id:
            Execution ID.
        timeout:
            Max seconds to wait before raising ``TimeoutError``.
        poll_interval:
            Seconds between polls.
        """
        self.wait_status(
            run_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        # Preserve the established return type. Repeated polls stay bounded;
        # compatibility pays for one full detail response after settlement.
        return self.get(run_id)

    def wait_status(
        self,
        run_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        max_requests: int = _RUN_WAIT_MAX_REQUESTS,
    ) -> RunProgressStatus:
        """Wait using bounded status pages and return the terminal projection."""
        page_size = _validate_cursor_page_size(page_size)
        _validate_wait_options(
            timeout=timeout,
            poll_interval=poll_interval,
            max_requests=max_requests,
        )
        deadline = time.monotonic() + timeout
        since = 0
        for _request_number in range(1, max_requests + 1):
            result = self.status(run_id, since=since, page_size=page_size)
            if result.poll_complete:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Run {run_id} did not complete within {timeout}s "
                    f"(last status: {result.status})"
                )
            if result.has_more:
                since = result.next_since
                continue
            if result.next_since > since:
                since = result.next_since
            time.sleep(poll_interval)
        raise RuntimeError(
            f"Run {run_id} status polling exceeded {max_requests} requests."
        )

    def stream_events(
        self,
        run_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 1.0,
    ):
        """Yield new :class:`ExecutionLog` entries as they appear.

        Polls the execution detail endpoint and yields logs that haven't been
        seen yet.  Stops when the run reaches a terminal status.
        """
        seen_ids: set = set()
        deadline = time.monotonic() + timeout
        while True:
            detail = self.get(run_id)
            for log in detail.logs:
                if log.id not in seen_ids:
                    seen_ids.add(log.id)
                    yield log
            if detail.status in _TERMINAL_STATUSES:
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(poll_interval)


class Tools:
    """Manage MCP tools. Maps to ``/api/v1/agents/tools/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        max_items: Optional[int] = None,
        mcp_tool: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> List[Tool]:
        """List configured connections, optionally stopping at ``max_items``."""
        page_size = _validate_cursor_page_size(page_size)
        max_items = _validate_max_items(max_items)
        request_page_size = min(page_size, max_items or page_size)
        base_params: Dict[str, str] = {}
        if mcp_tool is not None:
            base_params["mcp_tool"] = _validate_slug(
                mcp_tool,
                field_name="mcp_tool",
                max_length=255,
            )
        if alias is not None:
            base_params["alias"] = _validate_alias(alias)
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": request_page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                "/api/v1/agents/tools/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="Configured tools",
            )
            results.extend(page)
            if max_items is not None and len(results) >= max_items:
                return [Tool(**item) for item in results[:max_items]]
            request_page_size = min(
                page_size,
                (max_items - len(results)) if max_items is not None else page_size,
            )
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=request_page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="Configured tools",
                base_params=base_params,
            )
            if next_params is None:
                return [Tool(**item) for item in results]
            params = next_params

    def available(self) -> List[AvailableTool]:
        """List available tools from the catalog (no auth required)."""
        data = self._c._request("GET", "/api/v1/agents/tools/available/")
        return [AvailableTool(**item) for item in data]

    def create(
        self,
        *,
        mcp_tool: str,
        alias: Optional[str] = None,
        **kwargs: Any,
    ) -> Tool:
        """Add one exact tool connection instance to the user's account.

        Omitting ``alias`` preserves the legacy idempotent ``default``
        connection behavior. A non-default alias creates a separate instance.
        """
        body = {"mcp_tool": mcp_tool, **kwargs}
        if alias is not None:
            body["alias"] = _validate_alias(alias)
        data = self._c._request("POST", "/api/v1/agents/tools/", json=body)
        return Tool(**data)

    def get(self, tool_id: int) -> Tool:
        data = self._c._request("GET", f"/api/v1/agents/tools/{tool_id}/")
        return Tool(**data)

    def update(self, tool_id: int, **kwargs: Any) -> Tool:
        """Partial update (PATCH).  Pass ``user_config={...}`` to merge config."""
        if "alias" in kwargs:
            kwargs["alias"] = _validate_alias(kwargs["alias"])
        data = self._c._request(
            "PATCH", f"/api/v1/agents/tools/{tool_id}/", json=kwargs
        )
        return Tool(**data)

    def delete(self, tool_id: int) -> None:
        self._c._request("DELETE", f"/api/v1/agents/tools/{tool_id}/")

    def provide_config(self, tool_id: int, *, user_response: Any) -> Tool:
        """Answer the current ``ask_user`` configuration step."""
        data = self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/provide-config/",
            json={"user_response": user_response},
        )
        return Tool(**data)

    def call(
        self,
        tool_id: int,
        *,
        action: str,
        idempotency_key: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Invoke a tool action with one caller-owned durable identity."""
        data = self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/call/",
            json={"action": action, "arguments": arguments or {}},
            headers=_idempotency_headers(idempotency_key),
        )
        return data


class McpResourceSets:
    """CRUD for named sets of exact MCP connection instances."""

    def __init__(self, client: "SyncAgentClient") -> None:
        self._c = client

    def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        query: Optional[str] = None,
        authority_type: Optional[_McpResourceSetAuthorityTypeLike] = None,
        principal_id: Optional[str] = None,
    ) -> List[McpResourceSetSummary]:
        page_size = _validate_cursor_page_size(page_size)
        base_params: Dict[str, str] = {}
        query = _validate_collection_query(query)
        if query is not None:
            base_params["query"] = query
        normalized_authority_type = (
            _mcp_resource_set_authority_type_value(authority_type)
            if authority_type is not None
            else None
        )
        if normalized_authority_type is not None:
            base_params["authority_type"] = normalized_authority_type
        principal_id = _validate_principal_id_filter(principal_id)
        if (
            principal_id is not None
            and normalized_authority_type == McpResourceSetAuthorityType.OWNER.value
        ):
            raise ValueError(
                "principal_id cannot be combined with authority_type='owner'."
            )
        if principal_id is not None:
            base_params["principal_id"] = principal_id
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                "/api/v1/agents/mcp-resource-sets/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="MCP resource sets",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="MCP resource sets",
                base_params=base_params,
            )
            if next_params is None:
                return [McpResourceSetSummary(**item) for item in results]
            params = next_params

    def list_members(
        self,
        resource_set_id: str,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
    ) -> List[McpResourceSetMember]:
        """List one set's members without embedding them in collection rows."""
        page_size = _validate_cursor_page_size(page_size)
        params: Dict[str, Union[str, int]] = {"page_size": page_size}
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/members/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="MCP resource-set members",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="MCP resource-set members",
            )
            if next_params is None:
                return [McpResourceSetMember(**item) for item in results]
            params = next_params

    def get(self, resource_set_id: str) -> McpResourceSet:
        data = self._c._request(
            "GET",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
        )
        return McpResourceSet(**data)

    def create(
        self,
        *,
        name: str,
        idempotency_key: str,
        description: Optional[str] = None,
        status: Optional[_McpResourceSetStatusLike] = None,
        management_mode: _McpResourceSetManagementModeLike = (
            McpResourceSetManagementMode.FLYMYAI
        ),
        principal_id: Optional[str] = None,
    ) -> McpResourceSet:
        """Create an owner set or an exact external-principal customer set.

        Owner sets use ``management_mode='flymyai'`` without ``principal_id``.
        Customer-managed named mappings require both
        ``management_mode='customer'`` and the exact ``principal_id``.
        Deprecated authority fields are not accepted by this strict signature.
        """
        body = _resource_set_create_payload(
            name=name,
            description=description,
            status=status,
            management_mode=management_mode,
            principal_id=principal_id,
        )
        data = self._c._request(
            "POST",
            "/api/v1/agents/mcp-resource-sets/",
            json=body,
            headers=_idempotency_headers(idempotency_key),
        )
        return McpResourceSet(**data)

    def update(
        self,
        resource_set_id: str,
        *,
        expected_revision: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[_McpResourceSetStatusLike] = None,
    ) -> McpResourceSet:
        """PATCH mutable metadata using the revision originally loaded."""
        body = _resource_set_metadata_payload(
            expected_revision=expected_revision,
            name=name,
            description=description,
            status=status,
        )
        data = self._c._request(
            "PATCH",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
            json=body,
        )
        return McpResourceSet(**data)

    def replace(
        self,
        resource_set_id: str,
        *,
        expected_revision: int,
        name: str,
        description: str = "",
        status: _McpResourceSetStatusLike = McpResourceSetStatus.ACTIVE,
    ) -> McpResourceSet:
        """PUT the complete mutable metadata projection with revision CAS."""
        body = _resource_set_metadata_payload(
            expected_revision=expected_revision,
            name=name,
            description=description,
            status=status,
        )
        data = self._c._request(
            "PUT",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
            json=body,
        )
        return McpResourceSet(**data)

    def delete(self, resource_set_id: str) -> None:
        self._c._request(
            "DELETE",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
        )

    def replace_members(
        self,
        resource_set_id: str,
        *,
        expected_revision: int,
        members: Sequence[_McpResourceSetMemberLike],
    ) -> McpResourceSet:
        """Atomically replace members with compare-and-swap revision safety."""
        expected_revision = _validate_expected_revision(expected_revision)
        data = self._c._request(
            "POST",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/replace-members/",
            json={
                "expected_revision": expected_revision,
                "members": _resource_set_member_payloads(members),
            },
        )
        return McpResourceSet(**data)


class AgentGroups:
    """CRUD for flat agent groups and their atomic assignments."""

    def __init__(self, client: "SyncAgentClient") -> None:
        self._c = client

    def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        query: Optional[str] = None,
    ) -> List[AgentGroup]:
        page_size = _validate_cursor_page_size(page_size)
        base_params: Dict[str, str] = {}
        query = _validate_collection_query(query)
        if query is not None:
            base_params["query"] = query
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                "/api/v1/agents/agent-groups/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="Agent groups",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="Agent groups",
                base_params=base_params,
            )
            if next_params is None:
                return [AgentGroup(**item) for item in results]
            params = next_params

    def get(self, group_id: str) -> AgentGroup:
        data = self._c._request(
            "GET",
            f"/api/v1/agents/agent-groups/{group_id}/",
        )
        return AgentGroup(**data)

    def create(
        self,
        *,
        name: str,
        idempotency_key: str,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        agent_ids: Optional[Sequence[str]] = None,
        resource_set_ids: Optional[Sequence[str]] = None,
    ) -> AgentGroup:
        body: Dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        if is_active is not None:
            body["is_active"] = is_active
        if agent_ids is not None:
            body["agent_ids"] = list(agent_ids)
        if resource_set_ids is not None:
            body["resource_set_ids"] = list(resource_set_ids)
        data = self._c._request(
            "POST",
            "/api/v1/agents/agent-groups/",
            json=body,
            headers=_idempotency_headers(idempotency_key),
        )
        return AgentGroup(**data)

    def update(
        self,
        group_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        agent_ids: Optional[Sequence[str]] = None,
        resource_set_ids: Optional[Sequence[str]] = None,
    ) -> AgentGroup:
        """Patch metadata and atomically replace any supplied assignments."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if is_active is not None:
            body["is_active"] = is_active
        if agent_ids is not None:
            body["agent_ids"] = list(agent_ids)
        if resource_set_ids is not None:
            body["resource_set_ids"] = list(resource_set_ids)
        data = self._c._request(
            "PATCH",
            f"/api/v1/agents/agent-groups/{group_id}/",
            json=body,
        )
        return AgentGroup(**data)

    def replace_assignments(
        self,
        group_id: str,
        *,
        agent_ids: Sequence[str],
        resource_set_ids: Sequence[str],
    ) -> AgentGroup:
        """Atomically replace both agent and resource-set assignments."""
        return self.update(
            group_id,
            agent_ids=agent_ids,
            resource_set_ids=resource_set_ids,
        )

    def delete(self, group_id: str) -> None:
        self._c._request(
            "DELETE",
            f"/api/v1/agents/agent-groups/{group_id}/",
        )


class Compilations:
    """Frozen agent instructions. Maps to ``/api/v1/agents/compilations/``."""

    def __init__(self, client: SyncAgentClient) -> None:
        self._c = client

    def list(self) -> List[Compilation]:
        data = self._c._request("GET", "/api/v1/agents/compilations/")
        return [Compilation(**item) for item in data]

    def get(self, compilation_id: ResourceID) -> Compilation:
        data = self._c._request("GET", f"/api/v1/agents/compilations/{compilation_id}/")
        return Compilation(**data)

    def update(
        self,
        compilation_id: ResourceID,
        *,
        instruction_md: Optional[str] = None,
        cron_schedule: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Compilation:
        """Edit a frozen compilation (PATCH).

        Useful when you want to tweak the Markdown plan by hand after the
        backend froze it. Only allowed when the compilation is already in
        a terminal state (``compiled`` / ``running`` / ``completed``);
        editing during ``pending`` / ``compiling`` raises HTTP 400.

        Parameters
        ----------
        compilation_id:
            ID of the compilation to update.
        instruction_md:
            New Markdown instruction body. Pass a non-blank string.
        cron_schedule:
            New cron expression for scheduled re-runs (or ``""`` to clear).
        timezone:
            IANA timezone name for the cron schedule.
        """
        body: Dict[str, Any] = {}
        if instruction_md is not None:
            body["instruction_md"] = instruction_md
        if cron_schedule is not None:
            body["cron_schedule"] = cron_schedule
        if timezone is not None:
            body["timezone"] = timezone
        data = self._c._request(
            "PATCH",
            f"/api/v1/agents/compilations/{compilation_id}/",
            json=body,
        )
        return Compilation(**data)

    def compile(self, *, execution_id: ResourceID) -> Compilation:
        """Compile an execution into a reusable Python script.

        Note: this is the deterministic replay path (no variables). For
        parametric reuse, prefer :meth:`freeze` + :meth:`run_instruction`.
        """
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/compile/{execution_id}/"
        )
        return Compilation(**data)

    def freeze(self, *, execution_id: ResourceID) -> Compilation:
        """Freeze an execution into a reusable Markdown instruction.

        Alias for :meth:`Agents.freeze`.
        """
        data = self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{execution_id}/"
        )
        return Compilation(**data)

    def run(self, compilation_id: ResourceID) -> Compilation:
        """Reject the legacy keyless compiled-script endpoint before HTTP."""
        del compilation_id
        raise NotImplementedError(_DEPRECATED_COMPILATION_RUN_MESSAGE)

    def run_instruction(
        self,
        compilation_id: int,
        *,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        external_user_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
    ) -> RunDetail:
        """Run a frozen agent from its Markdown instruction.

        Spawns a fresh execution that follows the compiled instruction.
        Pass ``variables`` matching the source agent's ``input_schema``.
        For embedded runs, ``external_user_id`` and ``deployment_id`` must be
        supplied together. ``external_user_id`` is the application's stable,
        non-secret customer ID. ``connections`` is an optional mapping from a
        logical requirement slot to one connection UUID, or to a list of up to
        25 connection UUIDs for a multi-connection slot. ``resource_set_id``
        selects one named mapping owned by the resolved external principal.
        Supply either ``connections`` or ``resource_set_id``, never both.
        Omitting both uses the customer's saved deployment bindings.
        ``idempotency_key`` is required and sent unchanged as the
        ``Idempotency-Key`` HTTP header.
        Raises :class:`VariablesValidationError` on HTTP 400.
        """
        _validate_external_customer_pair(
            external_user_id=external_user_id,
            deployment_id=deployment_id,
        )
        _validate_runtime_resource_selection(
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
        )
        body: Dict[str, Any] = {}
        if variables:
            body["variables"] = variables
        if external_user_id is not None:
            body["external_user_id"] = external_user_id
        if deployment_id is not None:
            body["deployment_id"] = deployment_id
        if connections is not None:
            body["connections"] = _runtime_connections_payload(connections)
        if resource_set_id is not None:
            body["resource_set_id"] = resource_set_id
        if resource_set_revision is not None:
            body["resource_set_revision"] = resource_set_revision
        request_kwargs: Dict[str, Any] = {"json": body or None}
        request_kwargs["headers"] = _idempotency_headers(idempotency_key)
        data = self._c._request(
            "POST",
            f"/api/v1/agents/compilations/{compilation_id}/run-instruction/",
            **request_kwargs,
        )
        return RunDetail(**data)

    def run_instruction_and_wait(
        self,
        compilation_id: int,
        *,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        external_user_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Run an instruction and block until the resulting run finishes.

        Embedded context has the same contract as :meth:`run_instruction`.
        Reuse an idempotency key only for retries of the same logical request.
        """
        run = self.run_instruction(
            compilation_id,
            variables=variables,
            external_user_id=external_user_id,
            deployment_id=deployment_id,
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
            idempotency_key=idempotency_key,
        )
        return self._c.runs.wait(run.id, timeout=timeout, poll_interval=poll_interval)

    def wait(
        self,
        compilation_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Poll until the compilation leaves the ``compiling`` state."""
        deadline = time.monotonic() + timeout
        while True:
            comp = self.get(compilation_id)
            if (
                comp.status != CompilationStatus.COMPILING
                and comp.status != CompilationStatus.PENDING
            ):
                return comp
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Compilation {compilation_id} still {comp.status} after {timeout}s"
                )
            time.sleep(poll_interval)


class Versions:
    """Read immutable frozen versions for publishing."""

    def __init__(self, client: "SyncAgentClient") -> None:
        self._c = client

    def list(self, *, agent_id: Optional[str] = None) -> List[AgentVersion]:
        base_params = {"agent_task": agent_id} if agent_id is not None else {}
        params = base_params or None
        results: List[Any] = []
        visited: Set[_PaginationKey] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                "/api/v1/agents/versions/",
                params=params,
            )
            results.extend(_list_results(data))
            params = _guarded_next_list_params(
                data,
                base_params=base_params,
                visited=visited,
                page_count=page_count,
                resource_name="Agent versions",
            )
            if params is None:
                return [AgentVersion(**item) for item in results]

    def get(self, version_id: str) -> AgentVersion:
        data = self._c._request(
            "GET",
            f"/api/v1/agents/versions/{version_id}/",
        )
        return AgentVersion(**data)


class Deployments:
    """Publish and run frozen versions for embedded customers."""

    def __init__(self, client: "SyncAgentClient") -> None:
        self._c = client

    def list(
        self,
        *,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AgentDeployment]:
        base_params = {
            key: value
            for key, value in {
                "agent_task": agent_id,
                "status": status,
            }.items()
            if value is not None
        }
        page_params = base_params or None
        results: List[Any] = []
        visited: Set[_PaginationKey] = set()
        page_count = 0
        while True:
            page_count += 1
            data = self._c._request(
                "GET",
                "/api/v1/agents/deployments/",
                params=page_params,
            )
            results.extend(_list_results(data))
            page_params = _guarded_next_list_params(
                data,
                base_params=base_params,
                visited=visited,
                page_count=page_count,
                resource_name="Agent deployments",
            )
            if page_params is None:
                return [AgentDeployment(**item) for item in results]

    def get(self, deployment_id: str) -> AgentDeployment:
        data = self._c._request(
            "GET",
            f"/api/v1/agents/deployments/{deployment_id}/",
        )
        return AgentDeployment(**data)

    def create(
        self,
        *,
        agent_id: str,
        version_id: Optional[str],
        name: str = "Production",
        status: str = "draft",
        publish_mode: str = "embedded",
    ) -> AgentDeployment:
        body: Dict[str, Any] = {
            "agent_task": agent_id,
            "candidate_version": version_id,
            "name": name,
            "status": status,
            "publish_mode": publish_mode,
        }
        data = self._c._request(
            "POST",
            "/api/v1/agents/deployments/",
            json=body,
        )
        return AgentDeployment(**data)

    def update(self, deployment_id: str, **kwargs: Any) -> AgentDeployment:
        data = self._c._request(
            "PATCH",
            f"/api/v1/agents/deployments/{deployment_id}/",
            json=kwargs,
        )
        return AgentDeployment(**data)

    def publish(
        self,
        deployment_id: str,
        *,
        publish_mode: str = "embedded",
        version_id: Optional[str] = None,
    ) -> AgentDeployment:
        body: Dict[str, Any] = {"publish_mode": publish_mode}
        if version_id is not None:
            body["candidate_version"] = version_id
        data = self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/publish/",
            json=body,
        )
        return AgentDeployment(**data)

    def preflight(
        self,
        deployment_id: str,
        *,
        publish_mode: str = "embedded",
        version_id: Optional[str] = None,
    ) -> AgentDeploymentPreflight:
        body: Dict[str, Any] = {"publish_mode": publish_mode}
        if version_id is not None:
            body["candidate_version"] = version_id
        data = self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/preflight/",
            json=body,
        )
        return AgentDeploymentPreflight(**data)

    def access(
        self,
        deployment_id: str,
        *,
        external_user_id: Optional[str] = None,
    ) -> AgentDeploymentAccess:
        params = (
            {"external_user_id": external_user_id}
            if external_user_id is not None
            else None
        )
        data = self._c._request(
            "GET",
            f"/api/v1/agents/deployments/{deployment_id}/access/",
            params=params,
        )
        return AgentDeploymentAccess(**data)

    def create_connection_link(
        self,
        deployment_id: str,
        *,
        external_user_id: str,
        slot: str,
        alias: Optional[str] = None,
    ) -> AgentConnectionSession:
        slot = _validate_slot(slot)
        body: Dict[str, Any] = {
            "external_user_id": external_user_id,
            "slot": slot,
        }
        if alias is not None:
            body["alias"] = _validate_alias(alias)
        data = self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/connect-session/",
            json=body,
        )
        return AgentConnectionSession(**data)

    def run(
        self,
        deployment_id: str,
        *,
        external_user_id: str,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
    ) -> RunDetail:
        """Run the deployment's active frozen version for one customer.

        ``deployment_id`` is the deployment public UUID and remains stable when
        a newer immutable version is published. ``external_user_id`` is your
        application's stable, non-secret customer ID. ``connections`` can
        override saved bindings for this run by mapping a logical slot to one
        FlyMyAI connection UUID, or to up to 25 UUIDs for a multi-connection
        slot. Alternatively, ``resource_set_id`` selects one named mapping
        belonging to this exact external principal. ``resource_set_revision``
        adds compare-and-swap protection against a stale customer mapping.
        Reuse ``idempotency_key`` only when retrying the same request.
        """
        _validate_runtime_resource_selection(
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
        )
        body: Dict[str, Any] = {"external_user_id": external_user_id}
        if variables is not None:
            body["variables"] = variables
        if connections is not None:
            body["connections"] = _runtime_connections_payload(connections)
        if resource_set_id is not None:
            body["resource_set_id"] = resource_set_id
        if resource_set_revision is not None:
            body["resource_set_revision"] = resource_set_revision
        request_kwargs: Dict[str, Any] = {"json": body}
        request_kwargs["headers"] = _idempotency_headers(idempotency_key)
        data = self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/run/",
            **request_kwargs,
        )
        return RunDetail(**data)

    def run_and_wait(
        self,
        deployment_id: str,
        *,
        external_user_id: str,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Run a deployment and block until its execution finishes."""
        run = self.run(
            deployment_id,
            external_user_id=external_user_id,
            variables=variables,
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
            idempotency_key=idempotency_key,
        )
        return self._c.runs.wait(
            run.id,
            timeout=timeout,
            poll_interval=poll_interval,
        )


class AsyncAgents:
    """Async variant of :class:`Agents`."""

    def __init__(self, client: "AsyncAgentClient") -> None:
        self._c = client

    async def create(
        self,
        *,
        name: str,
        goal: str,
        tools: Optional[List[int]] = None,
        mcp_servers: Optional[List[int]] = None,
        mcp_resource_set_ids: Optional[List[str]] = None,
        mcp_access_mode: Optional[_McpAccessModeLike] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        input_description: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        output_description: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Agent:
        """Async variant of :meth:`Agents.create`. Same parameters."""
        body: Dict[str, Any] = {"name": name, "user_prompt": goal}
        if tools is not None:
            body["available_tools"] = tools
        if mcp_servers is not None:
            body["available_custom_mcp_servers"] = mcp_servers
        if mcp_resource_set_ids is not None:
            body["mcp_resource_set_ids"] = mcp_resource_set_ids
        if mcp_access_mode is not None:
            body["mcp_access_mode"] = _mcp_access_mode_value(mcp_access_mode)
        if input_schema is not None:
            body["input_schema"] = input_schema
        if input_description is not None:
            body["input_description"] = input_description
        if output_schema is not None:
            body["output_schema"] = output_schema
        if output_description is not None:
            body["output_description"] = output_description
        if status is not None:
            body["status"] = status
        data = await self._c._request("POST", "/api/v1/agents/tasks/", json=body)
        return Agent(**data)

    async def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
    ) -> List[Agent]:
        """Return agents through bounded cursor pages."""
        page_size = _validate_cursor_page_size(page_size)
        base_params = {"pagination": "cursor"}
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                "/api/v1/agents/tasks/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="Agents",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="Agents",
                base_params=base_params,
            )
            if next_params is None:
                return [Agent(**item) for item in results]
            params = next_params

    async def get(self, agent_id: str) -> AgentDetail:
        data = await self._c._request("GET", f"/api/v1/agents/tasks/{agent_id}/")
        return AgentDetail(**data)

    async def update(
        self,
        agent_id: str,
        *,
        mcp_access_mode: Optional[_McpAccessModeLike] = None,
        **kwargs: Any,
    ) -> Agent:
        if "goal" in kwargs:
            kwargs["user_prompt"] = kwargs.pop("goal")
        if mcp_access_mode is not None:
            kwargs["mcp_access_mode"] = _mcp_access_mode_value(mcp_access_mode)
        data = await self._c._request(
            "PATCH", f"/api/v1/agents/tasks/{agent_id}/", json=kwargs
        )
        return Agent(**data)

    async def delete(self, agent_id: str) -> None:
        await self._c._request("DELETE", f"/api/v1/agents/tasks/{agent_id}/")

    async def run(
        self,
        agent_id: str,
        *,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        body: Dict[str, Any] = {"variables": variables or {}}
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/tasks/{agent_id}/run-loop/",
            json=body,
            headers=_idempotency_headers(idempotency_key),
        )
        return RunDetail(**data)

    async def freeze(self, run_id: int) -> Compilation:
        """Freeze a completed run into a reusable instruction."""
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{run_id}/"
        )
        return Compilation(**data)

    async def suggest_schema(
        self,
        *,
        user_prompt: str,
        input_hint: Optional[str] = None,
        output_hint: Optional[str] = None,
        generate_descriptions: bool = False,
    ) -> SchemaSuggestion:
        """Async variant of :meth:`Agents.suggest_schema`."""
        body: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "generate_descriptions": generate_descriptions,
        }
        if input_hint is not None:
            body["input_hint"] = input_hint
        if output_hint is not None:
            body["output_hint"] = output_hint
        data = await self._c._request(
            "POST", "/api/v1/agents/tasks/suggest-schema/", json=body
        )
        return SchemaSuggestion(**data)

    async def compile_from_run(
        self,
        run_id: int,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Async variant of :meth:`Agents.compile_from_run`."""
        comp = await self.freeze(run_id)
        comp = await self._c.compilations.wait(
            comp.id, timeout=timeout, poll_interval=poll_interval
        )
        if comp.status == CompilationStatus.FAILED:
            from flymyai.agents._client import FlyMyAIAgentError

            raise FlyMyAIAgentError(
                f"Compilation {comp.id} failed: {comp.error or '(no error)'}",
                status_code=0,
                response_body=comp.model_dump(),
            )
        return comp


class AsyncRuns:
    """Async variant of :class:`Runs`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def create(
        self,
        *,
        agent_id: str,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> RunDetail:
        """Create a new run for the given agent (async)."""
        return await self._c.agents.run(
            agent_id,
            idempotency_key=idempotency_key,
            variables=variables,
        )

    async def list(self) -> List[Run]:
        params: Optional[Dict[str, str]] = None
        results: List[Any] = []
        visited: Set[_PaginationKey] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                "/api/v1/agents/executions/",
                params=params,
            )
            results.extend(_list_results(data))
            params = _guarded_next_list_params(
                data,
                base_params={},
                visited=visited,
                page_count=page_count,
                resource_name="Agent runs",
            )
            if params is None:
                return [Run(**item) for item in results]

    async def get(self, run_id: ResourceID) -> RunDetail:
        data = await self._c._request("GET", f"/api/v1/agents/executions/{run_id}/")
        return RunDetail(**data)

    async def status(
        self,
        run_id: ResourceID,
        *,
        since: int = 0,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
    ) -> RunProgressStatus:
        since = _validate_status_since(since)
        page_size = _validate_cursor_page_size(page_size)
        response = await self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/status/",
            params={
                "view": "bounded_v1",
                "since": since,
                "page_size": page_size,
            },
        )
        data = _bounded_response_json(
            response,
            resource_name="Run status",
            max_bytes=_RUN_STATUS_MAX_RESPONSE_BYTES,
        )
        result = RunProgressStatus(**data)
        _validate_progress_page(result, since=since, page_size=page_size)
        return result

    async def transcript(
        self,
        run_id: ResourceID,
        *,
        cursor: Optional[str] = None,
        page_size: int = 20,
    ) -> RunTranscriptPage:
        cursor = _validate_run_cursor(cursor)
        page_size = _validate_run_page_size(page_size, maximum=50)
        params: Dict[str, Union[str, int]] = {"page_size": page_size}
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/transcript/",
            params=params,
        )
        data = _bounded_response_json(
            response,
            resource_name="Run transcript",
            max_bytes=_RUN_PAGE_MAX_RESPONSE_BYTES,
        )
        page = RunTranscriptPage(**data)
        _validate_transcript_page(page, cursor=cursor, page_size=page_size)
        return page

    async def logs(
        self,
        run_id: ResourceID,
        *,
        cursor: Optional[str] = None,
        page_size: int = 50,
    ) -> RunLogPage:
        cursor = _validate_run_cursor(cursor)
        page_size = _validate_run_page_size(page_size, maximum=100)
        params: Dict[str, Union[str, int]] = {"page_size": page_size}
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/logs/",
            params=params,
        )
        data = _bounded_response_json(
            response,
            resource_name="Run logs",
            max_bytes=_RUN_PAGE_MAX_RESPONSE_BYTES,
        )
        page = RunLogPage(**data)
        _validate_log_page(page, cursor=cursor, page_size=page_size)
        return page

    async def resource_range(
        self,
        run_id: ResourceID,
        resource: RunResource,
        *,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> RunResourceRange:
        if not isinstance(resource, RunResource):
            raise ValueError("resource must be a RunResource.")
        _validate_run_resource(resource)
        if resource.size_bytes == 0:
            raise ValueError("An empty resource is complete in its inline projection.")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset < resource.size_bytes
        ):
            raise ValueError("offset must identify a byte inside the resource.")
        remaining = resource.size_bytes - offset
        if length is None:
            requested_length = min(resource.retrieval.max_range_bytes, remaining)
        else:
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or not 1 <= length <= resource.retrieval.max_range_bytes
                or length > remaining
            ):
                raise ValueError(
                    "length must fit the remaining resource and advertised range cap."
                )
            requested_length = length
        end = offset + requested_length - 1
        response = await self._c._raw_request(
            "GET",
            f"/api/v1/agents/executions/{run_id}/resource/",
            params={"ref": resource.ref},
            headers={"Range": f"bytes={offset}-{end}"},
        )
        return _resource_range_from_response(
            response,
            resource=resource,
            offset=offset,
            length=requested_length,
        )

    async def cancel(self, run_id: ResourceID) -> None:
        await self._c._request("POST", f"/api/v1/agents/executions/{run_id}/cancel/")

    async def append_message(
        self,
        run_id: ResourceID,
        *,
        text: str,
        effort: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AppendMessageResponse:
        body: Dict[str, Any] = {"text": text}
        if effort is not None:
            body["effort"] = effort
        if model is not None:
            body["model"] = model
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/append-message/",
            json=body,
        )
        return AppendMessageResponse(**data)

    async def suggest_schema(
        self,
        run_id: int,
        *,
        inputs_prompt: Optional[str] = None,
        outputs_prompt: Optional[str] = None,
    ) -> SchemaSuggestion:
        """Async variant of :meth:`Runs.suggest_schema`.

        .. warning::
            Also saves the resulting schemas onto the source agent.
        """
        body: Dict[str, Any] = {}
        if inputs_prompt is not None:
            body["inputs_prompt"] = inputs_prompt
        if outputs_prompt is not None:
            body["outputs_prompt"] = outputs_prompt
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/executions/{run_id}/suggest-schema/",
            json=body or None,
        )
        return SchemaSuggestion(**data)

    async def wait(
        self,
        run_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        await self.wait_status(
            run_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )
        return await self.get(run_id)

    async def wait_status(
        self,
        run_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        max_requests: int = _RUN_WAIT_MAX_REQUESTS,
    ) -> RunProgressStatus:
        page_size = _validate_cursor_page_size(page_size)
        _validate_wait_options(
            timeout=timeout,
            poll_interval=poll_interval,
            max_requests=max_requests,
        )
        deadline = time.monotonic() + timeout
        since = 0
        for _request_number in range(1, max_requests + 1):
            result = await self.status(run_id, since=since, page_size=page_size)
            if result.poll_complete:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Run {run_id} did not complete within {timeout}s "
                    f"(last status: {result.status})"
                )
            if result.has_more:
                since = result.next_since
                continue
            if result.next_since > since:
                since = result.next_since
            await asyncio.sleep(poll_interval)
        raise RuntimeError(
            f"Run {run_id} status polling exceeded {max_requests} requests."
        )

    async def stream_events(
        self,
        run_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 1.0,
    ):
        seen_ids: set = set()
        deadline = time.monotonic() + timeout
        while True:
            detail = await self.get(run_id)
            for log in detail.logs:
                if log.id not in seen_ids:
                    seen_ids.add(log.id)
                    yield log
            if detail.status in _TERMINAL_STATUSES:
                return
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(poll_interval)


class AsyncTools:
    """Async variant of :class:`Tools`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        max_items: Optional[int] = None,
        mcp_tool: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> List[Tool]:
        page_size = _validate_cursor_page_size(page_size)
        max_items = _validate_max_items(max_items)
        request_page_size = min(page_size, max_items or page_size)
        base_params: Dict[str, str] = {}
        if mcp_tool is not None:
            base_params["mcp_tool"] = _validate_slug(
                mcp_tool,
                field_name="mcp_tool",
                max_length=255,
            )
        if alias is not None:
            base_params["alias"] = _validate_alias(alias)
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": request_page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                "/api/v1/agents/tools/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="Configured tools",
            )
            results.extend(page)
            if max_items is not None and len(results) >= max_items:
                return [Tool(**item) for item in results[:max_items]]
            request_page_size = min(
                page_size,
                (max_items - len(results)) if max_items is not None else page_size,
            )
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=request_page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="Configured tools",
                base_params=base_params,
            )
            if next_params is None:
                return [Tool(**item) for item in results]
            params = next_params

    async def available(self) -> List[AvailableTool]:
        data = await self._c._request("GET", "/api/v1/agents/tools/available/")
        return [AvailableTool(**item) for item in data]

    async def create(
        self,
        *,
        mcp_tool: str,
        alias: Optional[str] = None,
        **kwargs: Any,
    ) -> Tool:
        body = {"mcp_tool": mcp_tool, **kwargs}
        if alias is not None:
            body["alias"] = _validate_alias(alias)
        data = await self._c._request("POST", "/api/v1/agents/tools/", json=body)
        return Tool(**data)

    async def get(self, tool_id: int) -> Tool:
        data = await self._c._request("GET", f"/api/v1/agents/tools/{tool_id}/")
        return Tool(**data)

    async def update(self, tool_id: int, **kwargs: Any) -> Tool:
        if "alias" in kwargs:
            kwargs["alias"] = _validate_alias(kwargs["alias"])
        data = await self._c._request(
            "PATCH", f"/api/v1/agents/tools/{tool_id}/", json=kwargs
        )
        return Tool(**data)

    async def delete(self, tool_id: int) -> None:
        await self._c._request("DELETE", f"/api/v1/agents/tools/{tool_id}/")

    async def provide_config(self, tool_id: int, *, user_response: Any) -> Tool:
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/provide-config/",
            json={"user_response": user_response},
        )
        return Tool(**data)

    async def call(
        self,
        tool_id: int,
        *,
        action: str,
        idempotency_key: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/tools/{tool_id}/call/",
            json={"action": action, "arguments": arguments or {}},
            headers=_idempotency_headers(idempotency_key),
        )
        return data


class AsyncMcpResourceSets:
    """Async variant of :class:`McpResourceSets`."""

    def __init__(self, client: "AsyncAgentClient") -> None:
        self._c = client

    async def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        query: Optional[str] = None,
        authority_type: Optional[_McpResourceSetAuthorityTypeLike] = None,
        principal_id: Optional[str] = None,
    ) -> List[McpResourceSetSummary]:
        page_size = _validate_cursor_page_size(page_size)
        base_params: Dict[str, str] = {}
        query = _validate_collection_query(query)
        if query is not None:
            base_params["query"] = query
        normalized_authority_type = (
            _mcp_resource_set_authority_type_value(authority_type)
            if authority_type is not None
            else None
        )
        if normalized_authority_type is not None:
            base_params["authority_type"] = normalized_authority_type
        principal_id = _validate_principal_id_filter(principal_id)
        if (
            principal_id is not None
            and normalized_authority_type == McpResourceSetAuthorityType.OWNER.value
        ):
            raise ValueError(
                "principal_id cannot be combined with authority_type='owner'."
            )
        if principal_id is not None:
            base_params["principal_id"] = principal_id
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                "/api/v1/agents/mcp-resource-sets/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="MCP resource sets",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="MCP resource sets",
                base_params=base_params,
            )
            if next_params is None:
                return [McpResourceSetSummary(**item) for item in results]
            params = next_params

    async def list_members(
        self,
        resource_set_id: str,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
    ) -> List[McpResourceSetMember]:
        page_size = _validate_cursor_page_size(page_size)
        params: Dict[str, Union[str, int]] = {"page_size": page_size}
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/members/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="MCP resource-set members",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="MCP resource-set members",
            )
            if next_params is None:
                return [McpResourceSetMember(**item) for item in results]
            params = next_params

    async def get(self, resource_set_id: str) -> McpResourceSet:
        data = await self._c._request(
            "GET",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
        )
        return McpResourceSet(**data)

    async def create(
        self,
        *,
        name: str,
        idempotency_key: str,
        description: Optional[str] = None,
        status: Optional[_McpResourceSetStatusLike] = None,
        management_mode: _McpResourceSetManagementModeLike = (
            McpResourceSetManagementMode.FLYMYAI
        ),
        principal_id: Optional[str] = None,
    ) -> McpResourceSet:
        body = _resource_set_create_payload(
            name=name,
            description=description,
            status=status,
            management_mode=management_mode,
            principal_id=principal_id,
        )
        data = await self._c._request(
            "POST",
            "/api/v1/agents/mcp-resource-sets/",
            json=body,
            headers=_idempotency_headers(idempotency_key),
        )
        return McpResourceSet(**data)

    async def update(
        self,
        resource_set_id: str,
        *,
        expected_revision: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[_McpResourceSetStatusLike] = None,
    ) -> McpResourceSet:
        body = _resource_set_metadata_payload(
            expected_revision=expected_revision,
            name=name,
            description=description,
            status=status,
        )
        data = await self._c._request(
            "PATCH",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
            json=body,
        )
        return McpResourceSet(**data)

    async def replace(
        self,
        resource_set_id: str,
        *,
        expected_revision: int,
        name: str,
        description: str = "",
        status: _McpResourceSetStatusLike = McpResourceSetStatus.ACTIVE,
    ) -> McpResourceSet:
        body = _resource_set_metadata_payload(
            expected_revision=expected_revision,
            name=name,
            description=description,
            status=status,
        )
        data = await self._c._request(
            "PUT",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
            json=body,
        )
        return McpResourceSet(**data)

    async def delete(self, resource_set_id: str) -> None:
        await self._c._request(
            "DELETE",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/",
        )

    async def replace_members(
        self,
        resource_set_id: str,
        *,
        expected_revision: int,
        members: Sequence[_McpResourceSetMemberLike],
    ) -> McpResourceSet:
        expected_revision = _validate_expected_revision(expected_revision)
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/mcp-resource-sets/{resource_set_id}/replace-members/",
            json={
                "expected_revision": expected_revision,
                "members": _resource_set_member_payloads(members),
            },
        )
        return McpResourceSet(**data)


class AsyncAgentGroups:
    """Async variant of :class:`AgentGroups`."""

    def __init__(self, client: "AsyncAgentClient") -> None:
        self._c = client

    async def list(
        self,
        *,
        page_size: int = _DEFAULT_CURSOR_PAGE_SIZE,
        query: Optional[str] = None,
    ) -> List[AgentGroup]:
        page_size = _validate_cursor_page_size(page_size)
        base_params: Dict[str, str] = {}
        query = _validate_collection_query(query)
        if query is not None:
            base_params["query"] = query
        params: Dict[str, Union[str, int]] = {
            **base_params,
            "page_size": page_size,
        }
        results: List[Any] = []
        visited: Set[str] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                "/api/v1/agents/agent-groups/",
                params=params,
            )
            page, next_cursor = _compact_cursor_page(
                data,
                resource_name="Agent groups",
            )
            results.extend(page)
            next_params = _next_compact_cursor_params(
                next_cursor,
                page_size=page_size,
                visited=visited,
                page_count=page_count,
                row_count=len(results),
                resource_name="Agent groups",
                base_params=base_params,
            )
            if next_params is None:
                return [AgentGroup(**item) for item in results]
            params = next_params

    async def get(self, group_id: str) -> AgentGroup:
        data = await self._c._request(
            "GET",
            f"/api/v1/agents/agent-groups/{group_id}/",
        )
        return AgentGroup(**data)

    async def create(
        self,
        *,
        name: str,
        idempotency_key: str,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        agent_ids: Optional[Sequence[str]] = None,
        resource_set_ids: Optional[Sequence[str]] = None,
    ) -> AgentGroup:
        body: Dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        if is_active is not None:
            body["is_active"] = is_active
        if agent_ids is not None:
            body["agent_ids"] = list(agent_ids)
        if resource_set_ids is not None:
            body["resource_set_ids"] = list(resource_set_ids)
        data = await self._c._request(
            "POST",
            "/api/v1/agents/agent-groups/",
            json=body,
            headers=_idempotency_headers(idempotency_key),
        )
        return AgentGroup(**data)

    async def update(
        self,
        group_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        agent_ids: Optional[Sequence[str]] = None,
        resource_set_ids: Optional[Sequence[str]] = None,
    ) -> AgentGroup:
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if is_active is not None:
            body["is_active"] = is_active
        if agent_ids is not None:
            body["agent_ids"] = list(agent_ids)
        if resource_set_ids is not None:
            body["resource_set_ids"] = list(resource_set_ids)
        data = await self._c._request(
            "PATCH",
            f"/api/v1/agents/agent-groups/{group_id}/",
            json=body,
        )
        return AgentGroup(**data)

    async def replace_assignments(
        self,
        group_id: str,
        *,
        agent_ids: Sequence[str],
        resource_set_ids: Sequence[str],
    ) -> AgentGroup:
        return await self.update(
            group_id,
            agent_ids=agent_ids,
            resource_set_ids=resource_set_ids,
        )

    async def delete(self, group_id: str) -> None:
        await self._c._request(
            "DELETE",
            f"/api/v1/agents/agent-groups/{group_id}/",
        )


class AsyncCompilations:
    """Async variant of :class:`Compilations`."""

    def __init__(self, client: AsyncAgentClient) -> None:
        self._c = client

    async def list(self) -> List[Compilation]:
        data = await self._c._request("GET", "/api/v1/agents/compilations/")
        return [Compilation(**item) for item in data]

    async def get(self, compilation_id: ResourceID) -> Compilation:
        data = await self._c._request(
            "GET", f"/api/v1/agents/compilations/{compilation_id}/"
        )
        return Compilation(**data)

    async def update(
        self,
        compilation_id: ResourceID,
        *,
        instruction_md: Optional[str] = None,
        cron_schedule: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Compilation:
        """Async variant of :meth:`Compilations.update`."""
        body: Dict[str, Any] = {}
        if instruction_md is not None:
            body["instruction_md"] = instruction_md
        if cron_schedule is not None:
            body["cron_schedule"] = cron_schedule
        if timezone is not None:
            body["timezone"] = timezone
        data = await self._c._request(
            "PATCH",
            f"/api/v1/agents/compilations/{compilation_id}/",
            json=body,
        )
        return Compilation(**data)

    async def compile(self, *, execution_id: ResourceID) -> Compilation:
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/compile/{execution_id}/"
        )
        return Compilation(**data)

    async def freeze(self, *, execution_id: ResourceID) -> Compilation:
        """Freeze an execution into a reusable Markdown instruction."""
        data = await self._c._request(
            "POST", f"/api/v1/agents/compilations/freeze-instruction/{execution_id}/"
        )
        return Compilation(**data)

    async def run(self, compilation_id: ResourceID) -> Compilation:
        """Reject the legacy keyless compiled-script endpoint before HTTP."""
        del compilation_id
        raise NotImplementedError(_DEPRECATED_COMPILATION_RUN_MESSAGE)

    async def run_instruction(
        self,
        compilation_id: int,
        *,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        external_user_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
    ) -> RunDetail:
        """Run a frozen agent from its Markdown instruction.

        For embedded runs, ``external_user_id`` and ``deployment_id`` must be
        supplied together. ``connections`` maps each logical slot to one
        connection UUID, or to a list of up to 25 UUIDs for a multi-connection
        slot. ``resource_set_id`` selects one named mapping for the resolved
        external principal. Supply it or ``connections``, never both. Omitting
        both uses the customer's saved deployment bindings.
        ``idempotency_key`` is safe to reuse only for an identical retry.
        Raises :class:`VariablesValidationError` on HTTP 400.
        """
        _validate_external_customer_pair(
            external_user_id=external_user_id,
            deployment_id=deployment_id,
        )
        _validate_runtime_resource_selection(
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
        )
        body: Dict[str, Any] = {}
        if variables:
            body["variables"] = variables
        if external_user_id is not None:
            body["external_user_id"] = external_user_id
        if deployment_id is not None:
            body["deployment_id"] = deployment_id
        if connections is not None:
            body["connections"] = _runtime_connections_payload(connections)
        if resource_set_id is not None:
            body["resource_set_id"] = resource_set_id
        if resource_set_revision is not None:
            body["resource_set_revision"] = resource_set_revision
        request_kwargs: Dict[str, Any] = {"json": body or None}
        request_kwargs["headers"] = _idempotency_headers(idempotency_key)
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/compilations/{compilation_id}/run-instruction/",
            **request_kwargs,
        )
        return RunDetail(**data)

    async def run_instruction_and_wait(
        self,
        compilation_id: int,
        *,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        external_user_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Run an instruction and await the resulting run.

        Embedded context has the same contract as :meth:`run_instruction`.
        """
        run = await self.run_instruction(
            compilation_id,
            variables=variables,
            external_user_id=external_user_id,
            deployment_id=deployment_id,
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
            idempotency_key=idempotency_key,
        )
        return await self._c.runs.wait(
            run.id, timeout=timeout, poll_interval=poll_interval
        )

    async def wait(
        self,
        compilation_id: ResourceID,
        *,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> Compilation:
        """Poll until the compilation leaves the ``compiling`` state."""
        deadline = time.monotonic() + timeout
        while True:
            comp = await self.get(compilation_id)
            if (
                comp.status != CompilationStatus.COMPILING
                and comp.status != CompilationStatus.PENDING
            ):
                return comp
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Compilation {compilation_id} still {comp.status} after {timeout}s"
                )
            await asyncio.sleep(poll_interval)


class AsyncVersions:
    """Async variant of :class:`Versions`."""

    def __init__(self, client: "AsyncAgentClient") -> None:
        self._c = client

    async def list(
        self,
        *,
        agent_id: Optional[str] = None,
    ) -> List[AgentVersion]:
        base_params = {"agent_task": agent_id} if agent_id is not None else {}
        params = base_params or None
        results: List[Any] = []
        visited: Set[_PaginationKey] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                "/api/v1/agents/versions/",
                params=params,
            )
            results.extend(_list_results(data))
            params = _guarded_next_list_params(
                data,
                base_params=base_params,
                visited=visited,
                page_count=page_count,
                resource_name="Agent versions",
            )
            if params is None:
                return [AgentVersion(**item) for item in results]

    async def get(self, version_id: str) -> AgentVersion:
        data = await self._c._request(
            "GET",
            f"/api/v1/agents/versions/{version_id}/",
        )
        return AgentVersion(**data)


class AsyncDeployments:
    """Async variant of :class:`Deployments`."""

    def __init__(self, client: "AsyncAgentClient") -> None:
        self._c = client

    async def list(
        self,
        *,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AgentDeployment]:
        base_params = {
            key: value
            for key, value in {
                "agent_task": agent_id,
                "status": status,
            }.items()
            if value is not None
        }
        page_params = base_params or None
        results: List[Any] = []
        visited: Set[_PaginationKey] = set()
        page_count = 0
        while True:
            page_count += 1
            data = await self._c._request(
                "GET",
                "/api/v1/agents/deployments/",
                params=page_params,
            )
            results.extend(_list_results(data))
            page_params = _guarded_next_list_params(
                data,
                base_params=base_params,
                visited=visited,
                page_count=page_count,
                resource_name="Agent deployments",
            )
            if page_params is None:
                return [AgentDeployment(**item) for item in results]

    async def get(self, deployment_id: str) -> AgentDeployment:
        data = await self._c._request(
            "GET",
            f"/api/v1/agents/deployments/{deployment_id}/",
        )
        return AgentDeployment(**data)

    async def create(
        self,
        *,
        agent_id: str,
        version_id: Optional[str],
        name: str = "Production",
        status: str = "draft",
        publish_mode: str = "embedded",
    ) -> AgentDeployment:
        body: Dict[str, Any] = {
            "agent_task": agent_id,
            "candidate_version": version_id,
            "name": name,
            "status": status,
            "publish_mode": publish_mode,
        }
        data = await self._c._request(
            "POST",
            "/api/v1/agents/deployments/",
            json=body,
        )
        return AgentDeployment(**data)

    async def update(
        self,
        deployment_id: str,
        **kwargs: Any,
    ) -> AgentDeployment:
        data = await self._c._request(
            "PATCH",
            f"/api/v1/agents/deployments/{deployment_id}/",
            json=kwargs,
        )
        return AgentDeployment(**data)

    async def publish(
        self,
        deployment_id: str,
        *,
        publish_mode: str = "embedded",
        version_id: Optional[str] = None,
    ) -> AgentDeployment:
        body: Dict[str, Any] = {"publish_mode": publish_mode}
        if version_id is not None:
            body["candidate_version"] = version_id
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/publish/",
            json=body,
        )
        return AgentDeployment(**data)

    async def preflight(
        self,
        deployment_id: str,
        *,
        publish_mode: str = "embedded",
        version_id: Optional[str] = None,
    ) -> AgentDeploymentPreflight:
        body: Dict[str, Any] = {"publish_mode": publish_mode}
        if version_id is not None:
            body["candidate_version"] = version_id
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/preflight/",
            json=body,
        )
        return AgentDeploymentPreflight(**data)

    async def access(
        self,
        deployment_id: str,
        *,
        external_user_id: Optional[str] = None,
    ) -> AgentDeploymentAccess:
        params = (
            {"external_user_id": external_user_id}
            if external_user_id is not None
            else None
        )
        data = await self._c._request(
            "GET",
            f"/api/v1/agents/deployments/{deployment_id}/access/",
            params=params,
        )
        return AgentDeploymentAccess(**data)

    async def create_connection_link(
        self,
        deployment_id: str,
        *,
        external_user_id: str,
        slot: str,
        alias: Optional[str] = None,
    ) -> AgentConnectionSession:
        slot = _validate_slot(slot)
        body: Dict[str, Any] = {
            "external_user_id": external_user_id,
            "slot": slot,
        }
        if alias is not None:
            body["alias"] = _validate_alias(alias)
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/connect-session/",
            json=body,
        )
        return AgentConnectionSession(**data)

    async def run(
        self,
        deployment_id: str,
        *,
        external_user_id: str,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
    ) -> RunDetail:
        """Run the deployment's active frozen version for one customer."""
        _validate_runtime_resource_selection(
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
        )
        body: Dict[str, Any] = {"external_user_id": external_user_id}
        if variables is not None:
            body["variables"] = variables
        if connections is not None:
            body["connections"] = _runtime_connections_payload(connections)
        if resource_set_id is not None:
            body["resource_set_id"] = resource_set_id
        if resource_set_revision is not None:
            body["resource_set_revision"] = resource_set_revision
        request_kwargs: Dict[str, Any] = {"json": body}
        request_kwargs["headers"] = _idempotency_headers(idempotency_key)
        data = await self._c._request(
            "POST",
            f"/api/v1/agents/deployments/{deployment_id}/run/",
            **request_kwargs,
        )
        return RunDetail(**data)

    async def run_and_wait(
        self,
        deployment_id: str,
        *,
        external_user_id: str,
        idempotency_key: str,
        variables: Optional[Dict[str, Any]] = None,
        connections: Optional[RuntimeConnections] = None,
        resource_set_id: Optional[str] = None,
        resource_set_revision: Optional[int] = None,
        timeout: float = 300,
        poll_interval: float = 2.0,
    ) -> RunDetail:
        """Run a deployment and await its execution."""
        run = await self.run(
            deployment_id,
            external_user_id=external_user_id,
            variables=variables,
            connections=connections,
            resource_set_id=resource_set_id,
            resource_set_revision=resource_set_revision,
            idempotency_key=idempotency_key,
        )
        return await self._c.runs.wait(
            run.id,
            timeout=timeout,
            poll_interval=poll_interval,
        )
