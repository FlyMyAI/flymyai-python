"""One owner-scoped page of saved-version metadata, without full sources."""

from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
_SCHEMA = "flymy.saved-versions.v1"
_MAX_BYTES = 64 * 1024
_FORBIDDEN = frozenset({
    "instruction_md",
    "script_code",
    "result",
    "error",
    "messages",
    "display_messages",
    "agent_result",
    "freeze_guidance",
    "example_variables",
    "schedule_variables",
    "browser_media_consent",
    "input_schema",
    "output_schema",
})


class SavedVersionSummary(BaseModel):
    """Metadata for one saved compilation. Use ``compilations.get(id)`` for details."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    id: int = Field(strict=True, ge=1)
    execution: str = Field(strict=True, max_length=16)
    user_agent_task: str = Field(strict=True)
    status: str = Field(strict=True, max_length=255)
    last_run_status: Optional[str] = Field(max_length=255)
    name: str = Field(strict=True, max_length=200)
    note: str = Field(strict=True, max_length=240)
    note_truncated: bool = Field(strict=True)
    is_live: bool = Field(strict=True)
    has_instruction: bool = Field(strict=True)
    cron_schedule: str = Field(strict=True, max_length=128)
    timezone: str = Field(strict=True, max_length=64)
    schedule_active: bool = Field(strict=True)
    schedule_paused: bool = Field(strict=True)
    schedule_paused_reason: str = Field(strict=True, max_length=240)
    schedule_paused_reason_truncated: bool = Field(strict=True)
    schedule_ready: bool = Field(strict=True)
    source_compilation: Optional[int] = Field(strict=True, ge=1)
    created_at: datetime
    updated_at: datetime


class SavedVersionPage(BaseModel):
    """A single page; pass ``next_cursor`` explicitly to read the next page."""

    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    schema_version: Literal["flymy.saved-versions.v1"] = Field(alias="schema")
    task: str
    results: List[SavedVersionSummary]
    next_cursor: Optional[str]


def _valid_cursor(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 512
        and all(0x21 <= ord(char) <= 0x7E for char in value)
    )


def saved_version_query(
    agent_id: str, limit: int, cursor: Optional[str]
) -> Dict[str, Any]:
    if not isinstance(agent_id, str) or _UUID.fullmatch(agent_id) is None:
        raise ValueError("agent_id must be the UUID of one agent you own.")
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer between 1 and 50.")
    params: Dict[str, Any] = {"task": agent_id.lower(), "limit": limit}
    if cursor is not None:
        if not _valid_cursor(cursor):
            raise ValueError(
                "cursor must be an opaque printable string of at most 512 characters."
            )
        params["cursor"] = cursor
    return params


def saved_version_page(value: Any, query: Dict[str, Any]) -> SavedVersionPage:
    """Refuse legacy arrays, wrong owner scope, full payloads and repeated pages."""
    if (
        not isinstance(value, dict)
        or value.get("schema") != _SCHEMA
        or value.get("task") != query["task"]
        or not isinstance(value.get("results"), list)
        or len(value["results"]) > query["limit"]
        or "next_cursor" not in value
    ):
        raise ValueError("Invalid saved-version summary page.")
    cursor = value["next_cursor"]
    if cursor is not None and (
        not _valid_cursor(cursor) or cursor == query.get("cursor")
    ):
        raise ValueError("Invalid saved-version next cursor.")
    if (
        len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        > _MAX_BYTES
    ):
        raise ValueError("Saved-version summary page exceeds 64 KiB.")
    previous_id: Optional[int] = None
    for row in value["results"]:
        if (
            not isinstance(row, dict)
            or type(row.get("id")) is not int
            or row["id"] < 1
            or (previous_id is not None and row["id"] >= previous_id)
            or row.get("user_agent_task") != query["task"]
            or bool(_FORBIDDEN.intersection(row))
        ):
            raise ValueError("Invalid saved-version summary scope or projection.")
        previous_id = row["id"]
    return SavedVersionPage(**value)
