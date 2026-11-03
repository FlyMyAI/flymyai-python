"""Feedback to the FlyMyAI team (``flymy.feedback.v1``).

``client.feedback.send(message, ...)`` reports a real problem: a call, run or model
that failed in a way you cannot fix, a wrong result, something missing from the
catalog, a charge that looks wrong or misleading docs. The same message within 10
minutes is not sent again (``deduplicated``), so a retry is harmless and the call takes
no idempotency key. Never put secrets or personal data in it. Unknown answer fields are
kept on the models (``extra="allow"``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

_FEEDBACK_PATH = "/api/v1/agents/feedback/"

FEEDBACK_CATEGORIES = (
    "bug",
    "wrong_result",
    "missing_tool",
    "billing",
    "docs",
    "idea",
    "praise",
    "other",
)
_MESSAGE_MAX_CHARS = 4000
_TOOL_MAX_CHARS = 255
_CLIENT_MAX_CHARS = 100
_EXECUTION_ID_MAX_CHARS = 64


class FeedbackReceipt(BaseModel):
    """The feedback was received; ``deduplicated`` when it repeats one sent within
    the last 10 minutes."""

    model_config = ConfigDict(
        extra="allow", hide_input_in_errors=True, populate_by_name=True
    )

    schema_version: str = Field(default="flymy.feedback.v1", alias="schema")
    feedback_id: str
    status: str
    deduplicated: bool = False


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    if len(value) > limit:
        raise ValueError(f"{name} must be at most {limit} characters.")
    return value


def feedback_body(
    message: str,
    *,
    category: Optional[str] = None,
    execution_id: Optional[str] = None,
    tool: Optional[str] = None,
    client: Optional[str] = None,
) -> Dict[str, Any]:
    """The request body; ``source`` is always ``sdk``."""
    body: Dict[str, Any] = {"message": _text(message, "message", _MESSAGE_MAX_CHARS)}
    if category is not None:
        if category not in FEEDBACK_CATEGORIES:
            raise ValueError(
                "category must be one of: " + ", ".join(FEEDBACK_CATEGORIES) + "."
            )
        body["category"] = category
    if execution_id is not None:
        body["execution_id"] = _text(
            execution_id, "execution_id", _EXECUTION_ID_MAX_CHARS
        )
    if tool is not None:
        body["tool"] = _text(tool, "tool", _TOOL_MAX_CHARS)
    if client is not None:
        body["client"] = _text(client, "client", _CLIENT_MAX_CHARS)
    body["source"] = "sdk"
    return body


class Feedback:
    """``client.feedback``: tell the FlyMyAI team about a problem."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def send(
        self,
        message: str,
        *,
        category: Optional[str] = None,
        execution_id: Optional[str] = None,
        tool: Optional[str] = None,
        client: Optional[str] = None,
    ) -> FeedbackReceipt:
        """Send ``message`` (1-4000 characters) to the FlyMyAI team.

        ``category`` is one of ``FEEDBACK_CATEGORIES`` (the server's default is
        ``other``), ``execution_id`` the run it is about, ``tool`` the tool (a
        runtime name or toolkit slug) and ``client`` your program's name. Raises
        :class:`~flymyai.agents.FeedbackRateLimitedError` when the account sent too
        much feedback for now."""
        return FeedbackReceipt.model_validate(
            self._c._request(
                "POST",
                _FEEDBACK_PATH,
                json=feedback_body(
                    message,
                    category=category,
                    execution_id=execution_id,
                    tool=tool,
                    client=client,
                ),
            )
        )


class AsyncFeedback:
    """``client.feedback`` of :class:`AsyncAgentClient`."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def send(
        self,
        message: str,
        *,
        category: Optional[str] = None,
        execution_id: Optional[str] = None,
        tool: Optional[str] = None,
        client: Optional[str] = None,
    ) -> FeedbackReceipt:
        return FeedbackReceipt.model_validate(
            await self._c._request(
                "POST",
                _FEEDBACK_PATH,
                json=feedback_body(
                    message,
                    category=category,
                    execution_id=execution_id,
                    tool=tool,
                    client=client,
                ),
            )
        )
