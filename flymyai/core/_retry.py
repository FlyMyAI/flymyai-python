"""When the prediction clients send a request again, and how long they wait first.

The rules of the official OpenAI and Anthropic SDKs (openai-python 3.49.0, anthropic
2.8.0, read from their sources):

- A request is sent again only before its answer started: when the connection failed,
  was reset or timed out before a response arrived, or when the response's status says
  so. The server decides with ``x-should-retry: true|false`` when it sends it; a JSON
  body with ``"retry_safe": false`` (the gateway's unknown-outcome answer) also stops
  it; otherwise 408, 409, 429 and every status from 500 retry.
- The wait honours ``retry-after-ms`` or ``retry-after`` (seconds or an HTTP date) up
  to two minutes; otherwise exponential backoff from 0.5 s to 8 s, less up to 25%.
- A stream that has delivered an event is never sent again: the model may be running
  and a second run would repeat the output and its cost. The caller gets
  ``FlyMyAIStreamInterrupted`` and decides.
"""

import email.utils
import json
import math
import random
import time
from typing import Any, Mapping, Optional

INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0
# A longer server-directed wait falls back to the exponential backoff.
MAX_RETRY_AFTER_DELAY = 120.0

# Statuses below 500 that are worth another attempt (as in the official SDKs).
RETRYABLE_4XX = frozenset({408, 409, 429})
# Gateway 409 codes that another attempt cannot change.
_FINAL_CONFLICT_CODES = frozenset({"idempotency_key_reused", "tool_outcome_unknown"})


def header_should_retry(headers: Optional[Mapping[str, str]]) -> Optional[bool]:
    """``x-should-retry`` as the server sent it: True, False, or None when absent."""
    if not headers:
        return None
    try:
        value = headers.get("x-should-retry")
    except AttributeError:
        return None
    if value is None:
        return None
    value = str(value).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _body_hints(content: Any) -> Optional[bool]:
    """False when a JSON error body says another attempt cannot help."""
    if not content:
        return None
    try:
        body = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    if body.get("retry_safe") is False:
        return False
    if body.get("code") in _FINAL_CONFLICT_CODES:
        return False
    return None


def status_should_retry(
    status_code: int,
    headers: Optional[Mapping[str, str]] = None,
    content: Any = None,
) -> bool:
    """Whether a response with this status, headers and body should be retried."""
    override = header_should_retry(headers)
    if override is not None:
        return override
    hint = _body_hints(content)
    if hint is not None:
        return hint
    return status_code in RETRYABLE_4XX or status_code >= 500


def exception_should_retry(exc: BaseException) -> bool:
    """A FlyMyAI error: the server's header or body decides, else the status mapping."""
    response = getattr(exc, "response", None)
    override = header_should_retry(getattr(response, "headers", None))
    if override is not None:
        return override
    hint = _body_hints(_content(response))
    if hint is not None:
        return hint
    return bool(getattr(exc, "requires_retry", False))


def _content(response: Any) -> Any:
    if response is None:
        return None
    try:
        return response.content
    except Exception:  # noqa: BLE001 - a streamed body that was never read
        return None


def parse_retry_after(headers: Optional[Mapping[str, str]]) -> Optional[float]:
    """Seconds the server asked to wait, or None."""
    if not headers:
        return None
    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            value = headers.get(header)
        except AttributeError:
            return None
        if value is None:
            continue
        try:
            return float(value) / divisor
        except (TypeError, ValueError):
            continue
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_tz(value)
        if parsed is None:
            return None
        return float(email.utils.mktime_tz(parsed) - time.time())
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def retry_delay(
    retries_done: int, headers: Optional[Mapping[str, str]] = None
) -> float:
    """Seconds to wait before the next attempt; ``retries_done`` counts from 0."""
    retry_after = parse_retry_after(headers)
    if (
        retry_after is not None
        and math.isfinite(retry_after)
        and 0 < retry_after <= MAX_RETRY_AFTER_DELAY
    ):
        return retry_after
    sleep_seconds = min(
        INITIAL_RETRY_DELAY * pow(2.0, min(max(retries_done, 0), 1000)),
        MAX_RETRY_DELAY,
    )
    return max(sleep_seconds * (1 - 0.25 * random.random()), 0.0)


def exception_retry_delay(retries_done: int, exc: BaseException) -> float:
    response = getattr(exc, "response", None)
    return retry_delay(retries_done, getattr(response, "headers", None))
