"""Opt-in native model operations on the existing authenticated backend API.

This is separate from the legacy multipart prediction clients. Reserve and save
your idempotency key and original revision/input before create(). Save the returned
operation UUID before waiting. Reconnect with get()/wait() on that SAME UUID and
revision; an unknown outcome never authorizes another key or automatic POST.

The catalog does not imply that a particular model is published or ready. Native
inputs/outputs are JSON, without text/media projection, defaults or price guesses.
Integers remain exact up to 4096 decimal digits; larger values are refused.
No OAuth, alternate payer, private worker-control or provider API is exposed.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from uuid import UUID

from pydantic import ValidationError

from flymyai.agents._types import (
    ModelProxyCatalogPage,
    ModelProxyContract,
    ModelProxyCreated,
    ModelProxyOperation,
)

if TYPE_CHECKING:
    from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

_ROOT = "/api/v1/model-proxy/"
_MAX_INPUT_BYTES = 262_144
_MAX_BODY_BYTES = 270_336
_MAX_RESPONSE_BYTES = 2_097_152
_MAX_INTEGER_DIGITS = 4096
_INTEGER_LIMIT = 10**_MAX_INTEGER_DIGITS
_MAX_INTEGER_BITS = _INTEGER_LIMIT.bit_length()
_REQUEST_SECONDS = 10.0
_MAX_WAIT_SECONDS = 7_200.0
_MAX_POLLS = 900
_REVISION = re.compile(r"[a-f0-9]{64}\Z")
_KEY = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?\Z")
_WAITING = frozenset({"queued", "claimed", "running", "file_pending"})
_STOPPED = frozenset({"succeeded", "failed", "rejected", "cancelled", "unknown"})


class ModelProxyError(Exception):
    """Safe failure metadata, never a raw response, URL, credential or input.

    ``outcome_unknown`` means the write may have been accepted. No retry is
    performed. A known operation ID can be read again; before receiving an ID,
    only an explicit identical revision/input/key admission can reconcile it.
    Identity attributes are for caller recovery, not included in str/repr.
    """

    def __init__(
        self,
        code: str,
        *,
        status_code: Optional[int] = None,
        outcome_unknown: bool = False,
        operation_id: Optional[str] = None,
        contract_revision: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> None:
        super().__init__("Model proxy request did not complete: " + code)
        self.code = code
        self.status_code = status_code
        self.outcome_unknown = outcome_unknown
        self.operation_id = operation_id
        self.contract_revision = contract_revision
        self.idempotency_key = idempotency_key


def _revision(value: str) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError("contract_revision must be an exact lowercase SHA256")
    return value


def _operation_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("operation_id must be a canonical UUID")
    try:
        if str(UUID(value)) != value:
            raise ValueError()
    except (ValueError, AttributeError):
        raise ValueError("operation_id must be a canonical UUID") from None
    return value


def _key(value: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 255
        or not _KEY.fullmatch(value)
    ):
        raise ValueError(
            "idempotency_key must be caller-owned printable ASCII (1-255), without edge spaces"
        )
    return value


def _seconds(value: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    if not 0 < value <= maximum or not math.isfinite(value):
        raise ValueError("timeout is outside the finite client bound")
    return float(value)


def _json_shape(value: Any, byte_limit: int) -> None:
    """Bound traversal before encoding; reject non-JSON types without coercion."""
    pending = [(value, 0)]
    nodes = 0
    characters = 0
    maximum = max(20_000, byte_limit // 4)
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > maximum or depth > 64:
            raise ValueError("JSON structure exceeds the bounded contract")
        kind = type(item)
        if kind is dict:
            if len(item) > maximum - nodes:
                raise ValueError("JSON object exceeds the bounded contract")
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                characters += len(key)
                pending.append((child, depth + 1))
        elif kind is list:
            if len(item) > maximum - nodes:
                raise ValueError("JSON array exceeds the bounded contract")
            pending.extend((child, depth + 1) for child in item)
        elif kind is str:
            characters += len(item)
        elif kind is int:
            # Check magnitude before ANY decimal conversion, including on
            # Python3.8 without an interpreter-wide integer digit guard.
            if (
                item.bit_length() > _MAX_INTEGER_BITS
                or item >= _INTEGER_LIMIT
                or item <= -_INTEGER_LIMIT
            ):
                raise ValueError("JSON integer exceeds the 4096-digit bound")
            # Count numeric text too: many individually valid integers must not
            # build a huge JSON string before the aggregate byte check.
            characters += len(str(item))
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
        elif item is not None and kind is not bool:
            raise ValueError("Native input must contain only JSON values")
        if characters > byte_limit or len(pending) > maximum:
            raise ValueError("JSON text exceeds the bounded contract")


def _encode(value: Any, limit: int) -> bytes:
    _json_shape(value, limit)
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("Native input is not bounded finite UTF-8 JSON") from None
    if len(encoded) > limit:
        raise ValueError("Native input exceeds the byte bound")
    return encoded


def _decode(raw: bytes) -> Dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError()
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError()

    def bounded_integer(token):
        # JSON supplies one decimal token, with at most a leading minus. Bound
        # its length before int(), not after parsing or converting to float.
        if len(token) - token.startswith("-") > _MAX_INTEGER_DIGITS:
            raise ValueError()
        return int(token)

    try:
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
            parse_int=bounded_integer,
        )
        _json_shape(value, _MAX_RESPONSE_BYTES)
        if type(value) is not dict:
            raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ModelProxyError("invalid_response") from None


def _model(cls, value):
    try:
        return cls.model_validate(value)
    except (ValidationError, ValueError, TypeError):
        raise ModelProxyError("invalid_response") from None


def _catalog_query(limit, cursor, prefix):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("catalog limit must be an integer from 1 through 100")
    if not isinstance(prefix, str) or len(prefix) > 200:
        raise ValueError("catalog prefix exceeds the API bound")
    if cursor is not None and (
        not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048
    ):
        raise ValueError("catalog cursor exceeds the API bound")
    return {
        "limit": str(limit),
        "prefix": prefix,
        **({} if cursor is None else {"cursor": cursor}),
    }


def _catalog(value, limit):
    page = _model(ModelProxyCatalogPage, value)
    if len(page.items) > limit or (
        page.release is None and (page.items or page.next_cursor)
    ):
        raise ModelProxyError("invalid_response")
    if len({row.route_key for row in page.items}) != len(page.items):
        raise ModelProxyError("invalid_response")
    return page


def _contract(value, revision):
    try:
        result = _model(ModelProxyContract, value)
        if result.contract_revision != revision:
            raise ModelProxyError("identity_mismatch")
        return result
    except ModelProxyError as error:
        error.contract_revision = revision
        raise


def _operation(value, revision, operation_id=None):
    try:
        result = _model(ModelProxyOperation, value)
        try:
            _operation_id(result.operation_id)
            _revision(result.contract_revision)
        except ValueError:
            raise ModelProxyError("invalid_response") from None
        if (
            result.prediction_id != result.operation_id
            or result.contract_revision != revision
            or (operation_id is not None and result.operation_id != operation_id)
        ):
            raise ModelProxyError("identity_mismatch")
        # The acknowledged tuple is usable for GET-only recovery even when
        # later status/output validation fails. Never adopt a mismatched UUID.
        operation_id = result.operation_id
        if result.status not in _WAITING | _STOPPED:
            raise ModelProxyError("invalid_response")
        if result.status == "succeeded" and "output" not in result.model_fields_set:
            raise ModelProxyError("invalid_response")
        return result
    except ModelProxyError as error:
        error.operation_id, error.contract_revision = operation_id, revision
        raise


def _created(value, revision):
    result = _model(ModelProxyCreated, value)
    operation = _operation(result.operation.model_dump(exclude_unset=True), revision)
    return result.model_copy(update={"operation": operation})


def _admission(revision, inputs, key):
    _revision(revision)
    _key(key)
    if type(inputs) is not dict:
        raise ValueError("inputs must be a native JSON object")
    _encode(inputs, _MAX_INPUT_BYTES)
    return _encode({"contract_revision": revision, "inputs": inputs}, _MAX_BODY_BYTES)


def _wait_bounds(timeout, poll_interval, max_polls):
    timeout = _seconds(timeout, _MAX_WAIT_SECONDS)
    interval = _seconds(poll_interval, 60.0)
    if (
        interval < 0.05
        or type(max_polls) is not int
        or not 1 <= max_polls <= _MAX_POLLS
    ):
        raise ValueError("wait requires a bounded interval and 1-900 GETs")
    return timeout, interval


def _check_wait(deadline, stopped, operation_id, revision):
    if stopped is not None and stopped():
        raise ModelProxyError(
            "wait_aborted", operation_id=operation_id, contract_revision=revision
        )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModelProxyError(
            "wait_timeout", operation_id=operation_id, contract_revision=revision
        )
    return remaining


class ModelProxy:
    """Sync catalog and durable operations. No method retries a POST.

    ``get`` and ``wait`` require the original revision as well as the saved UUID.
    ``wait`` returns UNKNOWN immediately, retaining the operation receipt. Its
    stop callback only ends local waiting; use explicit ``cancel`` to request
    server cancellation. Calls made from an event loop must use AsyncModelProxy.
    """

    def __init__(self, client: SyncAgentClient):
        self._c = client
        self._identity = (client._api_key, client._http.base_url)

    def _check_identity(self):
        # Clients belong to one caller and origin. Construct a new client when
        # switching accounts; never reconcile an old operation under a new key.
        if (self._c._api_key, self._c._http.base_url) != self._identity:
            raise ModelProxyError("client_identity_changed")

    def _request(
        self,
        method,
        suffix,
        *,
        timeout,
        query=None,
        body=None,
        key=None,
        revision=None,
        operation_id=None,
    ):
        received = False
        try:
            self._check_identity()
            status, raw = self._c._model_proxy_request(
                method,
                _ROOT + suffix,
                timeout=_seconds(timeout, _REQUEST_SECONDS),
                query=query,
                body=body,
                key=key,
            )
            received = True
            self._check_identity()
            data = _decode(raw)
            if (
                method == "POST"
                and suffix == "operations/"
                and data.get("created") is not (status == 202)
            ):
                raise ModelProxyError("invalid_response", status_code=status)
            return data
        except ModelProxyError as error:
            error.outcome_unknown |= received and method == "POST"
            error.operation_id = operation_id
            error.contract_revision = revision
            error.idempotency_key = key
            raise

    def catalog(
        self, *, limit=50, cursor=None, prefix="", timeout=10.0
    ) -> ModelProxyCatalogPage:
        """Read ONE bounded page; retain its release/cursor, never hydrate all models."""
        query = _catalog_query(limit, cursor, prefix)
        return _catalog(
            self._request("GET", "catalog/", query=query, timeout=timeout), limit
        )

    def contract(self, revision: str, *, timeout=10.0) -> ModelProxyContract:
        revision = _revision(revision)
        return _contract(
            self._request(
                "GET", "contracts/" + revision + "/", timeout=timeout, revision=revision
            ),
            revision,
        )

    def create(
        self,
        contract_revision: str,
        inputs: Dict[str, Any],
        *,
        idempotency_key: str,
        timeout=10.0,
    ) -> ModelProxyCreated:
        """Admit once. Save your original key/input before this call and UUID after it.

        An explicit identical call reconciles an uncertain admission; this method
        never retries or chooses a new key/contract, even after a lost response.
        """
        body = _admission(contract_revision, inputs, idempotency_key)
        received = False
        try:
            data = self._request(
                "POST",
                "operations/",
                body=body,
                key=idempotency_key,
                revision=contract_revision,
                timeout=timeout,
            )
            received = True
            return _created(data, contract_revision)
        except ModelProxyError as error:
            error.outcome_unknown |= received
            error.contract_revision, error.idempotency_key = (
                contract_revision,
                idempotency_key,
            )
            raise

    def get(
        self, operation_id: str, *, contract_revision: str, timeout=10.0
    ) -> ModelProxyOperation:
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        data = self._request(
            "GET",
            "operations/" + operation_id + "/",
            timeout=timeout,
            revision=revision,
            operation_id=operation_id,
        )
        return _operation(data.get("operation"), revision, operation_id)

    def cancel(
        self, operation_id: str, *, contract_revision: str, timeout=10.0
    ) -> ModelProxyOperation:
        """Request cancellation once. The returned state is not a refund/stop promise."""
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        received = False
        try:
            data = self._request(
                "POST",
                "operations/" + operation_id + "/cancel/",
                body=b"{}",
                timeout=timeout,
                revision=revision,
                operation_id=operation_id,
            )
            received = True
            return _operation(data.get("operation"), revision, operation_id)
        except ModelProxyError as error:
            error.outcome_unknown |= received
            error.operation_id, error.contract_revision = operation_id, revision
            raise

    def wait(
        self,
        operation_id: str,
        *,
        contract_revision: str,
        timeout=300.0,
        poll_interval=1.0,
        max_polls=300,
        stopped: Optional[Callable[[], bool]] = None,
    ) -> ModelProxyOperation:
        """Bounded GET-only wait on a saved UUID. Timeout never cancels/resubmits."""
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        timeout, interval = _wait_bounds(timeout, poll_interval, max_polls)
        deadline = time.monotonic() + timeout
        for attempt in range(max_polls):
            remaining = _check_wait(deadline, stopped, operation_id, revision)
            result = self.get(
                operation_id,
                contract_revision=revision,
                timeout=min(_REQUEST_SECONDS, remaining),
            )
            _check_wait(deadline, stopped, operation_id, revision)
            if result.status in _STOPPED:
                return result
            if attempt + 1 < max_polls:
                time.sleep(
                    min(
                        interval, _check_wait(deadline, stopped, operation_id, revision)
                    )
                )
        raise ModelProxyError(
            "poll_limit", operation_id=operation_id, contract_revision=revision
        )


class AsyncModelProxy(ModelProxy):
    """Async counterpart. Task cancellation ends waiting, never calls cancel()."""

    def __init__(self, client: AsyncAgentClient):
        super().__init__(client)

    async def _request(
        self,
        method,
        suffix,
        *,
        timeout,
        query=None,
        body=None,
        key=None,
        revision=None,
        operation_id=None,
    ):
        received = False
        try:
            self._check_identity()
            status, raw = await self._c._model_proxy_request(
                method,
                _ROOT + suffix,
                timeout=_seconds(timeout, _REQUEST_SECONDS),
                query=query,
                body=body,
                key=key,
            )
            received = True
            self._check_identity()
            data = _decode(raw)
            if (
                method == "POST"
                and suffix == "operations/"
                and data.get("created") is not (status == 202)
            ):
                raise ModelProxyError("invalid_response", status_code=status)
            return data
        except ModelProxyError as error:
            error.outcome_unknown |= received and method == "POST"
            error.operation_id, error.contract_revision, error.idempotency_key = (
                operation_id,
                revision,
                key,
            )
            raise

    async def catalog(
        self, *, limit=50, cursor=None, prefix="", timeout=10.0
    ) -> ModelProxyCatalogPage:
        query = _catalog_query(limit, cursor, prefix)
        return _catalog(
            await self._request("GET", "catalog/", query=query, timeout=timeout), limit
        )

    async def contract(self, revision: str, *, timeout=10.0) -> ModelProxyContract:
        revision = _revision(revision)
        return _contract(
            await self._request(
                "GET", "contracts/" + revision + "/", timeout=timeout, revision=revision
            ),
            revision,
        )

    async def create(
        self,
        contract_revision: str,
        inputs: Dict[str, Any],
        *,
        idempotency_key: str,
        timeout=10.0,
    ) -> ModelProxyCreated:
        body = _admission(contract_revision, inputs, idempotency_key)
        received = False
        try:
            data = await self._request(
                "POST",
                "operations/",
                body=body,
                key=idempotency_key,
                revision=contract_revision,
                timeout=timeout,
            )
            received = True
            return _created(data, contract_revision)
        except ModelProxyError as error:
            error.outcome_unknown |= received
            error.contract_revision, error.idempotency_key = (
                contract_revision,
                idempotency_key,
            )
            raise

    async def get(
        self, operation_id: str, *, contract_revision: str, timeout=10.0
    ) -> ModelProxyOperation:
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        data = await self._request(
            "GET",
            "operations/" + operation_id + "/",
            timeout=timeout,
            revision=revision,
            operation_id=operation_id,
        )
        return _operation(data.get("operation"), revision, operation_id)

    async def cancel(
        self, operation_id: str, *, contract_revision: str, timeout=10.0
    ) -> ModelProxyOperation:
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        received = False
        try:
            data = await self._request(
                "POST",
                "operations/" + operation_id + "/cancel/",
                body=b"{}",
                timeout=timeout,
                revision=revision,
                operation_id=operation_id,
            )
            received = True
            return _operation(data.get("operation"), revision, operation_id)
        except ModelProxyError as error:
            error.outcome_unknown |= received
            error.operation_id, error.contract_revision = operation_id, revision
            raise

    async def wait(
        self,
        operation_id: str,
        *,
        contract_revision: str,
        timeout=300.0,
        poll_interval=1.0,
        max_polls=300,
        stopped: Optional[Callable[[], bool]] = None,
    ) -> ModelProxyOperation:
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        timeout, interval = _wait_bounds(timeout, poll_interval, max_polls)
        deadline = time.monotonic() + timeout
        for attempt in range(max_polls):
            remaining = _check_wait(deadline, stopped, operation_id, revision)
            result = await self.get(
                operation_id,
                contract_revision=revision,
                timeout=min(_REQUEST_SECONDS, remaining),
            )
            _check_wait(deadline, stopped, operation_id, revision)
            if result.status in _STOPPED:
                return result
            if attempt + 1 < max_polls:
                await asyncio.sleep(
                    min(
                        interval, _check_wait(deadline, stopped, operation_id, revision)
                    )
                )
        raise ModelProxyError(
            "poll_limit", operation_id=operation_id, contract_revision=revision
        )
