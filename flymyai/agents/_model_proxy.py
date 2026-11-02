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
import mimetypes
import os
import re
import time
from contextlib import contextmanager
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    Optional,
    TYPE_CHECKING,
    Union,
)
from uuid import UUID

from pydantic import ValidationError

from flymyai.agents._types import (
    ModelProxyCatalogPage,
    ModelProxyContract,
    ModelProxyCreated,
    ModelProxyFile,
    ModelProxyOperation,
)

if TYPE_CHECKING:
    from flymyai.agents._client import AsyncAgentClient, SyncAgentClient

_ROOT = "/api/v1/model-proxy/"
_FILES_ROUTE = "/api/v1/agents/files/"
_MAX_INPUT_BYTES = 262_144
_MAX_BODY_BYTES = 270_336
_MAX_RESPONSE_BYTES = 2_097_152
_MAX_INTEGER_DIGITS = 4096
_INTEGER_LIMIT = 10**_MAX_INTEGER_DIGITS
_MAX_INTEGER_BITS = _INTEGER_LIMIT.bit_length()
_REQUEST_SECONDS = 10.0
_UPLOAD_SECONDS = 600.0
_MAX_FILE_KEY = 200
_MAX_WAIT_SECONDS = 7_200.0
_MAX_POLLS = 900
_REVISION = re.compile(r"[a-f0-9]{64}\Z")
_ENDPOINT = re.compile(r"[^/\x00]{1,150}/[^/\x00]{1,255}\Z")
_KEY = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?\Z")
_WAITING = frozenset({"queued", "claimed", "running", "file_pending"})
_STOPPED = frozenset({"succeeded", "failed", "rejected", "cancelled", "unknown"})


class ModelProxyError(Exception):
    """Safe failure metadata, never a raw response, URL, credential or input.

    ``outcome_unknown`` means the write may have been accepted. No retry is
    performed. A known operation ID can be read again; before receiving an ID,
    only an explicit identical revision/input/key admission can reconcile it.
    ``refusal_code`` is the backend's own fixed code when it named why it refused
    this request, such as ``h3_references``: that request started nothing.
    Identity attributes are for caller recovery, not included in str/repr.

    A wait that ended early also says what it last read: ``last_status``,
    ``waiting_code`` while the model is starting or busy, and ``reason`` in plain words.
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
        refusal_code: Optional[str] = None,
        last: Optional[ModelProxyOperation] = None,
    ) -> None:
        self.last_status = None if last is None else last.status
        self.waiting_code = (
            last.error_code if last is not None and last.waiting_for_capacity else None
        )
        self.reason = None if last is None else last.reason
        named = code if refusal_code is None else code + " " + refusal_code
        super().__init__(
            "Model proxy request did not complete: "
            + named
            + ("" if self.reason is None else ". " + self.reason)
        )
        self.code = code
        self.status_code = status_code
        self.outcome_unknown = outcome_unknown
        self.operation_id = operation_id
        self.contract_revision = contract_revision
        self.idempotency_key = idempotency_key
        self.refusal_code = refusal_code


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
            "idempotency_key must be caller-owned printable ASCII (1-255), without edge"
            " spaces"
        )
    return value


@contextmanager
def _upload_form(
    file: Union[str, "os.PathLike[str]", bytes, BinaryIO],
    key: str,
    filename: Optional[str],
    media_type: Optional[str],
) -> Iterator[Dict[str, Any]]:
    """The multipart form of one file. A path is streamed and closed here; its size is
    the backend's to refuse. A string is always a path, never the file's content."""
    if len(_key(key)) > _MAX_FILE_KEY:
        raise ValueError("idempotency_key of a file holds at most 200 characters")
    opened: Optional[BinaryIO] = None
    if isinstance(file, (str, os.PathLike)):
        opened = open(file, "rb")
        content: Any = opened
        name = filename or os.path.basename(os.fspath(file))
    elif isinstance(file, (bytes, bytearray)):
        content, name = bytes(file), filename
    else:
        content = file
        name = filename or os.path.basename(str(getattr(file, "name", "") or ""))
    try:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("filename is required for bytes or an unnamed stream")
        # The type decides the kind of the reference, so it is never left to chance.
        kind = media_type or mimetypes.guess_type(name)[0]
        if not kind or (media_type is None and kind == "application/octet-stream"):
            raise ValueError("media_type is required: the filename does not tell it")
        yield {
            "files": {"file": (name, content, kind)},
            "data": {"idempotency_key": key},
        }
    finally:
        if opened is not None:
            opened.close()


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


def _catalog_query(limit, cursor, prefix, endpoint_id=None):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("catalog limit must be an integer from 1 through 100")
    if not isinstance(prefix, str) or len(prefix) > 200:
        raise ValueError("catalog prefix exceeds the API bound")
    if cursor is not None and (
        not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048
    ):
        raise ValueError("catalog cursor exceeds the API bound")
    if endpoint_id is not None and (
        not isinstance(endpoint_id, str) or not _ENDPOINT.fullmatch(endpoint_id)
    ):
        raise ValueError("endpoint_id must be owner/name")
    return {
        "limit": str(limit),
        "prefix": prefix,
        **({} if cursor is None else {"cursor": cursor}),
        **({} if endpoint_id is None else {"endpoint_id": endpoint_id}),
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


def _check_wait(deadline, stopped, operation_id, revision, last=None):
    if stopped is not None and stopped():
        raise ModelProxyError(
            "wait_aborted",
            operation_id=operation_id,
            contract_revision=revision,
            last=last,
        )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ModelProxyError(
            "wait_timeout",
            operation_id=operation_id,
            contract_revision=revision,
            last=last,
        )
    return remaining


def _lost_read(error):
    """A timeout, transport failure or busy backend says nothing about the operation."""
    if error.code in {"request_timeout", "transport_error"}:
        return True
    return error.code == "http_status" and (
        error.status_code >= 500 or error.status_code in {408, 429}
    )


def _report(on_update, seen, result):
    """Tell the caller once per change of state. Returns what was last reported."""
    state = (result.status, result.error_code)
    if on_update is not None and state != seen:
        on_update(result)
    return state


class ModelProxy:
    """Sync catalog and durable operations. No method retries a POST.

    ``get`` and ``wait`` require the original revision as well as the saved UUID.
    ``wait`` returns UNKNOWN immediately, retaining the operation receipt. Its
    stop callback only ends local waiting; use explicit ``cancel`` to request
    server cancellation. Calls made from an event loop must use AsyncModelProxy.

    An operation with no capacity stays ``queued``: ``waiting_for_capacity`` is true,
    ``error_code`` names why and ``reason`` says it plainly. A cold start takes about
    15 minutes for MiniMax-H3 and up to 50 for DeepSeek, more than ``wait``'s default
    timeout, so pass a longer one or call ``wait`` again on the same UUID. Never call
    ``create`` for it again. ``rejected`` with one of those codes or ``proxy_deadline``
    means the model did not start in time: the wait ran out with no charge. ``failed``
    with ``model_lease_deleted`` or ``model_http_<status>`` means no result and no charge.
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
        form=None,
    ):
        received = False
        try:
            self._check_identity()
            status, raw = self._c._model_proxy_request(
                method,
                _ROOT + suffix if form is None else _FILES_ROUTE,
                timeout=_seconds(
                    timeout, _REQUEST_SECONDS if form is None else _UPLOAD_SECONDS
                ),
                query=query,
                body=body,
                key=key,
                form=form,
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
        self, *, limit=50, cursor=None, prefix="", timeout=10.0, endpoint_id=None
    ) -> ModelProxyCatalogPage:
        """Read ONE bounded page; retain its release/cursor, never hydrate all models.

        ``endpoint_id`` (``owner/name``) reads the rows of that exact model only.
        """
        query = _catalog_query(limit, cursor, prefix, endpoint_id)
        return _catalog(
            self._request("GET", "catalog/", query=query, timeout=timeout), limit
        )

    def upload_file(
        self,
        file: Union[str, "os.PathLike[str]", bytes, BinaryIO],
        *,
        idempotency_key: str,
        filename: Optional[str] = None,
        media_type: Optional[str] = None,
        timeout=120.0,
    ) -> ModelProxyFile:
        """Store your own file and return its exact reference, in ``artifact``.

        ``file`` is a path, bytes or a binary stream, up to the backend's limit of 96
        MiB. ``media_type`` decides the kind of the reference, so it must be the file's
        real type; it defaults from the filename. After a lost reply, call again with
        the same ``idempotency_key`` (at most 200 characters) and file: it returns the
        same reference. ``timeout`` is also bounded by the client's own timeout.
        """
        with _upload_form(file, idempotency_key, filename, media_type) as form:
            received = False
            try:
                data = self._request(
                    "POST", "", form=form, key=idempotency_key, timeout=timeout
                )
                received = True
                return _model(ModelProxyFile, data)
            except ModelProxyError as error:
                error.outcome_unknown |= received
                error.idempotency_key = idempotency_key
                raise

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
        on_update: Optional[Callable[[ModelProxyOperation], None]] = None,
    ) -> ModelProxyOperation:
        """Bounded GET-only wait on a saved UUID. Timeout never cancels/resubmits.

        ``on_update`` is called with the operation each time its status or code
        changes, so a caller can show why it waits. A wait that ends early raises
        :class:`ModelProxyError` with the same identity and what it last read; once a
        state is known, one lost read does not end the wait.
        """
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        timeout, interval = _wait_bounds(timeout, poll_interval, max_polls)
        deadline = time.monotonic() + timeout
        last, seen = None, None
        for attempt in range(max_polls):
            remaining = _check_wait(deadline, stopped, operation_id, revision, last)
            try:
                last = self.get(
                    operation_id,
                    contract_revision=revision,
                    timeout=min(_REQUEST_SECONDS, remaining),
                )
            except ModelProxyError as error:
                if last is None or not _lost_read(error):
                    raise
            else:
                seen = _report(on_update, seen, last)
            _check_wait(deadline, stopped, operation_id, revision, last)
            if last is not None and last.status in _STOPPED:
                return last
            if attempt + 1 < max_polls:
                time.sleep(
                    min(
                        interval,
                        _check_wait(deadline, stopped, operation_id, revision, last),
                    )
                )
        raise ModelProxyError(
            "poll_limit",
            operation_id=operation_id,
            contract_revision=revision,
            last=last,
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
        form=None,
    ):
        received = False
        try:
            self._check_identity()
            status, raw = await self._c._model_proxy_request(
                method,
                _ROOT + suffix if form is None else _FILES_ROUTE,
                timeout=_seconds(
                    timeout, _REQUEST_SECONDS if form is None else _UPLOAD_SECONDS
                ),
                query=query,
                body=body,
                key=key,
                form=form,
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
        self, *, limit=50, cursor=None, prefix="", timeout=10.0, endpoint_id=None
    ) -> ModelProxyCatalogPage:
        query = _catalog_query(limit, cursor, prefix, endpoint_id)
        return _catalog(
            await self._request("GET", "catalog/", query=query, timeout=timeout), limit
        )

    async def upload_file(
        self,
        file: Union[str, "os.PathLike[str]", bytes, BinaryIO],
        *,
        idempotency_key: str,
        filename: Optional[str] = None,
        media_type: Optional[str] = None,
        timeout=120.0,
    ) -> ModelProxyFile:
        with _upload_form(file, idempotency_key, filename, media_type) as form:
            received = False
            try:
                data = await self._request(
                    "POST", "", form=form, key=idempotency_key, timeout=timeout
                )
                received = True
                return _model(ModelProxyFile, data)
            except ModelProxyError as error:
                error.outcome_unknown |= received
                error.idempotency_key = idempotency_key
                raise

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
        on_update: Optional[Callable[[ModelProxyOperation], None]] = None,
    ) -> ModelProxyOperation:
        operation_id, revision = _operation_id(operation_id), _revision(
            contract_revision
        )
        timeout, interval = _wait_bounds(timeout, poll_interval, max_polls)
        deadline = time.monotonic() + timeout
        last, seen = None, None
        for attempt in range(max_polls):
            remaining = _check_wait(deadline, stopped, operation_id, revision, last)
            try:
                last = await self.get(
                    operation_id,
                    contract_revision=revision,
                    timeout=min(_REQUEST_SECONDS, remaining),
                )
            except ModelProxyError as error:
                if last is None or not _lost_read(error):
                    raise
            else:
                seen = _report(on_update, seen, last)
            _check_wait(deadline, stopped, operation_id, revision, last)
            if last is not None and last.status in _STOPPED:
                return last
            if attempt + 1 < max_polls:
                await asyncio.sleep(
                    min(
                        interval,
                        _check_wait(deadline, stopped, operation_id, revision, last),
                    )
                )
        raise ModelProxyError(
            "poll_limit",
            operation_id=operation_id,
            contract_revision=revision,
            last=last,
        )
