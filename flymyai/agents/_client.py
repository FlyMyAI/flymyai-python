from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

import httpx

from flymyai.agents._apps import Apps, AsyncApps
from flymyai.agents._artifacts import Artifacts, AsyncArtifacts
from flymyai.agents._billing import AsyncBilling, Billing
from flymyai.agents._feedback import AsyncFeedback, Feedback
from flymyai.agents._triggers import TriggersV1, AsyncTriggersV1
from flymyai.agents._projects import AsyncProjects, Projects
from flymyai.agents._mcp_sharing import McpTeams, AsyncMcpTeams
from flymyai.agents._mcp_personal import McpShares, AsyncMcpShares
from flymyai.agents._mcp_servers import McpServers, AsyncMcpServers
from flymyai.agents._model_proxy import (
    AsyncModelProxy,
    ModelProxy,
    ModelProxyError,
    _MAX_RESPONSE_BYTES,
    _UPLOAD_SECONDS,
)

from flymyai.agents._resources import (
    AgentGroups,
    Agents,
    AsyncAgentGroups,
    AsyncAgents,
    AsyncCompilations,
    AsyncDeployments,
    AsyncRuns,
    AsyncTools,
    AsyncVersions,
    Compilations,
    Deployments,
    McpResourceSets,
    AsyncMcpResourceSets,
    Runs,
    Tools,
    Versions,
)

_DEFAULT_BASE_URL = "https://backend.flymy.ai"
# Agents live on a different host from model inference (api.flymy.ai),
# so they use their own env var. `FLYMYAI_DSN` is reserved for the
# inference client and is intentionally NOT consulted here, otherwise
# pointing the model client to a staging host would silently break
# agents too.
_AGENTS_BASE_URL_ENV = "FLYMYAI_AGENTS_BASE_URL"
_DEFAULT_TIMEOUT = 60.0
_REFUSAL_BYTES = 4096
_REFUSAL_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")


class FlyMyAIAgentError(Exception):
    """Raised when the Agents API returns an error or an unexpected status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status_code={self.status_code}, "
            f"message={str(self)!r})"
        )


class VariablesValidationError(FlyMyAIAgentError):
    """Raised when the backend rejects ``variables`` on run / run_instruction.

    The server responds with HTTP 400 and a body of the form
    ``{"variables": ["'foo' is required", ...]}`` - those messages are
    exposed here as :attr:`messages`, and the field-to-message mapping
    (best-effort parse) as :attr:`field_errors`.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any,
        messages: list[str],
        field_errors: Dict[str, str],
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.messages = messages
        self.field_errors = field_errors


class SuggestSchemaError(FlyMyAIAgentError):
    """Raised when the server fails to generate schemas (HTTP 502).

    Typically means the server's Anthropic key is missing or the upstream
    call failed. The user-facing message is in :attr:`args`.
    """


class McpResourceSetStaleRevisionError(FlyMyAIAgentError):
    """Raised when a resource-set metadata or member write is stale."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any,
        current_revision: Optional[int],
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.current_revision = current_revision


class McpServerOAuthError(FlyMyAIAgentError):
    """A custom MCP server's OAuth refused (``code`` ``oauth_*``). With
    ``oauth_client_required`` the provider has no automatic client registration:
    register ``redirect_uri`` in your own OAuth app, save its ``oauth_client_id`` and
    ``oauth_client_secret`` on the server, then authorize again."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any = None,
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.code = code
        self.redirect_uri = redirect_uri


class ArtifactStaleBaseVersionError(FlyMyAIAgentError):
    """An artifact version was published on a ``base_version`` that is neither the
    latest nor the live version any more (HTTP 412): someone published meanwhile.
    Read the version your change starts from - :attr:`live_version` to keep a saved
    draft a draft, or :attr:`latest_version` - apply the change there and publish
    again with a new idempotency key."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any,
        latest_version: Optional[int],
        live_version: Optional[int] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.latest_version = latest_version
        self.live_version = live_version


class AppSecretLimitError(FlyMyAIAgentError):
    """An app holds at most 32 secrets and an owner 1024 (HTTP 409 ``too_many_secrets``):
    delete one with ``client.apps.secrets.delete`` first."""


def _usd(value: Any) -> Optional[Decimal]:
    """A dollar amount from an answer (a decimal string or a number), or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return amount if amount.is_finite() else None


class InsufficientBalanceError(FlyMyAIAgentError):
    """The balance cannot cover the call (HTTP 402 ``insufficient_balance``); nothing ran.

    :attr:`shortfall_usd` is how much it is short and :attr:`top_up_url` the page where
    the account owner adds funds, when the answer names them. A program can also make a
    payment link for an amount with ``client.billing.create_top_up_link``. Retry after
    the owner has paid; a durable call takes a new idempotency key."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any,
        shortfall_usd: Optional[Decimal] = None,
        top_up_url: Optional[str] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.shortfall_usd = shortfall_usd
        self.top_up_url = top_up_url


class InvalidAmountError(FlyMyAIAgentError):
    """A top-up amount outside the account's limits (HTTP 400 ``invalid_amount``):
    :attr:`minimum_usd` and :attr:`maximum_usd` (``None`` when the answer has none)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any,
        minimum_usd: Optional[Decimal] = None,
        maximum_usd: Optional[Decimal] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.minimum_usd = minimum_usd
        self.maximum_usd = maximum_usd


class RateLimitedError(FlyMyAIAgentError):
    """Too many requests of this kind for now (HTTP 429 ``rate_limited``); nothing ran.
    Wait :attr:`retry_after_seconds` (``None`` when the answer gives no time) and try
    again."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any,
        code: str,
        retry_after_seconds: Optional[int] = None,
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class FeedbackRateLimitedError(RateLimitedError):
    """This account sent too much feedback for now (HTTP 429 ``feedback_rate_limited``)."""


_RATE_LIMIT_CODES = {
    "rate_limited": RateLimitedError,
    "feedback_rate_limited": FeedbackRateLimitedError,
}


def _retry_after_seconds(body: Dict[str, Any], header: Optional[str]) -> Optional[int]:
    value = body.get("retry_after_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = header.strip() if isinstance(header, str) else None
        if not value or not value.isdigit() or len(value) > 9:
            return None
    seconds = int(value)
    return seconds if seconds >= 0 else None


def _typed_refusal(resp: httpx.Response, body: Any) -> Optional[FlyMyAIAgentError]:
    """Money and rate refusals that name what to do next, by their stable code."""
    if not isinstance(body, dict) or not isinstance(body.get("code"), str):
        return None
    code = body["code"]
    # The same message as an untyped refusal, so existing handling keeps working.
    message = f"HTTP {resp.status_code}: {body.get('detail', body)}"
    if resp.status_code == 402 and code == "insufficient_balance":
        url = body.get("top_up_url")
        return InsufficientBalanceError(
            message,
            status_code=402,
            response_body=body,
            shortfall_usd=_usd(body.get("shortfall_usd")),
            top_up_url=(
                url if isinstance(url, str) and url.startswith("https://") else None
            ),
        )
    if resp.status_code == 400 and code == "invalid_amount":
        return InvalidAmountError(
            message,
            status_code=400,
            response_body=body,
            minimum_usd=_usd(body.get("minimum_usd")),
            maximum_usd=_usd(body.get("maximum_usd")),
        )
    if resp.status_code == 429 and code in _RATE_LIMIT_CODES:
        return _RATE_LIMIT_CODES[code](
            message,
            status_code=429,
            response_body=body,
            code=code,
            retry_after_seconds=_retry_after_seconds(
                body, resp.headers.get("Retry-After")
            ),
        )
    return None


def _parse_variables_errors(body: Any) -> Optional[VariablesValidationError]:
    """Return a :class:`VariablesValidationError` if ``body`` looks like one."""
    if not isinstance(body, dict):
        return None
    raw = body.get("variables")
    messages: list[str] = []
    if isinstance(raw, list):
        messages = [str(m) for m in raw]
    elif isinstance(raw, str):
        messages = [raw]
    elif isinstance(raw, dict):
        messages = [f"{k}: {v}" for k, v in raw.items()]
    else:
        return None

    field_errors: Dict[str, str] = {}
    for msg in messages:
        # Messages often start with a quoted field name, e.g.
        #   "'website_url' is a required property"
        if msg.startswith("'"):
            end = msg.find("'", 1)
            if end > 1:
                field_errors[msg[1:end]] = msg
    return VariablesValidationError(
        f"Invalid variables: {'; '.join(messages)}",
        status_code=400,
        response_body=body,
        messages=messages,
        field_errors=field_errors,
    )


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        body = resp.json()
    except Exception:
        body = resp.text

    if resp.status_code == 400:
        err = _parse_variables_errors(body)
        if err is not None:
            raise err

    refusal = _typed_refusal(resp, body)
    if refusal is not None:
        raise refusal

    if (
        resp.status_code == 409
        and isinstance(body, dict)
        and body.get("code") == "stale_revision"
    ):
        current_revision = body.get("current_revision")
        raise McpResourceSetStaleRevisionError(
            str(body.get("detail", "The MCP resource set changed.")),
            status_code=409,
            response_body=body,
            current_revision=(
                current_revision if isinstance(current_revision, int) else None
            ),
        )

    if (
        resp.status_code == 409
        and isinstance(body, dict)
        and body.get("code") == "too_many_secrets"
    ):
        raise AppSecretLimitError(
            str(body.get("message", "Too many app secrets.")),
            status_code=409,
            response_body=body,
        )

    if (
        resp.status_code == 412
        and isinstance(body, dict)
        and body.get("code") == "stale_base_version"
    ):
        details = body.get("details")
        details = details if isinstance(details, dict) else {}
        latest = details.get("latest_version")
        live = details.get("live_version")
        raise ArtifactStaleBaseVersionError(
            str(body.get("detail", "A newer version of the artifact exists.")),
            status_code=412,
            response_body=body,
            latest_version=latest if isinstance(latest, int) else None,
            live_version=live if isinstance(live, int) else None,
        )

    if (
        isinstance(body, dict)
        and isinstance(body.get("code"), str)
        and body["code"].startswith("oauth_")
    ):
        redirect_uri = body.get("redirect_uri")
        raise McpServerOAuthError(
            f"HTTP {resp.status_code}: {body.get('detail', body['code'])}",
            status_code=resp.status_code,
            response_body=body,
            code=body["code"],
            redirect_uri=redirect_uri if isinstance(redirect_uri, str) else None,
        )

    detail = body.get("detail", body) if isinstance(body, dict) else body
    if resp.status_code == 502:
        raise SuggestSchemaError(
            f"HTTP 502: {detail}",
            status_code=502,
            response_body=body,
        )
    raise FlyMyAIAgentError(
        f"HTTP {resp.status_code}: {detail}",
        status_code=resp.status_code,
        response_body=body,
    )


def _model_proxy_budget(client, requested, ceiling=10.0):
    """Native calls cannot outlive the caller's configured request timeout."""
    if client._http.is_closed:
        raise ModelProxyError("client_closed")
    values = [requested]
    values.extend(
        value for value in client._http.timeout.as_dict().values() if value is not None
    )
    if any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v <= 0
        for v in values
    ):
        raise ModelProxyError("invalid_timeout")
    return min(ceiling, *values)


async def _refusal_code(response):
    """The backend's own fixed code of a refused request. Only the code crosses."""
    status = response.status_code
    if (
        not 400 <= status < 500
        or status in {401, 408}
        or response.headers.get("content-type", "").split(";")[0].strip().lower()
        != "application/json"
    ):
        return None
    data = bytearray()
    async for chunk in response.aiter_raw():
        data.extend(chunk[: _REFUSAL_BYTES - len(data)])
        if len(data) >= _REFUSAL_BYTES:
            break
    try:
        body = json.loads(bytes(data))
        # {code} on the model routes, {error: {code}} on the file route.
        code = body["error"]["code"] if "error" in body else body.get("code")
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return None
    return code if isinstance(code, str) and _REFUSAL_CODE.fullmatch(code) else None


async def _model_proxy_exchange(
    client, method, path, *, timeout, query, body, key, form=None
):
    """One streamed exchange, independent of legacy reconnect/retry clients.

    The wall clock includes response headers and body, not just HTTPX's per-read
    inactivity timeout. Async cancellation closes this exchange; it does not
    cancel or resubmit a server operation. OS scheduling/cleanup can overrun the
    deadline. No response body, credential or underlying exception is logged.
    A ``form`` (multipart ``files`` and ``data``) replaces the JSON body and may take
    longer than a JSON exchange.
    """
    ceiling = 10.0 if form is None else _UPLOAD_SECONDS
    budget = _model_proxy_budget(client, timeout, ceiling)
    deadline = time.monotonic() + budget
    dispatched = False
    try:
        base_url = client._http.base_url
        if (
            base_url.scheme not in {"http", "https"}
            or base_url.userinfo
            or base_url.query
            or base_url.fragment
        ):
            raise ModelProxyError("invalid_base_url")
        identity = (client._api_key, base_url)
        headers = {
            "X-API-KEY": identity[0],
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        if form is None:
            headers["Content-Type"] = "application/json"
        if key is not None:
            headers["Idempotency-Key"] = key

        async def exchange():
            nonlocal dispatched
            # No ambient proxy credentials, redirects, HTTP retries or pool
            # reuse/reconnect after an ambiguous POST. This opt-in transport
            # exists only for the new model_proxy resource.
            async with httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=budget,
                follow_redirects=False,
                trust_env=False,
            ) as http:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelProxyError("request_timeout")
                _model_proxy_budget(client, remaining)
                if (client._api_key, client._http.base_url) != identity:
                    raise ModelProxyError("client_identity_changed")
                dispatched = True
                async with http.stream(
                    method,
                    path,
                    params=query,
                    content=body,
                    timeout=min(budget, remaining),
                    **(form or {}),
                ) as response:
                    status = response.status_code
                    allowed = (
                        {200, 201}
                        if form is not None
                        else (
                            {200, 202}
                            if path.endswith("/operations/") and method == "POST"
                            else {200}
                        )
                    )
                    if status not in allowed:
                        # A named refusal proves this request started nothing.
                        refusal = await _refusal_code(response)
                        raise ModelProxyError(
                            "http_status",
                            status_code=status,
                            outcome_unknown=method == "POST" and refusal is None,
                            refusal_code=refusal,
                        )
                    if (
                        response.headers.get("content-encoding", "identity").lower()
                        != "identity"
                    ):
                        raise ModelProxyError(
                            "encoded_response",
                            status_code=status,
                            outcome_unknown=method == "POST",
                        )
                    length = response.headers.get("content-length")
                    if length is not None and (
                        not length.isascii()
                        or not length.isdigit()
                        or len(length) > 10
                        or int(length) > _MAX_RESPONSE_BYTES
                    ):
                        raise ModelProxyError(
                            "response_too_large",
                            status_code=status,
                            outcome_unknown=method == "POST",
                        )
                    data = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(chunk) > _MAX_RESPONSE_BYTES - len(data):
                            raise ModelProxyError(
                                "response_too_large",
                                status_code=status,
                                outcome_unknown=method == "POST",
                            )
                        data.extend(chunk)
                    return status, bytes(data)

        try:
            result = await asyncio.wait_for(exchange(), timeout=budget)
            if time.monotonic() > deadline:
                raise ModelProxyError(
                    "request_timeout", outcome_unknown=dispatched and method == "POST"
                )
            return result
        except (asyncio.TimeoutError, httpx.TimeoutException):
            raise ModelProxyError(
                "request_timeout", outcome_unknown=dispatched and method == "POST"
            ) from None
        except httpx.HTTPError:
            raise ModelProxyError(
                "transport_error", outcome_unknown=dispatched and method == "POST"
            ) from None
    except ModelProxyError:
        raise
    except (ValueError, TypeError, UnicodeError, OSError):
        raise ModelProxyError(
            "transport_configuration", outcome_unknown=dispatched and method == "POST"
        ) from None


class SyncAgentClient:
    """Synchronous client for the FlyMyAI Agents API.

    Usage::

        from flymyai import AgentClient

        client = AgentClient(api_key="fly-...")

        agent = client.agents.create(name="Researcher", goal="Search the web")
        run   = client.agents.run(agent.id, idempotency_key="research-run-1")
        result = client.runs.wait(run.id)
        print(result.output)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key or os.environ.get("FLYMYAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "api_key is required. Pass it directly or set FLYMYAI_API_KEY."
            )
        self._base_url = (
            base_url or os.environ.get(_AGENTS_BASE_URL_ENV) or _DEFAULT_BASE_URL
        )
        self._max_retries = max_retries
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={"X-API-KEY": self._api_key},
            timeout=httpx.Timeout(timeout),
        )

        self.agents = Agents(self)
        self.runs = Runs(self)
        self.tools = Tools(self)
        self.compilations = Compilations(self)
        self.versions = Versions(self)
        self.deployments = Deployments(self)
        self.mcp_resource_sets = McpResourceSets(self)
        self.agent_groups = AgentGroups(self)
        self.teams = McpTeams(self)
        self.shares = McpShares(self)
        self.mcp_servers = McpServers(self)
        self.artifacts = Artifacts(self)
        self.triggers_v1 = TriggersV1(self)
        self.projects = Projects(self)
        self.apps = Apps(self)
        self.model_proxy = ModelProxy(self)
        self.billing = Billing(self)
        self.feedback = Feedback(self)

    def _model_proxy_request(
        self, method, path, *, timeout, query=None, body=None, key=None, form=None
    ):
        _model_proxy_budget(self, timeout)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ModelProxyError("use_async_client")
        return asyncio.run(
            _model_proxy_exchange(
                self,
                method,
                path,
                timeout=timeout,
                query=query,
                body=body,
                key=key,
                form=form,
            )
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        resp = self._http.request(method, path, **kwargs)
        _raise_for_status(resp)
        if expected_status is not None and resp.status_code != expected_status:
            raise FlyMyAIAgentError(
                "Write outcome unknown: unexpected API response status. Retry with the"
                " same idempotency key.",
                status_code=resp.status_code,
            )
        if resp.status_code == 204:
            return None
        return resp.json()

    def __enter__(self) -> SyncAgentClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()


class AsyncAgentClient:
    """Async client for the FlyMyAI Agents API.

    Usage::

        from flymyai import AsyncAgentClient

        async with AsyncAgentClient(api_key="fly-...") as client:
            agent = await client.agents.create(name="Researcher", goal="Search the web")
            run   = await client.agents.run(
                agent.id,
                idempotency_key="research-run-1",
            )
            result = await client.runs.wait(run.id)
            print(result.output)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key or os.environ.get("FLYMYAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "api_key is required. Pass it directly or set FLYMYAI_API_KEY."
            )
        self._base_url = (
            base_url or os.environ.get(_AGENTS_BASE_URL_ENV) or _DEFAULT_BASE_URL
        )
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-KEY": self._api_key},
            timeout=httpx.Timeout(timeout),
        )

        self.agents = AsyncAgents(self)
        self.runs = AsyncRuns(self)
        self.tools = AsyncTools(self)
        self.compilations = AsyncCompilations(self)
        self.versions = AsyncVersions(self)
        self.deployments = AsyncDeployments(self)
        self.mcp_resource_sets = AsyncMcpResourceSets(self)
        self.agent_groups = AsyncAgentGroups(self)
        self.teams = AsyncMcpTeams(self)
        self.shares = AsyncMcpShares(self)
        self.mcp_servers = AsyncMcpServers(self)
        self.artifacts = AsyncArtifacts(self)
        self.triggers_v1 = AsyncTriggersV1(self)
        self.projects = AsyncProjects(self)
        self.apps = AsyncApps(self)
        self.model_proxy = AsyncModelProxy(self)
        self.billing = AsyncBilling(self)
        self.feedback = AsyncFeedback(self)

    async def _model_proxy_request(
        self, method, path, *, timeout, query=None, body=None, key=None, form=None
    ):
        return await _model_proxy_exchange(
            self,
            method,
            path,
            timeout=timeout,
            query=query,
            body=body,
            key=key,
            form=form,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        resp = await self._http.request(method, path, **kwargs)
        _raise_for_status(resp)
        if expected_status is not None and resp.status_code != expected_status:
            raise FlyMyAIAgentError(
                "Write outcome unknown: unexpected API response status. Retry with the"
                " same idempotency key.",
                status_code=resp.status_code,
            )
        if resp.status_code == 204:
            return None
        return resp.json()

    async def __aenter__(self) -> AsyncAgentClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()
