from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from flymyai.agents._files import AsyncFiles, Files
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
    AsyncWorkspaceGrants,
    Compilations,
    Deployments,
    McpResourceSets,
    AsyncMcpResourceSets,
    Runs,
    Tools,
    Versions,
    WorkspaceGrants,
)

_DEFAULT_BASE_URL = "https://backend.flymy.ai"
# Agents live on a different host from model inference (api.flymy.ai),
# so they use their own env var. `FLYMYAI_DSN` is reserved for the
# inference client and is intentionally NOT consulted here, otherwise
# pointing the model client to a staging host would silently break
# agents too.
_AGENTS_BASE_URL_ENV = "FLYMYAI_AGENTS_BASE_URL"
_DEFAULT_TIMEOUT = 60.0


class FlyMyAIAgentError(Exception):
    """Raised when the Agents API returns a non-2xx response."""

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


class RunObservationUnsupportedError(RuntimeError):
    """The server cannot supply bounded observation. No detail fallback is made."""


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


class WorkspaceGrantStaleRevisionError(FlyMyAIAgentError):
    """Raised when an owner workspace grant mutation loses its revision CAS."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: Any,
        expected_revision: Optional[int],
        actual_revision: Optional[int],
    ) -> None:
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision


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
        and body.get("code") == "FILE_STALE_VERSION"
    ):
        data = body.get("data")
        revision_data = data if isinstance(data, dict) else {}
        expected_revision = revision_data.get("expected_revision")
        actual_revision = revision_data.get("actual_revision")
        raise WorkspaceGrantStaleRevisionError(
            str(body.get("message", "The workspace revision changed.")),
            status_code=409,
            response_body=body,
            expected_revision=(
                expected_revision if type(expected_revision) is int else None
            ),
            actual_revision=(
                actual_revision if type(actual_revision) is int else None
            ),
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
        self.workspace_grants = WorkspaceGrants(self)
        self.files = Files(self)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._http.request(method, path, **kwargs)
        _raise_for_status(resp)
        if resp.status_code == 204:
            return None
        return resp.json()

    def _bounded_get(
        self, path: str, *, max_bytes: int, **kwargs: Any
    ) -> httpx.Response:
        """Read one finite page/range, including on an older or misrouted server."""
        headers = {**kwargs.pop("headers", {}), "Accept-Encoding": "identity"}
        with self._http.stream("GET", path, headers=headers, follow_redirects=False, **kwargs) as stream:
            if stream.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise RunObservationUnsupportedError("Bounded reads require identity content encoding.")
            content = bytearray()
            for chunk in stream.iter_raw(chunk_size=16384):
                if len(content) + len(chunk) > max_bytes:
                    raise RunObservationUnsupportedError(
                        "Server exceeded the bounded response limit; no detail fallback."
                    )
                content.extend(chunk)
            response = httpx.Response(
                stream.status_code,
                headers=stream.headers,
                content=bytes(content),
                request=stream.request,
            )
        _raise_for_status(response)
        return response

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
        self.workspace_grants = AsyncWorkspaceGrants(self)
        self.files = AsyncFiles(self)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = await self._http.request(method, path, **kwargs)
        _raise_for_status(resp)
        if resp.status_code == 204:
            return None
        return resp.json()

    async def _bounded_get(
        self, path: str, *, max_bytes: int, **kwargs: Any
    ) -> httpx.Response:
        """Async bounded read; closing/cancelling observation sends no run effect."""
        headers = {**kwargs.pop("headers", {}), "Accept-Encoding": "identity"}
        async with self._http.stream("GET", path, headers=headers, follow_redirects=False, **kwargs) as stream:
            if stream.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise RunObservationUnsupportedError("Bounded reads require identity content encoding.")
            content = bytearray()
            async for chunk in stream.aiter_raw(chunk_size=16384):
                if len(content) + len(chunk) > max_bytes:
                    raise RunObservationUnsupportedError(
                        "Server exceeded the bounded response limit; no detail fallback."
                    )
                content.extend(chunk)
            response = httpx.Response(
                stream.status_code,
                headers=stream.headers,
                content=bytes(content),
                request=stream.request,
            )
        _raise_for_status(response)
        return response

    async def __aenter__(self) -> AsyncAgentClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()
