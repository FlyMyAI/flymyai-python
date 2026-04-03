from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from flymyai.agents._resources import (
    Agents,
    AsyncAgents,
    AsyncCompilations,
    AsyncRuns,
    AsyncTools,
    Compilations,
    Runs,
    Tools,
)


_DEFAULT_BASE_URL = "https://api.flymy.ai"
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
            f"FlyMyAIAgentError(status_code={self.status_code}, "
            f"message={str(self)!r})"
        )


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.is_success:
        return
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    detail = body.get("detail", body) if isinstance(body, dict) else body
    raise FlyMyAIAgentError(
        f"HTTP {resp.status_code}: {detail}",
        status_code=resp.status_code,
        response_body=body,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Synchronous client
# ══════════════════════════════════════════════════════════════════════════════


class SyncAgentClient:
    """Synchronous client for the FlyMyAI Agents API.

    Usage::

        from flymyai import AgentClient

        client = AgentClient(api_key="fly-...")

        agent = client.agents.create(name="Researcher", goal="Search the web")
        run   = client.agents.run(agent.id)
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
            base_url
            or os.environ.get("FLYMYAI_DSN")
            or _DEFAULT_BASE_URL
        )
        self._max_retries = max_retries
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={"X-API-KEY": self._api_key},
            timeout=httpx.Timeout(timeout),
        )

        # resource namespaces
        self.agents = Agents(self)
        self.runs = Runs(self)
        self.tools = Tools(self)
        self.compilations = Compilations(self)

    # -- low-level request -----------------------------------------------------

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> Any:
        resp = self._http.request(method, path, **kwargs)
        _raise_for_status(resp)
        if resp.status_code == 204:
            return None
        return resp.json()

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> SyncAgentClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Asynchronous client
# ══════════════════════════════════════════════════════════════════════════════


class AsyncAgentClient:
    """Async client for the FlyMyAI Agents API.

    Usage::

        from flymyai import AsyncAgentClient

        async with AsyncAgentClient(api_key="fly-...") as client:
            agent = await client.agents.create(name="Researcher", goal="Search the web")
            run   = await client.agents.run(agent.id)
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
            base_url
            or os.environ.get("FLYMYAI_DSN")
            or _DEFAULT_BASE_URL
        )
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-KEY": self._api_key},
            timeout=httpx.Timeout(timeout),
        )

        # resource namespaces
        self.agents = AsyncAgents(self)
        self.runs = AsyncRuns(self)
        self.tools = AsyncTools(self)
        self.compilations = AsyncCompilations(self)

    # -- low-level request -----------------------------------------------------

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> Any:
        resp = await self._http.request(method, path, **kwargs)
        _raise_for_status(resp)
        if resp.status_code == 204:
            return None
        return resp.json()

    # -- context manager -------------------------------------------------------

    async def __aenter__(self) -> AsyncAgentClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()
