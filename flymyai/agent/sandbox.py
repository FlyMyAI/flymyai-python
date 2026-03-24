"""Sandbox client: sync and async wrappers around the agent-sandbox REST API.

agent-sandbox exposes a simple REST API that manages Kubernetes pods as
isolated execution environments. Each pod runs one sandbox session.

API endpoints used here:
  POST   /api/v1/sandboxes           — create pod, waits until Ready
  POST   /api/v1/sandboxes/{id}/exec — run a command, returns stdout/stderr/exit_code
  DELETE /api/v1/sandboxes/{id}      — delete pod

Pod isolation (enforced by agent-sandbox + GKE):
  - gVisor (runsc): userspace kernel; host kernel syscalls are not reachable
  - NetworkPolicy: deny-all egress from the sandbox namespace
  - securityContext: uid=1000, AllowPrivilegeEscalation=false, drop ALL capabilities
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

import httpx

# Maps language name → command prefix run inside the pod
_LANGUAGE_COMMANDS: dict[str, list[str]] = {
    "python": ["python3", "-c"],
    "bash": ["bash", "-c"],
    "sh": ["sh", "-c"],
    "javascript": ["node", "-e"],
}

# Maps language → agent-sandbox template (selects the pod image)
_LANGUAGE_TEMPLATES: dict[str, str] = {
    "python": "python",  # python:3.12-slim, 200m–1 CPU, 256Mi–1Gi RAM
    "javascript": "node",  # node:22-slim,     200m–1 CPU, 256Mi–1Gi RAM
    "bash": "base",  # ubuntu:22.04,     100m–500m CPU, 128Mi–512Mi RAM
    "sh": "base",
}

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 120
_POD_CREATE_TIMEOUT = 120  # cold gVisor pod start can take up to ~8s


@dataclass
class SandboxResult:
    """Result of a code execution inside an isolated sandbox pod."""

    exit_code: int
    stdout: str
    stderr: str
    sandbox_id: str

    @property
    def ok(self) -> bool:
        """True when exit_code is 0."""
        return self.exit_code == 0


class SandboxClient:
    """Synchronous sandbox client.

    Usage as a context manager (recommended — ensures cleanup):

        with SandboxClient("http://agent-sandbox.sandboxes.svc:8080") as sb:
            result = sb.execute_code("print('hello')", language="python")
            print(result.stdout)

    Or manually:

        sb = SandboxClient(url)
        sandbox_id = sb.create("python")
        try:
            result = sb.exec(sandbox_id, ["python3", "-c", "print(42)"])
        finally:
            sb.delete(sandbox_id)
    """

    def __init__(self, url: str, default_timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._url = url.rstrip("/")
        self._default_timeout = default_timeout
        self._http = httpx.Client(timeout=_MAX_TIMEOUT + 10)

    def __enter__(self) -> SandboxClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # ── high-level ────────────────────────────────────────────────────────────

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Create a pod, run the code, delete the pod, return the result.

        Args:
            code: Source code or shell script.
            language: "python" (default), "bash", "javascript", "sh".
            timeout: Max execution time in seconds (default: self.default_timeout).

        Returns:
            SandboxResult with exit_code, stdout, stderr, sandbox_id.
        """
        timeout = min(timeout or self._default_timeout, _MAX_TIMEOUT)
        lang = language.lower()
        self._validate_language(lang)

        template = _LANGUAGE_TEMPLATES.get(lang, "base")
        sandbox_id = self.create(template, ttl_seconds=timeout + 30)
        try:
            command = _LANGUAGE_COMMANDS[lang] + [code]
            raw = self.exec(sandbox_id, command, timeout_seconds=timeout)
        finally:
            self.delete(sandbox_id)

        return SandboxResult(
            exit_code=raw["exit_code"],
            stdout=raw.get("stdout", ""),
            stderr=raw.get("stderr", ""),
            sandbox_id=sandbox_id,
        )

    # ── low-level ─────────────────────────────────────────────────────────────

    def create(self, template: str = "python", ttl_seconds: int = 60) -> str:
        """POST /api/v1/sandboxes — create pod and wait for Ready.

        Returns the sandbox id (e.g. "sb-a4f9c2b1").
        Raises httpx.HTTPStatusError on API errors.
        """
        resp = self._http.post(
            f"{self._url}/api/v1/sandboxes",
            json={"template": template, "ttl_seconds": ttl_seconds},
            timeout=_POD_CREATE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        sb_id = data.get("id")
        if not sb_id:
            raise RuntimeError(f"agent-sandbox returned no sandbox id: {data}")
        return sb_id

    def exec(
        self,
        sandbox_id: str,
        command: list[str],
        timeout_seconds: int = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """POST /api/v1/sandboxes/{id}/exec — run command in the pod.

        Returns raw dict: {"exit_code": int, "stdout": str, "stderr": str}.
        """
        resp = self._http.post(
            f"{self._url}/api/v1/sandboxes/{sandbox_id}/exec",
            json={"command": command, "timeout_seconds": timeout_seconds},
            timeout=timeout_seconds + 10,
        )
        resp.raise_for_status()
        return resp.json()

    def delete(self, sandbox_id: str) -> None:
        """DELETE /api/v1/sandboxes/{id} — best effort, does not raise."""
        try:
            self._http.delete(
                f"{self._url}/api/v1/sandboxes/{sandbox_id}",
                timeout=10,
            )
        except Exception:
            pass  # TTL cleanup in agent-sandbox catches misses within 30s

    def templates(self) -> list[dict[str, Any]]:
        """GET /api/v1/templates — list available pod templates."""
        resp = self._http.get(f"{self._url}/api/v1/templates", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_language(lang: str) -> None:
        if lang not in _LANGUAGE_COMMANDS:
            supported = ", ".join(sorted(_LANGUAGE_COMMANDS))
            raise ValueError(f"Unsupported language {lang!r}. Supported: {supported}")


class AsyncSandboxClient:
    """Asynchronous sandbox client.

    Usage:

        async with AsyncSandboxClient("http://...") as sb:
            result = await sb.execute_code("print(42)")
            print(result.stdout)
    """

    def __init__(self, url: str, default_timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._url = url.rstrip("/")
        self._default_timeout = default_timeout
        self._http = httpx.AsyncClient(timeout=_MAX_TIMEOUT + 10)

    async def __aenter__(self) -> AsyncSandboxClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """Create a pod, run the code, delete the pod, return the result (async)."""
        timeout = min(timeout or self._default_timeout, _MAX_TIMEOUT)
        lang = language.lower()
        SandboxClient._validate_language(lang)

        template = _LANGUAGE_TEMPLATES.get(lang, "base")
        sandbox_id = await self.create(template, ttl_seconds=timeout + 30)
        try:
            command = _LANGUAGE_COMMANDS[lang] + [code]
            raw = await self.exec(sandbox_id, command, timeout_seconds=timeout)
        finally:
            await self.delete(sandbox_id)

        return SandboxResult(
            exit_code=raw["exit_code"],
            stdout=raw.get("stdout", ""),
            stderr=raw.get("stderr", ""),
            sandbox_id=sandbox_id,
        )

    async def create(self, template: str = "python", ttl_seconds: int = 60) -> str:
        """POST /api/v1/sandboxes — async, returns sandbox id."""
        resp = await self._http.post(
            f"{self._url}/api/v1/sandboxes",
            json={"template": template, "ttl_seconds": ttl_seconds},
            timeout=_POD_CREATE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        sb_id = data.get("id")
        if not sb_id:
            raise RuntimeError(f"agent-sandbox returned no sandbox id: {data}")
        return sb_id

    async def exec(
        self,
        sandbox_id: str,
        command: list[str],
        timeout_seconds: int = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """POST /api/v1/sandboxes/{id}/exec — async exec."""
        resp = await self._http.post(
            f"{self._url}/api/v1/sandboxes/{sandbox_id}/exec",
            json={"command": command, "timeout_seconds": timeout_seconds},
            timeout=timeout_seconds + 10,
        )
        resp.raise_for_status()
        return resp.json()

    async def delete(self, sandbox_id: str) -> None:
        """DELETE /api/v1/sandboxes/{id} — async, best effort."""
        try:
            await self._http.delete(
                f"{self._url}/api/v1/sandboxes/{sandbox_id}",
                timeout=10,
            )
        except Exception:
            pass

    async def templates(self) -> list[dict[str, Any]]:
        """GET /api/v1/templates — async, list available pod templates."""
        resp = await self._http.get(f"{self._url}/api/v1/templates", timeout=10)
        resp.raise_for_status()
        return resp.json()


# ── one-shot helpers ──────────────────────────────────────────────────────────


def execute_code(
    sandbox_url: str,
    code: str,
    language: str = "python",
    timeout: Optional[int] = None,
) -> SandboxResult:
    """One-shot: create pod → run code → delete pod → return result (sync).

    Args:
        sandbox_url: Base URL of the agent-sandbox service.
        code: Source code to execute.
        language: "python" (default), "bash", "javascript", "sh".
        timeout: Max execution time in seconds (default 30, max 120).

    Returns:
        SandboxResult(exit_code, stdout, stderr, sandbox_id).

    Example:
        result = execute_code(
            sandbox_url="http://agent-sandbox.sandboxes.svc:8080",
            code="print(sum(range(100)))",
        )
        print(result.stdout)   # 4950
        assert result.ok
    """
    with SandboxClient(sandbox_url) as client:
        return client.execute_code(code, language=language, timeout=timeout)


async def async_execute_code(
    sandbox_url: str,
    code: str,
    language: str = "python",
    timeout: Optional[int] = None,
) -> SandboxResult:
    """One-shot: create pod → run code → delete pod → return result (async).

    Example:
        result = await async_execute_code(
            sandbox_url="http://agent-sandbox.sandboxes.svc:8080",
            code="import asyncio; print('async ok')",
        )
    """
    async with AsyncSandboxClient(sandbox_url) as client:
        return await client.execute_code(code, language=language, timeout=timeout)
