"""Provider-neutral, project-scoped sandbox calls. No automatic write replay."""

from typing import Any, Dict, Optional

from flymyai.agents._artifacts import _key

ACTIONS = frozenset({"sandbox_create", "sandbox_list", "sandbox_status", "sandbox_command",
                     "sandbox_command_status", "sandbox_read_file", "sandbox_write_file",
                     "sandbox_desktop", "sandbox_connect", "sandbox_revoke", "sandbox_stop"})
READS = frozenset({"sandbox_list", "sandbox_status", "sandbox_command_status", "sandbox_read_file", "sandbox_desktop"})


def _request(action: str, app: str, resource: str, arguments: Optional[Dict[str, Any]], idempotency_key: Optional[str]) -> Dict[str, Any]:
    if action not in ACTIONS:
        raise ValueError("unknown sandbox action")
    if not app or not resource:
        raise ValueError("app and declared sandbox resource are required")
    headers = _key(idempotency_key) if idempotency_key is not None or action not in READS else {}
    return {"json": {"app": app, "resource": resource, "input": arguments or {}}, "headers": headers}


class Sandboxes:
    def __init__(self, client: Any) -> None:
        self._client = client

    def call(self, action: str, *, app: str, resource: str, arguments: Optional[Dict[str, Any]] = None, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        """Call the actions discovered through apps.catalog(). Writes require a stable key.

        sandbox_connect installs the native runtime SDK in the VM. This SDK's
        account API key must never be copied into customer code or the VM.
        """
        return self._client._request("POST", f"/api/v1/agents/apps/sandbox/{action}/", **_request(action, app, resource, arguments, idempotency_key))


class AsyncSandboxes:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def call(self, action: str, *, app: str, resource: str, arguments: Optional[Dict[str, Any]] = None, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        return await self._client._request("POST", f"/api/v1/agents/apps/sandbox/{action}/", **_request(action, app, resource, arguments, idempotency_key))
