"""Custom MCP servers: an MCP server the user adds by URL, for a service that is not
in the catalog or a provider's own MCP endpoint. The routes live on the agents host
under ``/api/v1/agents/mcp-servers/``.

``auth_type`` is ``none``, ``api_key`` (``auth_credentials={"api_key": ...}``),
``bearer_token`` (``{"token": ...}``), ``basic`` (``{"username": ..., "password":
...}``) or ``oauth``. With ``oauth`` FlyMyAI discovers the server's authorization (the
MCP Authorization spec) and registers itself where the provider allows it;
``oauth_client_id`` / ``oauth_client_secret`` / ``oauth_scopes`` bring the user's own
OAuth app for a provider without automatic registration (Slack's MCP).
:meth:`McpServers.authorize` returns the ``authorize_url`` a person opens in a browser;
after consent the server is connected. Credentials, client secrets and tokens are
write-only: no answer carries them. Unknown answer fields are kept on the models
(``extra="allow"``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from flymyai.agents._artifacts import _body, _key

_ROOT = "/api/v1/agents/mcp-servers/"
AUTH_TYPES = ("none", "api_key", "bearer_token", "basic", "oauth")
TRANSPORTS = ("streamable_http", "sse")


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)


class McpServerOAuth(_Model):
    """The owner's view of an OAuth server's authorization (never a token or a
    secret): ``authorized`` (a grant is stored), ``reconnect_required`` (authorize
    again), ``client`` (``user`` = your own OAuth app, ``registered`` = registered
    automatically) and ``redirect_uri`` (register it in your own OAuth app)."""

    authorized: bool = False
    reconnect_required: bool = False
    issuer: Optional[str] = None
    scopes: List[str] = []
    expires_at: Optional[str] = None
    client: Optional[str] = None
    client_id: Optional[str] = None
    has_client_secret: bool = False
    requested_scopes: Optional[str] = None
    redirect_uri: Optional[str] = None


class McpServer(_Model):
    """One custom MCP server: ``status`` pending, connected or error, the tools it
    discovered and, for an OAuth server's owner, ``oauth``."""

    id: int
    public_id: Optional[str] = None
    name: str
    url: str
    transport_type: Optional[str] = None
    auth_type: str
    has_auth_credentials: bool = False
    oauth: Optional[McpServerOAuth] = None
    is_active: bool = True
    status: str
    status_detail: str = ""
    discovered_tools: Any = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class McpServerAuthorization(_Model):
    """Answer of :meth:`McpServers.authorize`: open ``authorize_url`` in a browser
    (it works once, until ``expires_at``); consent connects ``server``."""

    authorize_url: str
    redirect_uri: str
    expires_at: str
    server: McpServer


def _server_id(server_id: Any) -> int:
    value = int(server_id)
    if value < 1:
        raise ValueError("server_id must be the id of one of your MCP servers")
    return value


def _path(server_id: Any, action: str = "") -> str:
    return f"{_ROOT}{_server_id(server_id)}/{action}"


def _fields(
    *,
    name: Optional[str],
    url: Optional[str],
    transport_type: Optional[str],
    auth_type: Optional[str],
    auth_credentials: Optional[Dict[str, str]],
    oauth_client_id: Optional[str],
    oauth_client_secret: Optional[str],
    oauth_scopes: Optional[str],
    is_active: Optional[bool],
) -> Dict[str, Any]:
    if auth_type is not None and auth_type not in AUTH_TYPES:
        raise ValueError(f"auth_type must be one of {', '.join(AUTH_TYPES)}")
    if transport_type is not None and transport_type not in TRANSPORTS:
        raise ValueError(f"transport_type must be one of {', '.join(TRANSPORTS)}")
    oauth_fields = (oauth_client_id, oauth_client_secret, oauth_scopes)
    if auth_type == "oauth":
        if auth_credentials:
            raise ValueError(
                "auth_type oauth takes no auth_credentials: pass oauth_client_id and"
                " oauth_client_secret for your own OAuth app instead"
            )
        if url is not None and not url.startswith("https://"):
            raise ValueError("auth_type oauth needs an https:// server url")
    elif auth_type is not None and any(field is not None for field in oauth_fields):
        raise ValueError("oauth_client_id, oauth_client_secret and oauth_scopes need auth_type oauth")
    return _body(
        name=name,
        url=url,
        transport_type=transport_type,
        auth_type=auth_type,
        auth_credentials=auth_credentials,
        oauth_client_id=oauth_client_id,
        oauth_client_secret=oauth_client_secret,
        oauth_scopes=oauth_scopes,
        is_active=is_active,
    )


def _servers(data: Any) -> List[McpServer]:
    rows = data.get("results", []) if isinstance(data, dict) else data
    return [McpServer.model_validate(row) for row in rows or []]


class McpServers:
    """``client.mcp_servers``: add, authorize, connect and call custom MCP servers."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def list(self) -> List[McpServer]:
        """Your custom MCP servers, by name."""
        return _servers(self._c._request("GET", _ROOT))

    def get(self, server_id: int) -> McpServer:
        return McpServer.model_validate(self._c._request("GET", _path(server_id)))

    def create(
        self,
        *,
        name: str,
        url: str,
        transport_type: str = "streamable_http",
        auth_type: str = "none",
        auth_credentials: Optional[Dict[str, str]] = None,
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
        oauth_scopes: Optional[str] = None,
        is_active: bool = True,
    ) -> McpServer:
        """Add a server. For ``auth_type="oauth"`` call :meth:`authorize` next; for the
        other types :meth:`connect` discovers its tools."""
        body = _fields(
            name=name,
            url=url,
            transport_type=transport_type,
            auth_type=auth_type,
            auth_credentials=auth_credentials,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
            oauth_scopes=oauth_scopes,
            is_active=is_active,
        )
        return McpServer.model_validate(self._c._request("POST", _ROOT, json=body))

    def update(
        self,
        server_id: int,
        *,
        name: Optional[str] = None,
        url: Optional[str] = None,
        transport_type: Optional[str] = None,
        auth_type: Optional[str] = None,
        auth_credentials: Optional[Dict[str, str]] = None,
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
        oauth_scopes: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> McpServer:
        """Change only the fields given. Saving ``oauth_client_id`` /
        ``oauth_client_secret`` keeps the current grant (authorize again to use them);
        changing ``url`` or ``auth_type`` drops it."""
        body = _fields(
            name=name,
            url=url,
            transport_type=transport_type,
            auth_type=auth_type,
            auth_credentials=auth_credentials,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
            oauth_scopes=oauth_scopes,
            is_active=is_active,
        )
        data = self._c._request("PATCH", _path(server_id), json=body)
        return McpServer.model_validate(data)

    def delete(self, server_id: int) -> None:
        self._c._request("DELETE", _path(server_id))

    def authorize(self, server_id: int) -> McpServerAuthorization:
        """Start the OAuth sign-in of an ``auth_type="oauth"`` server and return the
        ``authorize_url`` for a person to open. A provider without automatic client
        registration raises :class:`McpServerOAuthError` ``oauth_client_required`` with
        the ``redirect_uri`` to register in your own OAuth app."""
        data = self._c._request("POST", _path(server_id, "authorize/"))
        return McpServerAuthorization.model_validate(data)

    def connect(self, server_id: int) -> McpServer:
        """Open a session and store the tools the server lists."""
        data = self._c._request("POST", _path(server_id, "connect/"))
        return McpServer.model_validate(data)

    def disconnect(self, server_id: int) -> McpServer:
        """Reset the status and tools; an OAuth server also forgets its grant."""
        data = self._c._request("POST", _path(server_id, "disconnect/"))
        return McpServer.model_validate(data)

    def call(
        self,
        server_id: int,
        action: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        idempotency_key: str,
    ) -> Any:
        """Call one tool of a connected server directly. The call may change
        something on the provider: reserve ``idempotency_key`` before it and reuse it
        only to reconcile this same call."""
        body = {"action": action, "arguments": arguments or {}}
        return self._c._request(
            "POST", _path(server_id, "call/"), json=body, headers=_key(idempotency_key)
        )


class AsyncMcpServers:
    """``client.mcp_servers`` on the async client."""

    def __init__(self, client: Any) -> None:
        self._c = client

    async def list(self) -> List[McpServer]:
        return _servers(await self._c._request("GET", _ROOT))

    async def get(self, server_id: int) -> McpServer:
        return McpServer.model_validate(await self._c._request("GET", _path(server_id)))

    async def create(
        self,
        *,
        name: str,
        url: str,
        transport_type: str = "streamable_http",
        auth_type: str = "none",
        auth_credentials: Optional[Dict[str, str]] = None,
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
        oauth_scopes: Optional[str] = None,
        is_active: bool = True,
    ) -> McpServer:
        body = _fields(
            name=name,
            url=url,
            transport_type=transport_type,
            auth_type=auth_type,
            auth_credentials=auth_credentials,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
            oauth_scopes=oauth_scopes,
            is_active=is_active,
        )
        data = await self._c._request("POST", _ROOT, json=body)
        return McpServer.model_validate(data)

    async def update(
        self,
        server_id: int,
        *,
        name: Optional[str] = None,
        url: Optional[str] = None,
        transport_type: Optional[str] = None,
        auth_type: Optional[str] = None,
        auth_credentials: Optional[Dict[str, str]] = None,
        oauth_client_id: Optional[str] = None,
        oauth_client_secret: Optional[str] = None,
        oauth_scopes: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> McpServer:
        body = _fields(
            name=name,
            url=url,
            transport_type=transport_type,
            auth_type=auth_type,
            auth_credentials=auth_credentials,
            oauth_client_id=oauth_client_id,
            oauth_client_secret=oauth_client_secret,
            oauth_scopes=oauth_scopes,
            is_active=is_active,
        )
        data = await self._c._request("PATCH", _path(server_id), json=body)
        return McpServer.model_validate(data)

    async def delete(self, server_id: int) -> None:
        await self._c._request("DELETE", _path(server_id))

    async def authorize(self, server_id: int) -> McpServerAuthorization:
        data = await self._c._request("POST", _path(server_id, "authorize/"))
        return McpServerAuthorization.model_validate(data)

    async def connect(self, server_id: int) -> McpServer:
        data = await self._c._request("POST", _path(server_id, "connect/"))
        return McpServer.model_validate(data)

    async def disconnect(self, server_id: int) -> McpServer:
        data = await self._c._request("POST", _path(server_id, "disconnect/"))
        return McpServer.model_validate(data)

    async def call(
        self,
        server_id: int,
        action: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        idempotency_key: str,
    ) -> Any:
        body = {"action": action, "arguments": arguments or {}}
        return await self._c._request(
            "POST", _path(server_id, "call/"), json=body, headers=_key(idempotency_key)
        )
