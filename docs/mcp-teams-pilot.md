# MCP team pilot

The server feature flag defaults off; `client.teams` does not make network calls
until a method is invoked. Use a verified personal API key for management.
Device tokens are only for the team's MCP endpoint. Existing client resources
and retry behavior remain unchanged; team mutations are never automatically
retried. Sync and async namespaces have the same methods.

```python
from flymyai.agents import SyncAgentClient

with SyncAgentClient() as client:  # FLYMYAI_API_KEY, verified personal account
    page = client.teams.list()
    for team in page.items:
        print(team.username, team.role, team.spent)
    # Fetch further pages explicitly with cursor=page.next_cursor.
```

Management methods cover enable/get/update, sources and connections, members
and role/limit changes, invite/preview/claim/decision, device creation/revoke,
usage/activity and propose/accept/cancel transfer. Creation requires explicit
billing consent. Invitation acceptance and member removal require
`accept_team_scope=True`, because these are existing teams with project/history
permissions. Owner approval is required after a claim.

Connection sharing requires exact `actions`, `revision`, `shared` and `consent`.
Only the credential owner may expand access. The read pilot supports Linear,
Notion and Composio Gmail; unavailable actions fail on the server.

Choose and retain an `idempotency_key` for invite/device creation before sending
the request. Reuse it after an uncertain transport outcome. Device secrets are
returned once as `SecretStr`; replay returns no token. Invite secrets also use
`SecretStr`. Their repr and JSON serialization are masked. Never log
`get_secret_value()` or place it in a URL/query string.

Pages have at most25 items; requests64KiB/responses256KiB; network timeout35s,
no redirects or decompression. Errors expose only a bounded code/status.
No page iterator, credentials cache or unbounded account hydration is added.

For runtime, use the team URL shown by the frontend and your own Bearer device
token. Initialize and retain its `Mcp-Session-Id`. JSON-RPC ids must be unique
within the session; explicit Idempotency-Key covers reconnect/retry. An unknown
outcome is terminal for automatic retry. Owner revoke/expiry is checked on each
request and before delivering a result.
