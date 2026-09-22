# Repository instructions

## MCP sharing contract (2026-09-21)

Every MCP is shareable by default as a design requirement: native API-key/OAuth,
Composio, user custom servers, embedded MCP, resource sets, Agents MCP and media
MCP. This does not make any connection public or grant access automatically.
The owner must explicitly authorize recipients and actions. Credentials stay in
the backend owner proxy; never copy owner keys into recipient configs or logs.
Record a sharing decision for every catalog type and surface. Any exception must
say `shareable: false` and explain why. A design decision is not runtime approval;
unreviewed adapters/operations fail closed until isolation and resource caps pass.

Scope tools and read/write semantics on the server; enforce expiry, call/spend
limits, owner billing, per-call metadata audit and immediate denial of new calls
after revoke. Already admitted provider operations may finish. Teams need their
own human membership model; agent groups are not human teams. Read-only access
still exposes owner data. Never trust tool names or remote readOnlyHint alone.

Required conformance: owner shares, authenticated recipient calls an allowed
read and an allowed write in an isolated fixture, forbidden actions fail, owner
revokes, new calls fail, and responses/logs contain no owner secrets. During the
proposed read-only pilot, the write case must be denied with zero dispatch; this
is not evidence that delegated writes work. Test token audience, session swapping,
concurrent quotas, idempotency, expiry, reconnect and revision changes too.

Bound SQL counts, input/output bytes, CPU, heap, sessions, caches and audit
retention. No N+1, full-account hydration or unbounded schema/payload logs.
The MCP sharing scaffold flag defaults off. Do not expose sharing tools/routes
or claim availability in user docs before the runtime contract is implemented.
