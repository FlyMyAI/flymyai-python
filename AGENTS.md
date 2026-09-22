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
limits, consenting-payer billing, per-call metadata audit and immediate denial of new calls
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


## Team-first MCP sharing (stage 2, 2026-09-22)

The primary v1 flow is a human team: invite people, explicitly share a connection,
use one team MCP URL with distinct member/device tokens, inspect usage and revoke.
Guest links are supplementary scoped grants, never implicit team membership.
"Shareable by default" requires an adapter sharing design; new connections remain
private until their credential owner consents to exact team actions.

Human teams are not AgentGroup or multiagent Agent Teams. Legacy users.User teams
and TeamMembership already exist and carry project/data permissions. Do not
reuse, migrate or widen those privileges implicitly when adding MCP membership.
Keep actor, credential owner and consenting billing payer distinct. Offboarding
revokes devices and the departing person's delegations; joining again must not
reactivate old credentials. Ownership transfer requires the recipient's explicit
acceptance of future billing, and never transfers personal provider credentials.

Conformance must exercise team roles, private/shared visibility, the same MCP URL
with two member tokens, device revoke, member removal/rejoin, transfer during a
billable call, bounded usage/audit pagination, and no inherited project authority.
A read-only pilot is an intermediate slice, not evidence of complete team v1 or
allowed-write support. Preserve the complete MCP type inventory and exceptions.
