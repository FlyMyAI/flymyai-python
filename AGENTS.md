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
after revoke. Already admitted provider operations may finish. Human identity uses existing User.is_team / TeamMembership with explicit MCP
consent; agent groups are not human teams. Read-only access
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

## Team MCP v1 decision (Denis, 2026-09-22, task 15)

Reuse existing `User.is_team` / `TeamMembership` for human identity. MCP policy is
an explicit overlay; legacy membership alone never grants shared MCP access.
Invitations require verified personal login, explicit consent to the existing
team project/history scope, and owner approval. Removing a member removes that
legacy membership too, with explicit consent. AgentGroup and multiagent Agent
Teams remain separate products. Link invitations join this same human team;
isolated guest-only grants are deferred, not silently approximated.

All MCP types are shareable by default as a design requirement, never public by
default. New connections remain private. Runtime approval requires exact owner
account, reviewed actions, live expiry/revoke checks, bounded resources and
conformance tests. Current pilot permits only reviewed Linear issue summaries,
Notion search and Composio Gmail summaries. Unknown actions and delegated writes
fail before provider dispatch. Keep the full type inventory and reasoned
`shareable: false` exceptions in the catalog invariant.

Team wallet pays MCP runtime; actor, credential owner and payer are distinct.
Defaults: 7 days (maximum 30), invitation 24h, 500 calls/$5 monthly, audit 30 days.
Roles: owner/admin/member/viewer. Viewer has no device/runtime authority. Device
secrets are shown once; owner credentials stay server-side. Member offboarding
revokes personal delegations and devices; rejoin never reactivates old tokens.
Ownership transfer needs recipient acceptance of billing and retains the team
wallet. Personal credentials do not transfer. Existing in-flight charges keep
their payer and settle exactly once. MCP usage and an agent's own LLM cost are
separate charges.

Flags default off. No old API/MCP behavior or schemas change. Safety maintenance
may refund existing sharing holds and purge expired sharing data while dispatch
is off; it cannot create new calls or send invitations. Test account/session
swaps, two device tokens at one URL, live revoke, role/expiry changes, transfer
during a call, concurrent quotas, duplicate/unknown outcomes, output projections,
secret-free logs and finite retention. Pilot write tests prove denial, not write
support. Measure SQL counts, CPU, RSS and response bytes; do not hydrate accounts.


## Personal MCP sharing decision (Denis, 2026-09-23)

The primary UI is one MCP entry with My MCPs, Shared by me and Shared with me.
Personal sharing is an exact-connection grant to a verified email and does not
create TeamMembership or grant project/history access. Sending the addressed
invitation is the owner's approval; the verified recipient may register first
and accept without another owner approval. The verified recipient consents to and pays FlyMy execution from their own
wallet. Credential owner, caller and payer are separate. Never fall back to the
owner's wallet; provider charges remain separate.
Team sharing remains a separate existing User.is_team / TeamMembership flow with
explicit project/history consent, owner approval and the team wallet. This
supersedes the earlier deferral of isolated personal grants.

All MCP types remain shareable by default as a design requirement, never public
by default. Reviewed reads only; no new write or generic proxy adapters. Scope
personal device tokens to one grant; adding another grant must not expand old
tokens. Revocation and renewed source consent invalidate old devices. Email
links carry a public UUID only; authorization always requires the verified
addressed account. Browser return intent contains no email or bearer secret and
expires in 24 hours. Do not persist legacy bearer invitation fragments.

Test registration continuation, wrong/unverified email, two aliases of one MCP,
one-grant/device revoke, source reconnect, account/session swap, exact device
audience, recipient-pays reserve/settle/refund, idempotency and limits, bounded SQL/bytes/RSS, retention and
flag-off behavior. Personal recipients must not consume the owner's control
budget or another recipient's protocol budget. Preserve legacy APIs and teams.
