# MCP sharing v1 - disabled pilot


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

Implementation: backend `apps/mcp_sharing`, Agents MCP `teamSharing.ts` and
`teamManagement.ts`, frontend `/mcp-teams`, Python sync/async `client.teams`.
Migration lives in independent app `mcp_sharing/0002_team_workspace`, not the
conflicting agents0147 chain. Media endpoint and its eight tools are unchanged;
media delegation remains a reviewed-adapter gate. See backend
`docs/mcp-team-api.md` for the REST/device protocol and
`docs/agents/mcp_sharing_developer_protocol.md` for adding/testing MCP types.

No deployment or public availability is implied. Existing MCP OAuth/team API
keys cannot manage membership or replace a device token. No copied owner keys
in client configurations. Metadata pages 25, tools pages 20 connections, request
64KiB, response 256 KiB, provider deadline 30s, edge 35s. Eight runtime requests per
process, two active calls/member, eight/team, 100 protocol requests/minute/team.
There are no unbounded sessions, result caches or automatic mutation retries.
