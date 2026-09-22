# MCP sharing preparation - not released

Sharing is a required design capability for every MCP surface, not automatic
access to a connection. Runtime sharing is disabled and unimplemented here.

The planned owner proxy keeps connection credentials in the backend. A recipient
uses a grant-scoped token with a distinct audience. Existing API keys, REST
contracts, MCP tool lists, SDK namespaces and user routes keep their semantics.
The first pilot proposes verified read operations. Delegated writes and unknown
custom-server semantics require separate approval and conformance coverage.

New connectors must document credentials, actor/owner/billing identities, exact
action scope, expiry/revoke, budgets, audit and resource limits. Record explicit
`shareable: false` with a reason for exceptions. Test permitted read/write in a
fixture when supported; the read-only pilot must reject writes before dispatch.

Current preparation adds no delegated execution, invitation delivery, tokens,
billing, persistent models or user-visible sharing UI. No merge/deployment is
authorized by this note. The full proposal, mockup and evidence are sibling
workspace artifacts MCP_SHARING_PLAN.md, MCP_SHARING_MOCKUPS.html and
MCP_SHARING_REPORT_1.md. This note is intentionally outside published guides.
