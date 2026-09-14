<p align="center">
  <img src="assets/flymyai-logo.png" alt="FlyMy.AI" width="260" />
</p>

<h3 align="center">FlyMy.AI - the Agentic Cloud.</h3>

<p align="center">
  <pre>
ᕦ[▀̿_▀̿]ᕤ ⚡ ◢▆[◉◡◉]▆◣ ⚡ o=[•_•]=o ⚡ ╾━[⊙▂⊙]━╼ ⚡ ¬[°□°]¬ ⚡ 凸[¬_¬]凸 ⚡ <|¤_¤|> ⚡ ʕ[•ᴥ•]ʔ ⚡ ╚[ʘᗜʘ]╝ ⚡ d[-_-]b ⚡ q[◔౪◔]p ⚡ \[T_T]/ ⚡ ✧[◕‿◕]✧ ⚡ =^.^=
</pre>
</p>

**FlyMy.AI is the Agentic Cloud - Agents. Models. Serverless.**

Ship autonomous AI workers that plan, call tools (MCP), and deliver results - not just chat. Access any model through a unified API, or deploy your own custom models on serverless GPUs. One platform, pay per use, production-ready in minutes.

- **Agents**: build autonomous workers that plan, execute tools, and return structured results - then freeze a good run into a reusable, deterministic instruction you call as an API.
- **Models**: run any model through one unified API (image, video, audio, LLMs) in sync or async mode.
- **Serverless**: deploy your own custom models on autoscaling GPUs.
- **MCP tools**: plug in any MCP - web search, browsers, files, external APIs.

Agents tie everything together; Models and Serverless also work standalone - pick what you need and plug the rest in later.

## Website

For more information, visit [FlyMy.AI](https://flymy.ai), read the [docs](https://docs.flymy.ai), or join us on [Discord](https://discord.com/invite/t6hPBpSebw).

## Getting Started

This is the Python client for [FlyMy.AI](https://flymy.ai). Build and run agents, call any model, and drive serverless endpoints from Python - in sync or async mode.

## Requirements

- Python 3.8+

## Installation

Install the FlyMyAI client using pip:

```sh
pip install flymyai
```

## Authentication

Before using the client, you need to have your API key, username, and project name. In order to get credentials, you have to sign up on flymy.ai and get your personal data on [the profile](https://app.flymy.ai/profile).

> 📚 **Core documentation:** [docs.flymy.ai](https://docs.flymy.ai) - full guides for agents, inference, and MCP tools.

## Agents

Autonomous agents plan, call tools (MCP), and return structured results. Declare an `input_schema` to make an agent reusable with runtime `{{ variables }}`, then freeze a good run into a Markdown instruction you can re-run as an API.

```python
import asyncio
from flymyai import AsyncAgentClient

async def main():
    async with AsyncAgentClient(api_key="fly-secret-key") as client:
        # 1. Attach a tool (browse the full catalog with client.tools.available())
        tool = await client.tools.create(mcp_tool="tavily")

        # 2. Create a reusable agent. {{ variables }} require an input_schema.
        agent = await client.agents.create(
            name="News Brief",
            goal="Find the biggest news about {{ topic }} on {{ date }}. Return a one-line headline.",
            tools=[tool.id],
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string"}, "date": {"type": "string"}},
                "required": ["topic", "date"],
            },
        )

        # 3. Run with variables; stream progress; get the structured result
        run = await client.runs.create(
            agent_id=agent.id,
            idempotency_key="news-brief-tesla-2026-05-21-v1",
            variables={"topic": "Tesla", "date": "2026-05-21"},
        )
        since, run_seq = 0, run.run_seq
        async for event in client.runs.stream_events(run.id, run_seq=run_seq):
            print(f"[{event.type}] {event.message}")
            since, run_seq = event.id, event.observed_run_seq
        result = await client.runs.wait(run.id, since=since, run_seq=run_seq)
        # output is complete only when result.result is inline and not truncated.
        print(result.output)

        # 4. Chat: append a follow-up message and continue the same run
        resumed = await client.runs.append_message(run.id, text="Make it punchier.")
        await client.runs.wait(
            run.id, since=result.next_since, run_seq=resumed.run_seq,
            presentation_cursor=result.presentation_cursor_v1,
        )

        # 5. Freeze into a reusable instruction, re-run with fresh variables - your API
        compilation = await client.agents.compile_from_run(run.id)
        later = await client.compilations.run_instruction_and_wait(
            compilation.id,
            idempotency_key="news-brief-bitcoin-2026-05-19-v1",
            variables={"topic": "Bitcoin", "date": "2026-05-19"},
        )
        print(later.output)

asyncio.run(main())
```

Other agent methods: `client.tools.available()` / `provide_config()` / `call()`, `client.runs.get()` / `list()` / `cancel()`, `client.agents.update()` / `suggest_schema()`, `client.compilations.update()` (edit a frozen instruction). A synchronous `AgentClient` with the same method names (no `await`) is also available. Full reference: [docs.flymy.ai/agents](https://docs.flymy.ai/agents).

### Bounded run observation

`runs.wait()` and the instruction/deployment `*_and_wait()` helpers now return
`RunStatus`, not `RunDetail`. They poll `status/?view=bounded_v1` with 20 steps
per request and drain every `has_more` page before accepting `poll_complete`.
`stream_events()` yields compact `RunStep` values, not full `ExecutionLog`
objects. Labels and messages can be truncated; these events omit full log
`data` and timestamps. No history or seen-ID set accumulates in either iterator.

For manual observation, use `runs.status(id, since=last.next_since)` and retain
`run_seq` plus `presentation_cursor_v1`. Pass those scalars to `wait()` or
`stream_events()` when reconnecting. Log IDs continue across resumed runs;
`observed_run_seq` identifies the generation observed, not the generation in
which an older log was written. Only `poll_complete` ends observation. A goal
with status `verified` is progress; it does not mean the run is complete.

Result and error bodies are `RunResource` references. `status.output` returns
only a complete inline result and raises `ValueError` for a truncated result.
Read larger results or errors explicitly, with each range at most 65,536 bytes:

```python
import codecs
import hashlib

settled = client.runs.wait(run.id, run_seq=run.run_seq)
resource = settled.result  # settled.error uses the same range API
if resource is not None:
    offset = 0
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")()
    while offset < resource.size_bytes:
        chunk = client.runs.read_resource(run.id, resource, offset=offset, limit=65536)
        offset += len(chunk)
        digest.update(chunk)
        print(decoder.decode(chunk, final=offset == resource.size_bytes), end="")
    if digest.hexdigest() != resource.sha256:
        raise ValueError("Resource digest mismatch")
```

This example retains one range and decoder state. It does not assemble the
whole JSON document; callers needing that must choose their own storage and
size limits. Individual ranges can split UTF-8 characters or JSON tokens.
Ranges require the current owner's authorization; a stale resource reference
returns HTTP 409. Re-read status to obtain the current generation's reference.
The SDK validates range/ref/digest headers and hashes a complete single-range
resource. Multi-range consumers should hash their complete stream as above.

`runs.transcript(id, cursor=...)` and `runs.logs(id, cursor=...)` explicitly read
one history page (20 entries by default). The next cursor selects older history;
transcript messages within a page are chronological. A stale transcript cursor
returns HTTP 409; start again without the cursor. Preserve or discard each page
explicitly instead of accumulating account or run history in memory.

`runs.get(id)` remains the explicit legacy full-detail API, including full
messages/logs where the server supplies them. It can be large. Observation never
falls back to it: an old server's missing route raises `FlyMyAIAgentError`, and
an ignored bounded view or oversized response raises
`flymyai.agents.RunObservationUnsupportedError`. Malformed bounded pages fail
validation. Bounded GETs refuse redirects/compressed bodies and accept at most
512 KiB per page body or 64 KiB per resource range. Parsing and transport retain
additional bounded copies. Observation makes no automatic writes.

### Cancellation is a request

`runs.cancel(id)` sends one request. A successful return (HTTP 204) does not
confirm cancellation, and a transport exception leaves its outcome uncertain.
Continue the same observation with `wait()` or `stream_events()`. The server
may confirm `completed`, `failed`, `cancelled`, or `archived`; completion can win
the race with Stop. Retry cancellation only as an explicit caller decision.
Closing an iterator, an async task cancellation, or `TimeoutError` stops only
local observation and never cancels the server run. Event-stream timeouts now
raise `TimeoutError` instead of silently looking like normal completion.

## Personal connection first

A new user starts in one implicit personal space. The first connection of a
toolkit is the obvious `default` connection. Give that direct connection to
the same agent and run it without naming projects, groups, slots, principals,
revisions, or mappings.

```python
import time
import webbrowser

from flymyai import AgentClient, McpAccessMode

with AgentClient(api_key="fly-secret-key") as client:
    # Omitting alias keeps the first personal Gmail connection on `default`.
    gmail = client.tools.create(mcp_tool="gmail")

    # Gmail setup is hosted. Open the returned URL, finish OAuth, then wait for
    # the same connection row to become ready before assigning it or running.
    if not gmail.is_configured:
        if gmail.redirect_url:
            webbrowser.open(gmail.redirect_url)
            input("Finish Gmail authorization, then press Enter: ")
        elif gmail.next_configuration_step:
            raise RuntimeError(
                "This connector needs an answer. Inspect "
                "gmail.next_configuration_step and call "
                "client.tools.provide_config(gmail.id, user_response=...)."
            )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            gmail = client.tools.get(gmail.id)
            if gmail.is_configured:
                break
            time.sleep(2)
        else:
            raise TimeoutError("Gmail authorization did not become ready in 5 minutes.")

    agent = client.agents.create(
        name="Personal inbox helper",
        goal="Summarize my unread mail.",
        tools=[gmail.id],
        mcp_access_mode=McpAccessMode.LEGACY,
    )
    run = client.agents.run(
        agent.id,
        idempotency_key="personal-inbox-summary-v1",
    )
```

`legacy` here is the explicit compatibility projection for direct personal
connections. The response also exposes `agent.mcp_access_mode`; old server
responses that omit it parse as `McpAccessMode.LEGACY`. This simple path uses
the same exact connection row as the controls below, so it can grow without
rebuilding the agent or reconnecting the account.

## Advanced connection access

Open the advanced model only when the same agent needs a second account of one
toolkit, reusable sharing, group access, separate clients or environments, or
explicit policy and mapping control.

```python
from flymyai import (
    AgentClient,
    McpAccessMode,
    McpResourceSetManagementMode,
    McpResourceSetMemberInput,
    McpResourceType,
)

with AgentClient(api_key="fly-secret-key") as client:
    support = client.tools.create(mcp_tool="gmail", alias="support_mail")
    sales = client.tools.create(mcp_tool="gmail", alias="sales_mail")

    resource_set = client.mcp_resource_sets.create(
        name="Mail operations",
        management_mode=McpResourceSetManagementMode.FLYMYAI,
    )
    resource_set = client.mcp_resource_sets.replace_members(
        resource_set.id,
        expected_revision=resource_set.revision,
        members=[
            McpResourceSetMemberInput(
                resource_type=McpResourceType.USER_MCP_TOOL,
                resource_id=support.public_id,
                slot="support_read",
                allowed_actions=["GMAIL_SEARCH_EMAILS"],
            ),
            McpResourceSetMemberInput(
                resource_type=McpResourceType.USER_MCP_TOOL,
                resource_id=support.public_id,
                slot="support_send",
                allowed_actions=["GMAIL_SEND_EMAIL"],
            ),
            McpResourceSetMemberInput(
                resource_type=McpResourceType.USER_MCP_TOOL,
                resource_id=sales.public_id,
                slot="sales_mailbox",
            ),
        ],
    )
    agent = client.agents.create(
        name="Mail operations",
        goal="Work in the mailbox selected for this request.",
        mcp_resource_set_ids=[resource_set.id],
        mcp_access_mode=McpAccessMode.SCOPED,
    )

    # Metadata writes use the revision that this editor originally loaded.
    resource_set = client.mcp_resource_sets.update(
        resource_set.id,
        expected_revision=resource_set.revision,
        description="Support and sales mailboxes",
    )
```

Aliases and slots accept 1 to 64 and 1 to 128 characters respectively, using
only ASCII letters, digits, underscores, and hyphens. They are labels, not
authority. Select resources with stable public UUIDs. `client.tools.list()`
follows bounded cursor pages and can filter by exact `mcp_tool` or `alias`.
Metadata `update()`
uses PATCH, complete metadata `replace()` uses PUT, and member replacement
uses `replace_members()`; all three require `expected_revision`.

`client.mcp_resource_sets.list()` and `client.agent_groups.list()` follow the
backend's compact cursor envelope across pages. They reject malformed or
repeated cursors and stop at fixed page and row safety limits. A bounded plain
array remains accepted for compatibility with an older server during rollout.
Resource-set list rows are `McpResourceSetSummary` values with `member_count`
and no nested members. Use `client.mcp_resource_sets.get(id)` for one bounded
full set or `client.mcp_resource_sets.list_members(id)` to traverse a large
membership collection. Sync and async clients expose the same contract.

Member identity is the exact `(resource_type, resource_id, slot)` tuple. The
same resource UUID may therefore appear in several different slots, as the
support connection does above, and every slot keeps its own
`allowed_actions` ceiling. Repeating the same tuple is invalid. Members pooled
inside one slot must use the same action ceiling.

Returning a scoped agent to personal legacy access is intentionally a safe
multi-request workflow. First remove the agent from every group while
preserving each group's other agents and resource sets, then clear its direct
resource-set grants. Only after every cleanup request succeeds, switch the mode
in a separate PATCH:

```python
for group in client.agent_groups.list():
    if agent.id in group.agent_ids:
        client.agent_groups.replace_assignments(
            group.id,
            agent_ids=[item for item in group.agent_ids if item != agent.id],
            resource_set_ids=group.resource_set_ids,
        )

client.agents.update(agent.id, mcp_resource_set_ids=[])
client.agents.update(agent.id, mcp_access_mode=McpAccessMode.LEGACY)
```

If any cleanup request fails, do not send the final mode PATCH. The agent then
remains fail-closed in `scoped` mode. The SDK deliberately does not hide these
independent writes in a convenience helper: automatic rollback could restore
stale grants or overwrite another editor's group assignment.

An owner project uses `management_mode=McpResourceSetManagementMode.FLYMYAI`
and omits `principal_id`. A customer-managed named mapping uses the exact
typed principal UUID. The complete embedded flow below obtains that principal
with `access.principal_for_external_user(customer_id)`, creates the mapping
with `principal_id=principal.id`, installs exact connection UUID members, and
runs with the returned mapping revision.

The deprecated `external_principal_id` field is not an accepted SDK argument.
For a FlyMyAI-managed deployment run, omit both `resource_set_id` and
`connections` so saved bindings apply. Send `resource_set_id` only for a
customer-managed named mapping, or `connections` for a one-off explicit
mapping. The last two are mutually exclusive.

`AgentClient` and `AsyncAgentClient` call the Agents REST API. They expose
Python methods such as `client.agents.create()`,
`client.mcp_resource_sets.replace_members()`, and
`client.deployments.run()`. The Agents MCP gateway is a separate assistant
surface with its own discovered tool names. Do not translate a REST SDK method
into an invented MCP-only lifecycle call, and do not treat an MCP tool name as
a Python method.

## Embedded customer agents

An embedded deployment publishes one immutable frozen version while each
customer authorizes their own MCP accounts. Hosted authorization links must be
created by your trusted application backend because the FlyMyAI API key must
never be sent to the customer browser.

`client.deployments.access()` is read-only and never creates a principal. It
can inspect the published requirements. The first mutating customer bootstrap
is `client.deployments.create_connection_link()`, which calls REST
`POST /api/v1/agents/deployments/{deployment_id}/connect-session/` with the
stable `external_user_id` and one exact required slot.

Complete synchronous flow for a customer-managed named mapping:

```python
import os
import time
import webbrowser

from flymyai import (
    AgentClient,
    McpResourceSetManagementMode,
    McpResourceSetMemberInput,
    McpResourceType,
)

customer_id = os.environ["PRODUCT_CUSTOMER_ID"]  # Stable, non-secret product ID
agent_id = os.environ["FLYMYAI_AGENT_ID"]
accepted_run_id = int(os.environ["FLYMYAI_ACCEPTED_RUN_ID"])

with AgentClient(api_key="fly-secret-key") as client:
    # compile_from_run freezes and polls until compilation has finished.
    compilation = client.agents.compile_from_run(accepted_run_id)
    version = next(
        item
        for item in client.versions.list(agent_id=agent_id)
        if item.source_compilation == compilation.id
    )

    deployment = client.deployments.create(
        agent_id=agent_id,
        version_id=version.id,
        name="Production",
    )
    manifest = client.deployments.access(deployment.id)  # Read-only.
    preflight = client.deployments.preflight(
        deployment.id,
        publish_mode="embedded",
        version_id=version.id,
    )
    if not preflight.ready:
        raise RuntimeError("Deployment preflight is not ready.")
    deployment = client.deployments.publish(
        deployment.id,
        publish_mode="embedded",
        version_id=version.id,
    )

    # Each POST is the mutating principal/bootstrap step for one exact slot.
    sessions = [
        client.deployments.create_connection_link(
            deployment.id,
            external_user_id=customer_id,
            slot=requirement.slot,
        )
        for requirement in manifest.requirements
        if requirement.connection_required
    ]
    for session in sessions:
        webbrowser.open(session.redirect_url)
    if sessions:
        input("Finish every customer authorization, then press Enter: ")

    # Poll the filtered access projection. The typed helpers fail closed until
    # one exact principal and active connection IDs exist for every slot.
    deadline = time.monotonic() + 300
    while True:
        access = client.deployments.access(
            deployment.id,
            external_user_id=customer_id,
        )
        try:
            principal = access.principal_for_external_user(customer_id)
            ready_by_slot = {
                requirement.slot: access.ready_connection_ids_for_slot(
                    principal_id=principal.id,
                    slot=requirement.slot,
                )
                for requirement in access.requirements
                if requirement.connection_required
            }
            break
        except ValueError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Customer connections did not become ready.")
            time.sleep(2)

    customer_mapping = client.mcp_resource_sets.create(
        name=f"{customer_id} connections",
        management_mode=McpResourceSetManagementMode.CUSTOMER,
        principal_id=principal.id,
    )
    customer_mapping = client.mcp_resource_sets.replace_members(
        customer_mapping.id,
        expected_revision=customer_mapping.revision,
        members=[
            McpResourceSetMemberInput(
                resource_type=McpResourceType.INTEGRATION_CONNECTION,
                resource_id=connection_id,
                slot=requirement.slot,
                allowed_actions=requirement.exact_actions,
            )
            for requirement in access.requirements
            if requirement.connection_required
            for connection_id in ready_by_slot[requirement.slot]
        ],
    )
    result = client.deployments.run_and_wait(
        deployment.id,
        variables={"topic": "Q3 pipeline"},
        external_user_id=customer_id,
        resource_set_id=customer_mapping.id,
        resource_set_revision=customer_mapping.revision,
        idempotency_key=f"{customer_id}-q3-pipeline-v1",
    )
```

`AsyncAgentClient` exposes the same typed lifecycle. Await the network methods;
`principal_for_external_user()` and `ready_connection_ids_for_slot()` remain
ordinary local model helpers.

The access request must include the exact `external_user_id`; do not select a
principal or connection by display name, email, alias, provider account ID, or
row order. `McpResourceSets.list()` supports `query`, `authority_type`, and
`principal_id`; `AgentGroups.list()` supports `query`. These filters remain on
every compact-cursor page.

`client.deployments.run()` and `run_and_wait()` call the stable deployment
endpoint, so callers need the deployment UUID but never need a compilation ID.
Normally a run uses the saved customer bindings. To choose among several
authorized accounts for one run, pass
`connections={"sender_inbox": "8335876a-ee78-45db-9d49-0ae148bd0158"}` or a
list of up to 25 connection UUIDs for a multi-connection slot. Provider account
IDs and aliases are not accepted in this mapping. Customer file attachments
are not enabled for embedded runs in this beta. Wait for every hosted
authorization callback to complete before starting the customer's first run.

The legacy `client.compilations.run_instruction()` and
`run_instruction_and_wait()` methods remain available for compilation-scoped
owner runs. Deprecated `client.compilations.run()` and its async counterpart
fail locally before HTTP because that legacy endpoint has no caller-owned
replay contract.

Every `agents.run()`, `runs.create()`, `tools.call()`,
`compilations.run_instruction()`, and `deployments.run()` call, including their
wait helpers and async variants,
requires a caller-owned `idempotency_key`. It must be non-blank and no longer
than 255 characters and contain no control or non-printable characters. The
SDK forwards the exact value as `Idempotency-Key` and never generates one
for the caller. Reuse a key only for an identical retry; choose a new key for a new
logical execution. This is an intentional compatibility break for keyless
effect calls because a lost response must not create a second execution that
repeats external writes.

## Neural Network Inference

Run any model on the platform with `flymyai.async_run` (async) or `flymyai.run` (sync).

#### Image generation - Nano Banana 🍌

```python
import asyncio
import base64
import flymyai

async def main():
    response = await flymyai.async_run(
        apikey="fly-secret-key",
        model="flymyai/nano-banana",
        payload={"prompt": "a cute cat astronaut floating in a neon nebula, studio lighting"},
    )
    with open("nano_banana.jpg", "wb") as f:
        f.write(base64.b64decode(response.output_data["image"][0]))

asyncio.run(main())
```

#### Video generation - Veo 3.1 Fast

```python
import asyncio
import flymyai

async def main():
    response = await flymyai.async_run(
        apikey="fly-secret-key",
        model="flymyai/veo31-fast-generate",
        payload={"prompt": "a red sports car driving along a coastal road at sunset, cinematic"},
    )
    print(response.output_data["video"][0])  # public URL to the generated .mp4

asyncio.run(main())
```

#### Parallel generation

Fire many requests concurrently with `asyncio.gather`:

```python
import asyncio
import base64
import flymyai

PROMPTS = ["a neon city at night", "a serene mountain lake at dawn", "a retro robot barista"]

async def main():
    results = await asyncio.gather(*[
        flymyai.async_run(
            apikey="fly-secret-key",
            model="flymyai/nano-banana",
            payload={"prompt": p},
        )
        for p in PROMPTS
    ])
    for i, r in enumerate(results):
        with open(f"img_{i}.jpg", "wb") as f:
            f.write(base64.b64decode(r.output_data["image"][0]))

asyncio.run(main())
```

## Advanced agent helpers

#### Draft an `input_schema` from a prompt

```python
import asyncio
from flymyai import AsyncAgentClient

async def main():
    async with AsyncAgentClient(api_key="fly-secret-key") as client:
        suggestion = await client.agents.suggest_schema(
            user_prompt="Summarize {{ url }} in {{ n_sentences }} sentences.",
            generate_descriptions=True,
        )
        print(suggestion.input_schema)
        print(suggestion.input_description)

asyncio.run(main())
```

#### Handle invalid variables

When `variables` don't match the agent's `input_schema`, the server returns `HTTP 400` and the client raises `VariablesValidationError`:

```python
from flymyai import VariablesValidationError

try:
    await client.agents.run(
        agent.id,
        idempotency_key="validate-agent-input-v1",
        variables={},
    )
except VariablesValidationError as err:
    print(err.messages)      # ["'url' is a required property", ...]
    print(err.field_errors)  # {"url": "'url' is a required property"}
```

#### Draft schemas from a finished run

```python
# Infer schemas from a completed run's chat + tool trace
# (also persists them onto the source agent).
suggestion = await client.runs.suggest_schema(
    run.id,
    inputs_prompt="A URL and a sentence count",
    outputs_prompt="A short summary",
)
```

## Bounded Files V2 workspaces and artifacts

`client.files` uses the same canonical owner library/task workspaces as the
browser. The owner personal run default needs no workspace field. To share
files, first obtain the real `ws_` reference; names and guessed IDs do not
confer authority:

```python
# agent.id is the exact UUID returned by the owner Agents API.
personal = client.files.list(agent_uuid=agent.id, limit=20)
print(personal.workspace, personal.revision)

# One discovery page, with no implicit selection or automatic next-page walk.
choices = client.files.workspaces(limit=20)
for workspace in choices.workspaces:
    print(workspace.workspace, workspace.name)
# Request the next page explicitly with cursor=choices.next_cursor, if present.

# Subjects are also one page of owner IDs, without group membership hydration.
subjects = client.files.subjects("group", limit=20)
for subject in subjects.subjects:
    print(subject.id, subject.name)

# Choose the exact subject UUID and workspace from those responses.
grants = client.workspace_grants.list(selected_workspace, limit=20)
client.workspace_grants.grant(
    selected_workspace,
    subject_kind="group",
    subject_id=selected_group_id,
    role="write",
    expected_revision=grants.revision,
    idempotency_key=grant_operation_key,  # caller-owned, reserved before dispatch
)
# Pass workspace=selected_workspace to compilations.run_instruction() for an owner run.
```

`files.list()` without selectors bootstraps the owner's library. With
`agent_uuid`, it bootstraps that exact owner's agent workspace. With
`workspace`, it lists that exact selected workspace. Selectors are mutually
exclusive. All return `FilePage(workspace, revision, files, total, next_cursor)`.
`files.workspaces()` lists existing persistent workspaces; it does not create a
workspace per account agent. Pages default to 20 and accept at most 50 items.
Keep or discard each page explicitly; Files cursors are at most 128 characters
and grant no authority and a refresh
can reflect a newer workspace revision. This is not a snapshot tree.

Read a large `af_` artifact without sending it to a legacy integer download:

```python
meta = client.files.stat(artifact_ref, workspace=selected_workspace)  # server-issued af_ ref
print(meta.ref, meta.size, meta.sha256)
if meta.size:
    window = client.files.read(meta, workspace=selected_workspace, offset=0, limit=min(meta.size, 65536))
    print(window.ref, window.offset, window.total_bytes)
    # window.content is bytes. This window can split UTF-8 or JSON tokens.
    # Request another explicit offset with the SAME meta to keep the version.
```

`files.read(ref, offset=..., limit=...)` may instead make one bounded metadata
request first. It then reads exactly one version-pinned byte range. Pass
`workspace=page.workspace` to stat/read to select the exact binding when an
artifact is shared across workspaces. Omitting it never picks a first binding;
ambiguous bindings remain an explicit server error. It never
downloads the whole artifact or advances a cursor automatically. Zero-byte files
need only `stat`. Each range is 1..65536 bytes; metadata/pages are capped at
64 KiB/512 KiB of response body. Sync and async clients expose the same methods
and models; use `await` for each async method. Files models and
`FilesContractUnsupportedError` are exported from `flymyai.agents`.

`execution-resource:v1:...` result/error references remain separate: obtain them
from `client.runs.status()`/`wait()` and use `client.runs.read_resource()` with
its returned `RunResource`. Never pass these refs or `af_` refs into legacy
integer file APIs. Legacy full-detail/history methods remain explicit opt-ins.

The Files page/workspace/subject/meta routes and bounded content headers require
the matching backend source integration. Missing routes raise
`FlyMyAIAgentError`; unsupported/oversized/malformed Files responses raise
`FilesContractUnsupportedError`. No trailing-slash tree fallback is attempted.
Authorization, missing versions, 409 conflicts and 416 ranges remain explicit.
A read does not acquire a publication lease. Grant writes preserve server CAS;
a lost response is uncertain, and retries must be explicit identical replays
with the original idempotency key and expected revision. No provider effects
or automatic write retries are introduced by Files reads.
