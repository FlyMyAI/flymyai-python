# FlyMy.AI

<p align="center">
  <pre>
  ᕦ[▀̿_▀̿]ᕤ  ╓"Hi!"        ◢▆[◉◡◉]▆◣        o=[•_•]=o        ╾━[⊙▂⊙]━╼  ╓"Generate image"

  ¬[°□°]¬     凸[¬_¬]凸     <|¤_¤|>     ʕ[•ᴥ•]ʔ ╓"beep boop"     ╤[■_■]╤     d[-_-]b

  q[◔౪◔]p ╓"Freeze it!"     \[T_T]/     ▟[•̪●]▙       ¬[•‿•]╦═ =^.^= ═╦[•‿•]¬
</pre>
</p>

Welcome to FlyMy.AI inference platform. Our goal is to provide the fastest and most affordable deployment solutions for neural networks and AI applications.

- **Fast Inference**: Experience the fastest Stable Diffusion inference globally.
- **Scalability**: Autoscaling to millions of users per second.
- **Ease of Use**: One-click deployment for any publicly available neural networks.

## Website

For more information, visit our website: [FlyMy.AI](https://flymy.ai)
Or connect with us and other users on Discord: [Join Discord](https://discord.com/invite/t6hPBpSebw)

## Getting Started

This is a Python client for [FlyMyAI](https://flymy.ai). It allows you to easily run models and get predictions from your Python code in sync and async mode.

## Requirements

- Python 3.8+

## Installation

Install the FlyMyAI client using pip:

```sh
pip install flymyai
```

## Authentication

Before using the client, you need to have your API key, username, and project name. In order to get credentials, you have to sign up on flymy.ai and get your personal data on [the profile](https://app.flymy.ai/profile).

> 📚 **Core documentation:** [docs.flymy.ai](https://docs.flymy.ai) — full guides for agents, inference, and MCP tools.

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
            agent_id=agent.id, variables={"topic": "Tesla", "date": "2026-05-21"}
        )
        async for event in client.runs.stream_events(run.id):
            print(f"[{event.type}] {event.message}")
        result = await client.runs.wait(run.id)
        print(result.output)

        # 4. Chat: append a follow-up message and continue the same run
        await client.runs.append_message(run.id, text="Make it punchier.")
        await client.runs.wait(run.id)

        # 5. Freeze into a reusable instruction, re-run with fresh variables — your API
        compilation = await client.agents.compile_from_run(run.id)
        later = await client.compilations.run_instruction_and_wait(
            compilation.id, variables={"topic": "Bitcoin", "date": "2026-05-19"}
        )
        print(later.output)

asyncio.run(main())
```

Other agent methods: `client.tools.available()` / `provide_config()` / `call()`, `client.runs.get()` / `list()` / `cancel()`, `client.agents.update()` / `suggest_schema()`, `client.compilations.update()` (edit a frozen instruction). A synchronous `AgentClient` with the same method names (no `await`) is also available. Full reference: [docs.flymy.ai/agents](https://docs.flymy.ai/agents).

## Neural Network Inference

Run any model on the platform with `flymyai.async_run` (async) or `flymyai.run` (sync).

#### Image generation — Nano Banana 🍌

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
        f.write(base64.b64decode(response.output_data["image"]))

asyncio.run(main())
```

#### Video generation — Veo 3.1 Fast

```python
import asyncio
import flymyai

async def main():
    response = await flymyai.async_run(
        apikey="fly-secret-key",
        model="flymyai/veo31-fast-generate",
        payload={"prompt": "a red sports car driving along a coastal road at sunset, cinematic"},
    )
    print(response.output_data["video"])  # public URL to the generated .mp4

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
            f.write(base64.b64decode(r.output_data["image"]))

asyncio.run(main())
```

## M1 Agent Usage

### Using Synchronous Client

```python
from flymyai import m1_client

client = m1_client(apikey="fly-secret-key")
result = client.generate("An Iron Man")
print(result.data.text, result.data.file_url)
```

FlymyAI M1 client also stores request history for later generation context:

```python
from flymyai import m1_client

client = m1_client(apikey="fly-secret-key")

result = client.generate("An Iron Man")
print(result.data.text, result.data.file_url)

result = client.generate("Add him Captain America's shield")
print(result.data.text, result.data.file_url)
```

#### Passing image

```python
from pathlib import Path
from flymyai import m1_client

client = m1_client(apikey="fly-secret-key")
result = client.generate("An Iron Man", image=Path("./image.png"))
print(result.data.text, result.data.file_url)
```

### Using Asynchronous Client

```python
import asyncio
from flymyai import async_m1_client


async def main():
    client = async_m1_client(apikey="fly-secret-key")
    result = await client.generate("An Iron Man")
    print(result.data.text, result.data.file_url)


asyncio.run(main())
```

#### Passing image

```python
import asyncio
from pathlib import Path
from flymyai import async_m1_client


async def main():
    client = async_m1_client(apikey="fly-secret-key")
    result = await client.generate("An Iron Man", image=Path("./image.png"))
    print(result.data.text, result.data.file_url)


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
    await client.agents.run(agent.id, variables={})
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
