# FlyMy.AI

<p align="center">
  <pre>
                            (( ◌ ))
                               │
                    ╔══════════╧══════════╗
                  ╔═╝░░░░░░░░░░░░░░░░░░░░░╚═╗
                  ║░╭───────────────────╮░║
                  ║░│   ▟▀▀▀▙     ▟▀▀▀▙   │░║
                  ║░│   ▐ ◉ ▌     ▐ ◉ ▌   │░║      ╭────────────────╮
                  ║░│   ▜▄▄▄▛     ▜▄▄▄▛   │░║◖─────┤  analyzing ...  │
                  ║░│        ▁▁▁▁▁        │░║      ╰────────────────╯
                  ║░│     ╰┴┴┴┴┴┴┴┴╯      │░║
                  ║░╰───────────────────╯░║
                  ╚═╗░░░░░░░░░░░░░░░░░░░░░╔═╝
            ▟▀▀▙    ╚═════════╤═════════╝    ▟▀▀▙
            █▒▒█▆▆▆▆▆▆▆╗       │       ╔▆▆▆▆▆▆▆█▒▒█
            ▜▄▄▛       ╚═╗ ╔═══╧═══╗ ╔═╝       ▜▄▄▛
                         ╚═╝▓▓▓▓▓▓▓╚═╝
                       ╔═══╝▓ F M A ▓╚═══╗
                       ║▓▓▓▓▓ AGENT ▓▓▓▓▓║
                       ║▓▓░▒▓██▓▒░▓▓▓▓▓▓▓║
                       ║▓▓ ⌬ · CORE · ⌬ ▓║
                       ╚═╗▓▓▓▓▓▓▓▓▓▓▓▓▓╔═╝
                         ║▆▆▆▆▆▆▆▆▆▆▆▆▆║
                      ╔══╝             ╚══╗
                   ▟██▛                   ▜██▙
                  ▟███▙                   ▟███▙
                  ▀▀▀▀▀                   ▀▀▀▀▀
</pre>
  </br>Generated with FlyMy.AI  <b>in 🚀 70ms </b>
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

## Agents

Autonomous agents plan, call tools (MCP), and return structured results. Declare an `input_schema` to make an agent reusable with runtime `{{ variables }}`, then freeze a good run into a Markdown instruction you can re-run as an API.

```python
from flymyai import AgentClient

client = AgentClient(api_key="fly-secret-key")

# 1. Attach a tool (browse the full catalog with client.tools.available())
tool = client.tools.create(mcp_tool="tavily")

# 2. Create a reusable agent. {{ variables }} require an input_schema.
agent = client.agents.create(
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
run = client.runs.create(agent_id=agent.id, variables={"topic": "Tesla", "date": "2026-05-21"})
for event in client.runs.stream_events(run.id):
    print(f"[{event.type}] {event.message}")
result = client.runs.wait(run.id)
print(result.output)

# 4. Chat: append a follow-up message and continue the same run
client.runs.append_message(run.id, text="Make it punchier.")
client.runs.wait(run.id)

# 5. Freeze into a reusable instruction, then re-run with fresh variables — your API
compilation = client.agents.compile_from_run(run.id)
later = client.compilations.run_instruction_and_wait(
    compilation.id, variables={"topic": "Bitcoin", "date": "2026-05-19"}
)
print(later.output)
```

Other agent methods: `client.tools.available()` / `provide_config()` / `call()`, `client.runs.get()` / `list()` / `cancel()`, `client.agents.update()` / `suggest_schema()`, `client.compilations.update()` (edit a frozen instruction). Use `AsyncAgentClient` with `await` for async. See more under **Agent variables, freeze & re-run** below and at [docs.flymy.ai/agents](https://docs.flymy.ai/agents).

## Neural Network Inference

Run any model on the platform with `flymyai.run` (sync) or `flymyai.async_run` (async).

#### BERT Sentiment analysis

```python
import flymyai

response = flymyai.run(
    apikey="fly-secret-key",
    model="flymyai/bert",
    payload={"text": "What a fabulous fancy building! It looks like a palace!"}
)
print(response.output_data["logits"][0])
```

#### Image generation — Nano Banana 🍌

```python
import base64
import flymyai

response = flymyai.run(
    apikey="fly-secret-key",
    model="flymyai/nano-banana",
    payload={"prompt": "a cute cat astronaut floating in a neon nebula, studio lighting"},
)
with open("nano_banana.jpg", "wb") as f:
    f.write(base64.b64decode(response.output_data["image"]))
```

#### Video generation — Veo 3.1 Fast

```python
import flymyai

response = flymyai.run(
    apikey="fly-secret-key",
    model="flymyai/veo31-fast-generate",
    payload={"prompt": "a red sports car driving along a coastal road at sunset, cinematic"},
)
print(response.output_data["video"])  # public URL to the generated .mp4
```

## Sync Streams

For llms you should use stream method

#### llama 3.1 8b

```python
from flymyai import client, FlyMyAIPredictException

fma_client = client(apikey="fly-secret-key")

stream_iterator = fma_client.stream(
    payload={
        "prompt": "tell me a story about christmas tree",
        "best_of": 12,
        "max_tokens": 1024,
        "stop": 1,
        "temperature": 1,
        "top_k": 1,
        "top_p": "0.95",
    },
    model="flymyai/llama-v3-1-8b"
)
try:
    for response in stream_iterator:
        if response.output_data.get("output"):
            print(response.output_data["output"].pop(), end="")
except FlyMyAIPredictException as e:
    print(e)
    raise e
finally:
    print()
    print(stream_iterator.stream_details)
```

## Async Streams

For llms you should use stream method

#### Stable Code Instruct 3b

```python
import asyncio

from flymyai import async_client, FlyMyAIPredictException


async def run_stable_code():
    fma_client = async_client(apikey="fly-secret-key")
    stream_iterator = fma_client.stream(
        payload={
            "prompt": "What's the difference between an iterator and a generator in Python?",
            "best_of": 12,
            "max_tokens": 512,
            "stop": 1,
            "temperature": 1,
            "top_k": 1,
            "top_p": "0.95",
        },
        model="flymyai/Stable-Code-Instruct-3b"
    )
    try:
        async for response in stream_iterator:
            if response.output_data.get("output"):
                print(response.output_data["output"].pop(), end="")
    except FlyMyAIPredictException as e:
        print(e)
        raise e
    finally:
        print()
        print(stream_iterator.stream_details)


asyncio.run(run_stable_code())
```

## File Inputs

#### ResNet image classification

You can pass file inputs to models using file paths:

```python
import pathlib

import flymyai

response = flymyai.run(
    apikey="fly-secret-key",
    model="flymyai/resnet",
    payload={"image": pathlib.Path("/path/to/image.png")}
)
print(response.output_data["495"])
```

## File Response Handling

Files received from the neural network are always encoded in base64 format. To process these files, you need to decode them first. Here's an example of how to handle an image file:

#### StableDiffusion Turbo image generation in ~50ms 🚀

```python
import base64
import flymyai

response = flymyai.run(
    apikey="fly-secret-key",
    model="flymyai/SDTurboFMAAceleratedH100",
    payload={
        "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic, photorealistic",
    }
)
base64_image = response.output_data["sample"][0]
image_data = base64.b64decode(base64_image)
with open("generated_image.jpg", "wb") as file:
    file.write(image_data)
```

## Asynchronous Requests

FlyMyAI supports asynchronous requests for improved performance. Here's how to use it:

```python
import asyncio
import flymyai


async def main():
    payloads = [
        {
            "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic, photorealistic",
            "negative_prompt": "Dark colors, gloomy atmosphere, horror",
            "seed": count,
            "denoising_steps": 4,
            "scheduler": "DPM++ SDE"
         }
        for count in range(1, 10)
    ]
    async with asyncio.TaskGroup() as gr:
        tasks = [
            gr.create_task(
                flymyai.async_run(
                    apikey="fly-secret-key",
                    model="flymyai/DreamShaperV2-1",
                    payload=payload
                )
            )
            for payload in payloads
        ]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result.output_data["output"])


asyncio.run(main())
```

## Running Models in the Background

To run a model in the background, simply use the async_run() method:

```python
import asyncio
import flymyai
import pathlib


async def background_task():
    payload = {"audio": pathlib.Path("/path/to/audio.mp3")}
    response = await flymyai.async_run(
        apikey="fly-secret-key",
        model="flymyai/whisper",
        payload=payload
    )
    print("Background task completed:", response.output_data["transcription"])


async def main():
    task = asyncio.create_task(background_task())
    await task

asyncio.run(main())
# Continue with other operations while the model runs in the background
```

## Asynchronous Prediction Tasks

For long-running operations, FlyMyAI provides asynchronous prediction tasks. This allows you to submit a task and check its status later, which is useful for handling time-consuming predictions without blocking your application.

### Using Synchronous Client

```python
from flymyai import client
from flymyai.core.exceptions import (
    RetryTimeoutExceededException,
    FlyMyAIExceptionGroup,
)

# Initialize client
fma_client = client(apikey="fly-secret-key")

# Submit async prediction task
prediction_task = fma_client.predict_async_task(
    model="flymyai/flux-schnell",
    payload={"prompt": "Funny Cat with Stupid Dog"}
)

try:
    # Get result
    result = prediction_task.result()

    print(f"Prediction completed: {result.inference_responses}")
except RetryTimeoutExceededException:
    print("Prediction is taking longer than expected")
except FlyMyAIExceptionGroup as e:
    print(f"Prediction failed: {e}")
```

### Using Asynchronous Client

```python
import asyncio
from flymyai import async_client
from flymyai.core.exceptions import (
    RetryTimeoutExceededException,
    FlyMyAIExceptionGroup,
)

async def run_prediction():
    # Initialize async client
    fma_client = async_client(apikey="fly-secret-key")

    # Submit async prediction task
    prediction_task = await fma_client.predict_async_task(
        model="flymyai/flux-schnell",
        payload={"prompt": "Funny Cat with Stupid Dog"}
)

    try:
        # Await result with default timeout
        result = await prediction_task.result()
        print(f"Prediction completed: {result.inference_responses}")

        # Check response status
        all_successful = all(
            resp.infer_details["status"] == 200
            for resp in result.inference_responses
        )
        print(f"All predictions successful: {all_successful}")

    except RetryTimeoutExceededException:
        print("Prediction is taking longer than expected")
    except FlyMyAIExceptionGroup as e:
        print(f"Prediction failed: {e}")

# Run async function
asyncio.run(run_prediction())
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

## Agent variables, freeze & re-run

Agents can accept runtime `{{ variables }}` and be "compiled" into a
reusable Markdown instruction that you re-run later with fresh values.

### Draft an `input_schema` from a prompt

```python
from flymyai import AgentClient

client = AgentClient(api_key="fly-secret-key")

suggestion = client.agents.suggest_schema(
    user_prompt="Summarize {{ url }} in {{ n_sentences }} sentences.",
    generate_descriptions=True,
)
print(suggestion.input_schema)
print(suggestion.input_description)
```

### Create an agent, run it with variables, then freeze + re-run

```python
agent = client.agents.create(
    name="Web summarizer",
    goal="Summarize {{ url }} in {{ n_sentences }} sentences.",
    input_schema=suggestion.input_schema,
    output_schema=suggestion.output_schema,
)

# 1) First run — with variables
run = client.agents.run(
    agent.id,
    variables={"url": "https://example.com", "n_sentences": 3},
)
run = client.runs.wait(run.id)
print(run.output)

# 2) Freeze this run into a reusable instruction (Compile)
compilation = client.agents.compile_from_run(run.id)
print(compilation.instruction_md)

# 3) Re-run later with different variables — no chat history carried over
later = client.compilations.run_instruction_and_wait(
    compilation.id,
    variables={"url": "https://news.ycombinator.com", "n_sentences": 5},
)
print(later.output)
```

### Handling invalid variables

When `variables` don't match the agent's `input_schema` the server replies
`HTTP 400` and the client raises `VariablesValidationError`:

```python
from flymyai import VariablesValidationError

try:
    client.agents.run(agent.id, variables={})
except VariablesValidationError as err:
    print(err.messages)       # ["'url' is a required property", ...]
    print(err.field_errors)   # {"url": "'url' is a required property"}
```

### Drafting schemas from an existing run

If you already have a completed run, you can ask the server to infer
schemas from the actual chat history and tool trace:

```python
# NOTE: this also persists the resulting schemas onto the source agent.
suggestion = client.runs.suggest_schema(
    run.id,
    inputs_prompt="A URL and a sentence count",
    outputs_prompt="A short summary",
)
```
