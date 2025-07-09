# FlyMy.AI

<p align="center">
  <img src="https://telegra.ph/file/d76588fc58b3445be4291.png" alt="Generated with FlyMy.AI in 🚀 70ms" width="500" />
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

## Basic Usage

Here's a simple example of how to use the FlyMyAI client:

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
