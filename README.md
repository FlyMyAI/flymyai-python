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

#### llama 3 8b
```python
from flymyai import client, FlyMyAIPredictException

fma_client = client(apikey="fly-secret-key")

try:
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
            model="flymyai/llama3"
        )
    for response in stream_iterator:
        print(response.output_data["output"].pop(), end="")
except FlyMyAIPredictException as e:
    print(e)
    raise e
finally:
    print()
```

## Async Streams
For llms you should use stream method

#### Stable Code Instruct 3b
```python
from flymyai import async_client, FlyMyAIPredictException
import asyncio

async def run_stable_code():
    fma_client = async_client(apikey="fly-secret-key")
    try:
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
        async for response in stream_iterator:
            print(response.output_data["output"].pop(), end="")
    except FlyMyAIPredictException as e:
        print(e)
        raise e
    finally:
        print()


asyncio.run(run_stable_code())
```



## File Inputs
#### ResNet image classification
You can pass file inputs to models using file paths:

```python
import flymyai
import pathlib

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

