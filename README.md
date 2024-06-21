# FlyMyAI Python Client

This is a Python client for [FlyMyAI](https://flymy.ai) - the fastest inference service available. It allows you to easily run models and get predictions from your Python code.

## Requirements

- Python 3.10+

## Installation

Install the FlyMyAI client using pip:

```sh
pip install flymyai-client
```

## Authentication
Before using the client, you need to have your API key, username, and project name. In order to get credentials, you have to sign up on flymy.ai and get your personal data on [the profile](https://app.flymy.ai/profile).

## Basic Usage
Here's a simple example of how to use the FlyMyAI client:

```python
import flymyai

response = flymyai.run(
    auth={
        "apikey": "fly-secret-key",
        "username": "flymyai",
        "project_name": "bert",
    },
    payload={"i_text": "What a fabulous fancy building! It looks like a palace!"}
)
print(response.output_data["o_logits"])
```

## File Inputs
You can pass file inputs to models using file paths:

```python
import flymyai
import pathlib

response = flymyai.run(
    auth={
        "apikey": "fly-secret-key",
        "username": "flymyai",
        "project_name": "resnet",
    },
    payload={"i_image": pathlib.Path("/path/to/image.png")}
)
print(response.output_data["o_495"])
```


## File Response Handling
Files received from the neural network are always encoded in base64 format. To process these files, you need to decode them first. Here's an example of how to handle an image file:

```python
import base64
import flymyai

response = flymyai.run(
    auth={
        "apikey": "fly-secret-key",
        "username": "flymyai",
        "project_name": "StableDiffusionXL",
    },
    payload={
        "i_prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic, photorealistic",
        "i_negative_prompt": "Dark colors, gloomy atmosphere, horror",
        "i_seed": 42,
        "i_denoising_steps": 25
    }
)
base64_image = response.output_data["o_sample"][0]
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
    auth = {
            "apikey": "fly-secret-key",
            "username": "flymyai",
            "project_name": "llama-v3-8b",
    }
    prompts = [
        {"i_prompt": f"Some random stuff number {count}"}
        for count in range(1, 10)
    ]
    async with asyncio.TaskGroup() as gr:
        tasks = [
            gr.create_task(flymyai.async_run(auth, payload=prompt))
            for prompt in prompts
        ]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result.output_data["o_output"])
        
asyncio.run(main())
```

## Running Models in the Background
To run a model in the background, simply use the async_run() method:

```python
import asyncio
import flymyai
import pathlib

async def background_task():
    auth = {
        "apikey": "fly-secret-key",
        "username": "flymyai",
        "project_name": "whisper"
    }
    payload = {"i_audio": pathlib.Path("/path/to/audio.mp3")}
    
    response = await flymyai.async_run(auth, payload=payload)
    print("Background task completed:", response.output_data)

asyncio.create_task(background_task())
# Continue with other operations while the model runs in the background
```

## Error Handling
The FlyMyAI client includes error information in the exc_history field of the response. Always check this field to ensure your request was processed successfully:

```python
response = flymyai.run(auth, payload=payload)
if response.exc_history:
    print("An error occurred:", response.exc_history)
else:
    print("Success:", response.output_data)
```

