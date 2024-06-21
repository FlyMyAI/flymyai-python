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
print(response.output_data["o_logits"][0])
```
or 
```python
import flymyai

response = flymyai.run(
    auth={
        "apikey": "fly-secret-key",
        "username": "flymyai",
        "project_name": "llama-v3-8b",
    },
    payload=
    {
        "i_prompt": f"You discover the last library in the world, hidden in a forgotten city. What secrets does it hold, and why is it protected by ancient guardians?"
    }
)
print(response.output_data["o_output"][0])
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

