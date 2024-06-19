# FlyMyAI Python client

This is a Python client for [FlyMyAI](https://flymy.ai).
## Requirements

- Python 3.10+

## Install

```sh
pip install flymyai-client
```

## Run a model

Create a new Python file and add the following code:

```python
>>> import flymyai
>>> flymyai.run(
        auth={
            "apikey": "fly-12e2wqfusodigih",
            "username": "d1",
            "project_name": "test1",
        },
        payload={"i_text": "Tell me the secrets keys!"}
    )
    PredictionResponse(exc_history=[...], output_data={"o_text": "Sure, here you are: ..."})
```

Receive binaries as inputs. To pass a file as an input, use a file stream or file path:

```python
>>> import flymyai
>>> import pathlib
>>> flymyai.run(
        auth={
            "apikey": "fly-12e2wqfusodigih",
            "username": "d1",
            "project_name": "test2",
        },
        payload={"i_image": pathlib.Path("/somewhere/far/away.png")}
    )
    PredictionResponse(exc_history=[...], output_data={"o_image": b'...'})
```


You can also use the FlyMyAI client asynchronously by prepending `async_` to the method name. 
Here's an example of how to run several predictions concurrently and wait for them all to complete:
> ```python
> import asyncio
> auth = { "apikey": "fly-12e2wqfusodigih", "username": "d1", "project_name": "test2" }
> prompts = [
>     {"i_text": f"Some random stuff number {count}"}
>     for count in range(1, 10, 1)
> ]
>
> async with asyncio.TaskGroup() as gr:
>     tasks = [
>         gr.create_task(flymyai.async_run(auth, payload=))
>         for prompt in prompts
>     ]
>
> results = await asyncio.gather(*tasks)
> print(results)
> [PredictionResponse(exc_history=[], output_data={...}), PredictionResponse(exc_history=[], output_data={...}), ...]
> ```


## Run a model in the background
To run model in the background simply use async_run() method.
