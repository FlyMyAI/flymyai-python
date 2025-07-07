import os
import time
from typing import Union, Optional
from pathlib import Path

import httpx

from flymyai.core._response import FlyMyAIM1Response
from flymyai.core.types.m1 import M1GenerationTask, M1Record, M1Role
from flymyai.core.clients.base_m1_client import BaseM1Client, _predict_timeout


class BaseM1SyncClient(BaseM1Client[httpx.Client]):
    """Synchronous client for interacting with FlyMyAI M1 chat generation models.
    Handles image uploads, chat history tracking, and result polling.
    """

    def _construct_client(self):
        return httpx.Client(
            http2=True,
            headers=self._headers,
            base_url=os.getenv("FLYMYAI_M1_DSN", "https://api.chat.flymy.ai/"),
            timeout=_predict_timeout,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()

    def generate(self, prompt: str, image: Optional[Union[str, Path]] = None):
        """Submit a chat prompt with optional image input and return the final generation result.

        :param prompt: User input string to send to the model.
        :param image: Local image file (as `Path`) or remote image URL (as `str`).
        :return: FlyMyAIM1Response with generated content and metadata.
        """
        self._process_image(image)
        self._m1_history.add(M1Record(role=M1Role.user, content=prompt))
        generation_task = self.generation_task()
        result = self.generation_task_result(generation_task)
        return result

    def _process_image(self, image: Optional[Union[str, Path]]) -> Optional[str]:
        if image is None:
            return
        image_url = None

        if isinstance(image, Path):
            image_url = self.upload_image(image)
        elif isinstance(image, str):
            image_url = image

        self._image = image_url
        return image_url

    def generation_task(self) -> M1GenerationTask:
        payload = {
            "chat_history": self._m1_history.serialize(),
            "image_url": self._image,
        }
        response = self._client.post(
            self._generation_path, json=payload, headers=self._headers
        )
        response.raise_for_status()
        response_data = response.json()
        return M1GenerationTask(request_id=response_data["request_id"])

    def generation_task_result(
        self, generation_task: M1GenerationTask
    ) -> FlyMyAIM1Response:
        while True:
            response = self._client.get(self._populate_result_path(generation_task))
            response.raise_for_status()
            response_data = response.json()

            if response_data.get("success"):
                self._m1_history.add(
                    M1Record(
                        role=M1Role.assistant,
                        content=response_data.get("data", {}).get("text", ""),
                    )
                )
                if file_url := response_data.get("data", {}).get("file_url", ""):
                    if not file_url.endswith(".mp4"):
                        self._image = (
                            os.getenv("FLYMYAI_M1_DSN", "https://api.chat.flymy.ai/")
                            + file_url
                        )
                return FlyMyAIM1Response.from_httpx(response)

            if response_data.get("error") == "Still processing":
                time.sleep(1)
                continue

            raise RuntimeError(
                f"Generation failed with status {response_data.get('status')}: {response_data.get('error')}"
            )

    def upload_image(self, image: Union[str, Path]) -> str:
        """Upload a local image file and receive a hosted URL.

        :param image: Local file path (as `str` or `Path`).
        :return: Hosted image URL returned by the server.
        """
        image_path = Path(image) if isinstance(image, str) else image
        with image_path.open("rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            response = self._client.post(self._image_upload_path, files=files)
        response.raise_for_status()
        response_data = response.json()
        return (
            os.getenv("FLYMYAI_M1_DSN", "https://api.chat.flymy.ai/")
            + response_data["url"]
        )
