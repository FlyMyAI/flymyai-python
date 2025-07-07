import os
from pathlib import Path
from typing import overload, Union, TypeVar, Generic, Optional

import httpx

from flymyai.core._response import FlyMyAIM1Response
from flymyai.core.types.m1 import M1GenerationTask
from flymyai.core.models.m1_history import M1History

DEFAULT_RETRY_COUNT = os.getenv("FLYMYAI_MAX_RETRIES", 2)

_PossibleClients = TypeVar(
    "_PossibleClients", bound=Union[httpx.Client, httpx.AsyncClient]
)


_predict_timeout = httpx.Timeout(
    connect=int(os.getenv("FMA_CONNECT_TIMEOUT", 999999)),
    read=int(os.getenv("FMA_READ_TIMEOUT", 999999)),
    write=int(os.getenv("FMA_WRITE_TIMEOUT", 999999)),
    pool=int(os.getenv("FMA_POOL_TIMEOUT", 999999)),
)


class BaseM1Client(Generic[_PossibleClients]):
    client: _PossibleClients
    _m1_history: M1History
    _image: Optional[str]

    def __init__(self, apikey: str):
        self._apikey = apikey
        self._client = self._construct_client()
        self._m1_history = M1History()
        self._image = None

    def reset_history(self):
        self._m1_history = M1History()

    @overload
    def generate(
        self, prompt: str, image: Optional[Union[str, Path]] = None
    ) -> FlyMyAIM1Response: ...

    @overload
    def generation_task(
        self, prompt: str, image: Optional[Union[str, Path]] = None
    ) -> M1GenerationTask: ...

    @overload
    def generation_task_result(
        self, generation_task: M1GenerationTask
    ) -> FlyMyAIM1Response: ...

    @overload
    def upload_image(self, image: Union[str, Path]): ...

    @overload
    async def generate(
        self, prompt: str, image: Optional[Union[str, Path]] = None
    ) -> FlyMyAIM1Response: ...

    @overload
    async def generation_task(
        self, prompt: str, image: Optional[Union[str, Path]] = None
    ) -> M1GenerationTask: ...

    @overload
    async def generation_task_result(
        self, generation_task: M1GenerationTask
    ) -> FlyMyAIM1Response: ...

    @overload
    async def upload_image(self, image: Union[str, Path]): ...

    @property
    def _headers(self):
        return {"X-API-KEY": self._apikey}

    @property
    def _generation_path(self):
        return "/chat"

    @property
    def _result_path(self):
        return "/chat-result/"

    def _populate_result_path(self, generation_task: M1GenerationTask):
        return "".join([self._result_path, generation_task.request_id])

    @property
    def _image_upload_path(self):
        return "/upload-image"
