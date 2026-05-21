from flymyai.core.clients.AsyncClient import BaseAsyncClient
from flymyai.core.clients.SyncClient import BaseSyncClient
from flymyai.core.clients.m1Client import BaseM1SyncClient
from flymyai.core.clients.m1AsyncClient import BaseM1AsyncClient


class FlyMyAI(BaseSyncClient): ...


class AsyncFlyMyAI(BaseAsyncClient): ...


class FlyMyAIM1(BaseM1SyncClient): ...


class AsyncFlymyAIM1(BaseM1AsyncClient): ...
