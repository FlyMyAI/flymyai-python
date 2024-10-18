import asyncio
import threading
import time
from typing import Callable, Awaitable, Type, Optional

import httpx

from flymyai.core.exceptions import RetryTimeoutExceededException


def retryable_callback(
    cb: Callable,
    retries: Optional[int],
    append_on_exception_cls: Type[Exception],
    exception_group_cls: Type[Exception],
    timeout_seconds: Optional[float] = None,
    await_treshold: Optional[float] = None,
):
    """
    Decorator to retry a function
    """

    should_stop = False
    result_container = None
    exc_container = None

    def wrapper():
        nonlocal should_stop, result_container, exc_container
        retries_history = []
        r = 0
        while r != retries:
            try:
                res = cb()
                result_container = retries_history, res
                return
            except append_on_exception_cls as e:
                retries_history.append(e)
                if e.requires_retry and not should_stop:
                    time.sleep(await_treshold or 0)
                    r += 1
                    continue
                else:
                    exc_container = exception_group_cls(retries_history)
                    return
            except exception_group_cls as e1:
                exc_container = e1
                return
            except Exception as e2:
                exc_container = e2
                return
        exception_gr = exception_group_cls(retries_history)
        exc_container = exception_gr

    waiting_thread = threading.Thread(target=wrapper)
    waiting_thread.start()
    if timeout_seconds is not None:
        timeout_seconds += 0.01
    waiting_thread.join(timeout=timeout_seconds)
    should_stop = True
    if not result_container and not exc_container:
        raise RetryTimeoutExceededException()
    if exc_container:
        raise exc_container
    return result_container


async def aretryable_callback(
    cb: Callable[..., Awaitable[httpx.Response]],
    retries,
    append_on_exception_cls: Type[Exception],
    exception_group_cls: Type[Exception],
    timeout_seconds: Optional[float] = None,
    await_treshold: Optional[float] = None,
):
    """
    Decorator to retry a function
    """
    if timeout_seconds is not None:
        timeout_seconds += 0.01

    async def wrapper():
        retries_history = []
        r = 0
        while r != retries:
            try:
                res = await cb()
                return retries_history, res
            except append_on_exception_cls as e1:
                retries_history.append(e1)
                if e1.requires_retry:
                    await asyncio.sleep(await_treshold or 0)
                    r += 1
                    continue
                else:
                    raise exception_group_cls(retries_history)
            except Exception as e2:
                raise e2
        else:
            exception_gr = exception_group_cls(retries_history)
            raise exception_gr

    try:
        return await asyncio.wait_for(wrapper(), timeout_seconds)
    except asyncio.TimeoutError as e:
        raise RetryTimeoutExceededException() from e
