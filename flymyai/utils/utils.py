from typing import Callable, Awaitable, Type, Any, Coroutine

import httpx


def retryable_callback(
    cb: Callable,
    retries: int,
    append_on_exception_cls: Type[Exception],
    exception_group_cls: Type[Exception],
):
    """
    Decorator to retry a function
    """
    retries_history = []
    for _ in range(retries):
        try:
            res = cb()
            return retries_history, res
        except append_on_exception_cls as e:
            retries_history.append(e)
            if e.requires_retry:
                continue
            else:
                raise exception_group_cls(retries_history)
    else:
        exception_gr = exception_group_cls(retries_history)
        raise exception_gr


async def aretryable_callback(
    cb: Callable[..., Awaitable[httpx.Response]],
    retries,
    append_on_exception_cls: Type[Exception],
    exception_group_cls: Type[Exception],
):
    """
    Decorator to retry a function
    """
    retries_history = []
    for _ in range(retries):
        try:
            res = await cb()
            return retries_history, res
        except append_on_exception_cls as e:
            retries_history.append(e)
            if e.requires_retry:
                continue
            else:
                raise exception_group_cls(retries_history)
    else:
        exception_gr = exception_group_cls(retries_history)
        raise exception_gr
