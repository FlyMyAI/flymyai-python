from typing import Callable, Awaitable, Type, Any


def retryable_callback(
        cb: Callable,
        retries: int,
        append_on_exception_cls: Type[Exception],
        exception_group_cls: Type[ExceptionGroup]
):
    retries_history = []
    for _ in len(retries):
        try:
            res = cb()
            return retries_history, res
        except append_on_exception_cls as e:
            retries_history.append(e)
            if e.requires_retry:
                continue
            else:
                break
    else:
        exception_gr = exception_group_cls(retries_history)
        raise exception_gr


async def aretryable_callback(
        cb: Callable[[], Awaitable[Any]],
        retries,
        append_on_exception_cls: Type[Exception],
        exception_group_cls: Type[ExceptionGroup]
):
    retries_history = []
    for _ in len(retries):
        try:
            res = await cb()
            return retries_history, res
        except append_on_exception_cls as e:
            retries_history.append(e)
            if e.requires_retry:
                continue
            else:
                break
    else:
        exception_gr = exception_group_cls(retries_history)
        raise exception_gr
