import asyncio
import os
import threading
import time
from typing import Callable, Awaitable, Type, Optional

import httpx

from flymyai.core._retry import exception_retry_delay, exception_should_retry
from flymyai.core.exceptions import RetryTimeoutExceededException

# While polling (retries=None), errors other than "not ready yet" are asked again at
# most this many times in a row; "not ready yet" (425) is asked until the timeout.
_POLL_ERROR_RETRIES = int(os.getenv("FLYMYAI_MAX_RETRIES", 2))


def _is_not_ready(exc: BaseException) -> bool:
    return getattr(getattr(exc, "response", None), "status_code", None) == 425


def _next_wait(
    exc: BaseException,
    retries: Optional[int],
    attempt: int,
    errors_in_row: int,
    await_treshold: Optional[float],
):
    """(seconds to wait, errors in a row) before the next attempt, or None to stop."""
    if retries is None and not _is_not_ready(exc):
        errors_in_row += 1
        if errors_in_row > _POLL_ERROR_RETRIES:
            return None
        return exception_retry_delay(errors_in_row - 1, exc), errors_in_row
    if await_treshold is not None:
        return await_treshold, 0
    return exception_retry_delay(attempt, exc), errors_in_row


def retryable_callback(
    cb: Callable,
    retries: Optional[int],
    append_on_exception_cls: Type[Exception],
    exception_group_cls: Type[Exception],
    timeout_seconds: Optional[float] = None,
    await_treshold: Optional[float] = None,
):
    """
    Call ``cb`` until it returns: at most ``retries`` attempts, and only while its
    error allows another one (flymyai.core._retry: the server's ``x-should-retry``
    or JSON body first, otherwise the status). Between attempts it waits the
    server's ``retry-after``, else exponential backoff from 0.5 s to 8 s with jitter;
    ``await_treshold`` is the polling interval for "not ready yet" (425).
    ``retries=None`` polls until ``timeout_seconds``.
    """

    should_stop = False
    result_container = None
    exc_container = None

    def wrapper():
        nonlocal should_stop, result_container, exc_container
        retries_history = []
        r = 0
        errors_in_row = 0
        while r != retries:
            try:
                res = cb()
                result_container = retries_history, res
                return
            except append_on_exception_cls as e:
                retries_history.append(e)
                wait = None
                if not should_stop and exception_should_retry(e):
                    wait = _next_wait(e, retries, r, errors_in_row, await_treshold)
                if wait is None:
                    exc_container = exception_group_cls(retries_history)
                    return
                delay, errors_in_row = wait
                time.sleep(delay)
                r += 1
                continue
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
    The asyncio twin of ``retryable_callback``: the same attempts, decisions and waits.
    """
    if timeout_seconds is not None:
        timeout_seconds += 0.01

    async def wrapper():
        retries_history = []
        r = 0
        errors_in_row = 0
        while r != retries:
            try:
                res = await cb()
                return retries_history, res
            except append_on_exception_cls as e1:
                retries_history.append(e1)
                wait = None
                if exception_should_retry(e1):
                    wait = _next_wait(e1, retries, r, errors_in_row, await_treshold)
                if wait is None:
                    raise exception_group_cls(retries_history)
                delay, errors_in_row = wait
                await asyncio.sleep(delay)
                r += 1
                continue
            except Exception as e2:
                raise e2
        else:
            exception_gr = exception_group_cls(retries_history)
            raise exception_gr

    try:
        return await asyncio.wait_for(wrapper(), timeout_seconds)
    except asyncio.TimeoutError as e:
        raise RetryTimeoutExceededException() from e
