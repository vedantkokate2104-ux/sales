"""
process_isolation.py
---------------------
Runs a callable in a separate OS process and reports its outcome back as
plain data — (True, result) or (False, error_message) — instead of ever
letting the callable's failure reach the caller directly.

Why this exists: a Python try/except can only catch things that raise a
Python exception. It CANNOT catch a hard native crash — a segmentation
fault, an abort, an illegal instruction, a fatal error inside a compiled
C/C++ extension — because those terminate the OS process itself; there is
no Python stack left to unwind, and no `except` clause ever runs. This is
exactly the failure mode observed with Prophet 1.1.5 + cmdstanpy on some
Python 3.12 environments: `Prophet(...)` construction can raise
`AttributeError: 'Prophet' object has no attribute 'stan_backend'` (Stan
backend initialisation partially failing) immediately followed by a
segmentation fault in the native Stan/cmdstanpy code, which would
otherwise take the entire Flask process down with it — no amount of
try/except inside forecast/models/prophet.py can prevent that on its own.

The fix is architectural, not defensive-coding: move the risky call into
its own child process. If that child process dies — for any reason,
caught exception or hard crash — only the child dies. The parent (this
module's caller, ultimately the Flask process) observes an ordinary,
catchable "the operation failed" result and continues running.

Uses multiprocessing's "spawn" start method explicitly, not the POSIX
default "fork". fork() duplicates the parent's entire process image,
including any state a native library like cmdstanpy may hold in
partially-initialised or thread-local form — and the caller here
(forecast.trainer's ThreadPoolExecutor) is itself multi-threaded, which is
a well-known source of additional instability when combined with fork().
spawn() instead starts the child from a fresh interpreter that only
re-imports what the target callable needs, with none of the parent's
process state carried over.

Resource cleanup (runtime stability audit): every `multiprocessing.Queue`
and `Process` object created here is now explicitly closed after use
(`Queue.close()`/`join_thread()`, `Process.close()`), not just left for
garbage collection. A `Queue` holds an OS-level pipe plus (on most
platforms, including macOS) a semaphore registered with
`multiprocessing.resource_tracker`; letting it be garbage-collected
instead of explicitly closed is exactly what produces
`UserWarning: resource_tracker: There appear to be leaked semaphore
objects` at interpreter shutdown — harmless in a short-lived script, but
this project runs one (or several, under a walk-forward validation loop)
isolated call per forecast request inside a long-running Flask process,
so leaked semaphores accumulate for the life of the server rather than
being cleaned up once at exit.
"""

import logging
import multiprocessing
import traceback
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_CTX = multiprocessing.get_context("spawn")

# How long a single isolated call is allowed to run before it is treated
# as hung and killed. Generous — Prophet's Stan backend can legitimately
# take several seconds per fit, and a spawned interpreter also pays a
# one-time startup+import cost on top of that — but bounded, so a truly
# stuck child (e.g. Stan compilation hanging) can never block forecasting
# indefinitely.
DEFAULT_TIMEOUT_SECONDS = 90


def _isolated_worker(fn: Callable, args: tuple, kwargs: dict, result_queue) -> None:
    """
    Runs entirely inside the spawned child process. Any ORDINARY Python
    exception raised by `fn` is caught right here and reported back as
    data through the queue — only a crash severe enough to kill this
    process outright (segfault/abort/illegal instruction/OOM-kill) skips
    this except clause and is instead detected by the parent as an
    abnormal process exit (see run_isolated() below).
    """
    try:
        result_queue.put(("ok", fn(*args, **kwargs)))
    except Exception as exc:  # noqa: BLE001 — deliberately broad: this is the whole point
        result_queue.put((
            "error",
            "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        ))


def run_isolated(
    fn: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Tuple[bool, Any]:
    """
    Execute fn(*args, **kwargs) in an isolated child process.

    `fn` and every value in `args`/`kwargs` must be picklable (spawn
    re-imports the target module fresh in the child and sends the call
    arguments over a pipe) — plain functions defined at module level with
    plain data (pandas Series, dicts of numbers/strings, etc.) all satisfy
    this; bound methods and objects holding open file handles/native
    resources do not, so callers should pass a module-level function.

    Returns:
        (True, result)  — fn completed normally; `result` is its return value.
        (False, message) — fn raised an ordinary exception, the child
            process was killed by a signal or crashed natively (message
            explains which), or the call exceeded `timeout`.

    This function itself is defensive to its core: no failure mode of the
    child process, and no failure of the isolation machinery itself
    (starting the process, reading the queue), is allowed to raise out of
    here and reach the caller.
    """
    kwargs = kwargs or {}
    process: Optional[multiprocessing.process.BaseProcess] = None
    result_queue = None
    try:
        result_queue = _CTX.Queue()
        process = _CTX.Process(
            target=_isolated_worker, args=(fn, args, kwargs, result_queue), daemon=True,
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            return False, f"Timed out after {timeout}s and was terminated."

        if not result_queue.empty():
            status, payload = result_queue.get()
            if status == "ok":
                return True, payload
            return False, payload

        # The child exited without ever putting anything on the queue.
        # _isolated_worker() always puts exactly one ("ok", ...) or
        # ("error", ...) entry before returning normally — the only way
        # to reach this branch is for the child to have been killed by a
        # signal (segfault, abort, illegal instruction, OOM) before it
        # could do so. This is precisely the case a plain try/except in
        # the caller's own process could never have detected.
        exitcode = process.exitcode
        return False, (
            f"Process terminated unexpectedly (exit code {exitcode}) before returning a "
            f"result — likely a native crash inside the isolated call, not a Python exception."
        )

    except Exception as exc:
        # Defensive: even a failure in the isolation machinery itself
        # (e.g. the platform cannot spawn a new process) must never
        # propagate to the caller as an uncaught exception.
        logger.exception("Process isolation itself failed")
        return False, f"Process isolation failed: {exc}"

    finally:
        # Runtime stability audit — explicit resource cleanup. Every
        # branch above returns before this runs, so by this point the
        # queue has already been drained (if it had anything to give) and
        # the process has already been joined/terminated — this block's
        # only remaining job is to release the underlying OS resources
        # (pipes, semaphores, process handle) instead of leaving them for
        # garbage collection, which is what produced the reported
        # "resource_tracker: leaked semaphore objects" warning.
        if process is not None:
            if process.is_alive():
                try:
                    process.terminate()
                    process.join(5)
                except Exception:
                    pass
            try:
                process.close()
            except Exception:
                # close() raises if the process is still alive (shouldn't
                # happen after the join() above) or has already been
                # closed — either way, never let cleanup itself crash.
                pass
        if result_queue is not None:
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception:
                pass
