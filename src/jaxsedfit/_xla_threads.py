"""Keep XLA's CPU thread pool large enough for the NUTS gradient callback.

With ``InferenceConfig.nuts_gradient_callback`` NUTS gets the potential's value and
gradient from a ``jax.pure_callback`` that runs a separately compiled JAX function
(``core._callback_potential``). XLA:CPU may run the sampler's loop, and therefore the
callback, on a thread of its intra-op thread pool. The nested computation then hands
part of its work to that same pool and waits for it. If the pool has a single thread,
that thread is the one waiting, so the work never runs and the process deadlocks
(the hang of one-CPU batch jobs). Any second pool thread runs the work.

XLA sizes the pool once, when the JAX backend starts, from the first of the
``PJRT_NPROC`` and ``NPROC`` environment variables that is set, else from the number of
CPUs the process may run on, and never below the number of CPU devices. A pool of two
threads costs nothing measurable on one CPU: the threads share it. So, on import, if
the pool would get fewer than two threads and the backend has not started yet,
:func:`reserve_callback_threads` sets ``PJRT_NPROC=2``. :func:`callback_is_safe`
tells the NUTS setup whether the callback can be used in this process.
"""

from __future__ import annotations

import os
import re

#: Pool threads needed by one sampler that blocks a pool thread inside the callback.
MIN_CALLBACK_THREADS = 2

# Pool size that XLA got (or will get) in this process; None if unknown.
_pool_threads: int | None = None


def _env_threads() -> int | None:
    """Thread count from ``PJRT_NPROC`` / ``NPROC``, read as XLA reads them."""
    for name in ("PJRT_NPROC", "NPROC"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return max(0, int(value.strip()))
        except ValueError:
            continue  # XLA also ignores a value that is not an integer
    return None


def _schedulable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _forced_device_count() -> int:
    match = re.search(r"--xla_force_host_platform_device_count=(\d+)", os.environ.get("XLA_FLAGS", ""))
    return int(match.group(1)) if match else 1


def expected_pool_threads() -> int:
    """Size XLA:CPU gives its thread pools if the backend starts now."""
    threads = _env_threads()
    if threads is None:
        threads = _schedulable_cpus()
    return max(threads, _forced_device_count())


def _backend_started() -> bool:
    try:
        from jax._src import xla_bridge

        return bool(xla_bridge.backends_are_initialized())
    except Exception:  # pragma: no cover - private JAX API changed
        return False


def reserve_callback_threads() -> None:
    """Before the JAX backend starts, make sure its CPU pool gets two or more threads."""
    global _pool_threads
    if _backend_started():
        # Too late to change the pool; assume the environment it started with.
        _pool_threads = expected_pool_threads()
        return
    if expected_pool_threads() < MIN_CALLBACK_THREADS:
        os.environ["PJRT_NPROC"] = str(MIN_CALLBACK_THREADS)
    _pool_threads = None  # decided when the backend starts; see callback_is_safe


def callback_is_safe(concurrent_callbacks: int = 1) -> bool:
    """Whether the CPU pool has a thread left while ``concurrent_callbacks`` wait."""
    if _pool_threads is not None:
        threads = _pool_threads
    else:
        threads = expected_pool_threads()
    return threads >= concurrent_callbacks + 1
