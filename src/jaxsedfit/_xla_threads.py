"""Keep XLA's CPU thread pool large enough for the NUTS gradient callback.

With ``InferenceConfig.nuts_gradient_callback`` NUTS gets the potential's value and
gradient from a ``jax.pure_callback`` that runs a separately compiled JAX function
(``core._callback_potential``). XLA:CPU may run the sampler's loop, and therefore the
callback, on a thread of its intra-op thread pool. The nested computation then hands
part of its work to that same pool and waits for it. If every pool thread is waiting
inside a callback, that work never runs and the process deadlocks (the hang of one-CPU
batch jobs). So the pool needs one more thread than the callbacks that can run at once:
two for one chain, ``n + 1`` for ``n`` chains run in parallel on ``n`` CPU devices.

XLA sizes the pool once, when the JAX backend starts: from the first of the
``PJRT_NPROC`` and ``NPROC`` environment variables that holds an integer, else from the
number of CPUs the process may run on, and never below the number of CPU devices. Extra
threads cost nothing measurable on a small CPU allocation: they share it.

On import, :func:`reserve_callback_threads` therefore sets ``PJRT_NPROC`` to
``max(2, devices + 1)`` if the pool would be smaller, the backend has not started, and
``PJRT_NPROC`` is not set already (an explicit value is respected). The CPU device count
(``--xla_force_host_platform_device_count`` in ``XLA_FLAGS`` or ``jax_num_cpu_devices``)
must be set before jaxsedfit is imported for its parallel chains to be counted. The
variable lands in ``os.environ``, so subprocesses started afterwards inherit it.
:func:`callback_is_safe` tells the NUTS setup whether the callback can be used.
"""

from __future__ import annotations

import os
import re

#: Pool threads needed by one sampler that blocks a pool thread inside the callback.
MIN_CALLBACK_THREADS = 2

# Pool size of the running backend, recorded once it has started; None until then.
_pool_threads: int | None = None


def _int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None  # XLA also ignores a value that is not an integer


def _env_threads() -> int | None:
    """Thread count from ``PJRT_NPROC`` / ``NPROC``, read as XLA reads them."""
    for name in ("PJRT_NPROC", "NPROC"):
        value = _int_env(name)
        if value is not None:
            return max(0, value)
    return None


def _schedulable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _jax_num_cpu_devices() -> int:
    """``jax_num_cpu_devices`` (-1 if unset), which JAX passes to XLA ahead of XLA_FLAGS."""
    try:
        import jax

        return int(jax.config.jax_num_cpu_devices)
    except Exception:  # pragma: no cover - option renamed or removed
        value = _int_env("JAX_NUM_CPU_DEVICES")
        return -1 if value is None else value


def _forced_device_count() -> int:
    num = _jax_num_cpu_devices()
    if num >= 0:
        return max(1, num)
    # XLA parses the flags in order, so the last occurrence wins.
    matches = re.findall(r"--xla_force_host_platform_device_count=(\d+)", os.environ.get("XLA_FLAGS", ""))
    return max(1, int(matches[-1])) if matches else 1


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
        return True  # unknown: assume too late, so an unsafe pool falls back instead of hanging


def reserve_callback_threads() -> None:
    """Before the JAX backend starts, give its CPU pool a thread per parallel chain plus one."""
    global _pool_threads
    if _backend_started():
        # Too late to change the pool; record the environment it started with.
        _pool_threads = expected_pool_threads()
        return
    _pool_threads = None
    if _int_env("PJRT_NPROC") is not None:
        return  # an explicit choice; callback_is_safe judges it
    needed = max(MIN_CALLBACK_THREADS, _forced_device_count() + 1)
    if expected_pool_threads() < needed:
        os.environ["PJRT_NPROC"] = str(needed)


def callback_is_safe(concurrent_callbacks: int = 1) -> bool:
    """Whether the CPU pool keeps a thread free while ``concurrent_callbacks`` wait."""
    global _pool_threads
    if _pool_threads is None and _backend_started():
        _pool_threads = expected_pool_threads()  # the backend has its pool now; fix the value
    threads = _pool_threads if _pool_threads is not None else expected_pool_threads()
    return threads >= concurrent_callbacks + 1
