"""NUTS with the gradient callback must not deadlock when XLA would get one CPU thread."""

import os
import subprocess
import sys
import textwrap

import pytest

from jaxsedfit import _xla_threads

# A model whose NUTS step and whose gradient each have more independent operations than
# XLA's thunk executor runs on one thread (8), so XLA splits both across its CPU pool:
# the sampler loop, and with it the callback, then runs on a pool thread, and the
# gradient the callback starts queues work for the same pool. With a one-thread pool
# this deadlocked on the first NUTS step.
_SCRIPT = textwrap.dedent(
    """
    import jaxsedfit  # first, before the JAX backend starts, as in a normal fit
    import jax, jax.numpy as jnp, numpy as np
    import numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_uniform
    from jaxsedfit.core import _use_callback_gradient

    n, k = 64, 8
    rng = np.random.default_rng(0)
    mats = [jnp.asarray(rng.normal(size=(n, n)) / np.sqrt(n)) for _ in range(k)]
    obs = [jnp.asarray(rng.normal(size=n)) for _ in range(k)]

    def model():
        for i in range(k):
            x = numpyro.sample(f"x{i}", dist.Normal(0, 1).expand([n]))
            numpyro.sample(f"y{i}", dist.Normal(jnp.tanh(mats[i] @ x), 1.0), obs=obs[i])

    key = jax.random.PRNGKey(1)
    kernel = NUTS(model, init_strategy=init_to_uniform(), max_tree_depth=4,
                  dense_mass=[(f"x{i}",) for i in range(k)])
    _use_callback_gradient(kernel, key, init_to_uniform())
    mcmc = MCMC(kernel, num_warmup=20, num_samples=10, progress_bar=False)
    mcmc.run(key)
    print("DONE", float(mcmc.get_samples()["x0"].mean()))
    """
)


@pytest.mark.parametrize("cpu_env", [{"NPROC": "1"}, {"PJRT_NPROC": "1"}])
def test_callback_nuts_completes_with_one_xla_thread(cpu_env):
    env = {k: v for k, v in os.environ.items() if k not in {"NPROC", "PJRT_NPROC", "XLA_FLAGS"}}
    env.update(cpu_env)
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "DONE" in result.stdout


def test_pool_size_follows_xla_rules(monkeypatch):
    for name in ("PJRT_NPROC", "NPROC", "XLA_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_xla_threads, "_schedulable_cpus", lambda: 1)
    assert _xla_threads.expected_pool_threads() == 1
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    assert _xla_threads.expected_pool_threads() == 4
    monkeypatch.setenv("NPROC", "8")
    assert _xla_threads.expected_pool_threads() == 8
    monkeypatch.setenv("PJRT_NPROC", "1")  # read before NPROC, as XLA does
    monkeypatch.delenv("XLA_FLAGS")
    assert _xla_threads.expected_pool_threads() == 1


def test_reserve_threads_only_before_backend_starts(monkeypatch):
    for name in ("PJRT_NPROC", "NPROC", "XLA_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_xla_threads, "_schedulable_cpus", lambda: 1)
    monkeypatch.setattr(_xla_threads, "_pool_threads", None)

    monkeypatch.setattr(_xla_threads, "_backend_started", lambda: False)
    _xla_threads.reserve_callback_threads()
    assert os.environ["PJRT_NPROC"] == "2"
    assert _xla_threads.callback_is_safe()
    assert not _xla_threads.callback_is_safe(concurrent_callbacks=2)

    monkeypatch.delenv("PJRT_NPROC")
    monkeypatch.setattr(_xla_threads, "_backend_started", lambda: True)
    _xla_threads.reserve_callback_threads()
    assert "PJRT_NPROC" not in os.environ
    assert not _xla_threads.callback_is_safe()  # the NUTS setup then compiles the gradient in-loop
