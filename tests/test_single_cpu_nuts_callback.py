"""NUTS with the gradient callback must not deadlock when XLA's CPU pool is small."""

import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from jaxsedfit import JAXSEDFit, _xla_threads

from test_diffstar_host import _mock_config

_ENV_NAMES = ("PJRT_NPROC", "NPROC", "XLA_FLAGS", "JAX_NUM_CPU_DEVICES")

# A model whose NUTS step and whose gradient each have more independent operations than
# XLA's thunk executor runs on one thread (8), so XLA splits both across its CPU pool:
# the sampler loop, and with it the callback, then runs on a pool thread, and the
# gradient the callback starts queues work for the same pool. With a one-thread pool
# this deadlocked on the first NUTS step.
_SCRIPT = textwrap.dedent(
    """
    import sys
    import jaxsedfit  # first, before the JAX backend starts, as in a normal fit
    import jax, jax.numpy as jnp, numpy as np
    import numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_uniform
    from jaxsedfit.core import _use_callback_gradient_if_safe

    num_chains = int(sys.argv[1])
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
    used = _use_callback_gradient_if_safe(kernel, key, init_to_uniform(), num_chains)
    print("CALLBACK", used, "DEVICES", jax.local_device_count(), flush=True)
    mcmc = MCMC(kernel, num_warmup=20, num_samples=10, num_chains=num_chains, progress_bar=False)
    mcmc.run(key)
    print("DONE", float(mcmc.get_samples()["x0"].mean()))
    """
)


@pytest.mark.parametrize(
    ("cpu_env", "num_chains", "expect_callback"),
    [
        ({"NPROC": "1"}, 1, True),  # one-CPU batch job: jaxsedfit sets PJRT_NPROC=2
        ({"PJRT_NPROC": "1"}, 1, False),  # explicit one-thread pool is respected: falls back
        ({"NPROC": "1", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}, 2, True),
        ({"PJRT_NPROC": "2", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}, 2, False),
    ],
)
def test_callback_nuts_completes_on_small_cpu_pools(cpu_env, num_chains, expect_callback):
    env = {k: v for k, v in os.environ.items() if k not in _ENV_NAMES}
    env.update(cpu_env)
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(num_chains)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "DONE" in result.stdout
    assert f"CALLBACK {expect_callback} DEVICES {num_chains}" in result.stdout
    assert ("thread pool is too small" in result.stderr) == (not expect_callback)


@pytest.fixture
def clean_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_xla_threads, "_schedulable_cpus", lambda: 1)
    monkeypatch.setattr(_xla_threads, "_jax_num_cpu_devices", lambda: -1)
    monkeypatch.setattr(_xla_threads, "_pool_threads", None)
    return monkeypatch


def test_pool_size_follows_xla_rules(clean_env):
    monkeypatch = clean_env
    assert _xla_threads.expected_pool_threads() == 1
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
    assert _xla_threads.expected_pool_threads() == 4
    # XLA keeps the last occurrence of a flag.
    monkeypatch.setenv(
        "XLA_FLAGS", "--xla_force_host_platform_device_count=8 --xla_force_host_platform_device_count=3"
    )
    assert _xla_threads.expected_pool_threads() == 3
    # jax_num_cpu_devices, when set, is what JAX passes to XLA.
    monkeypatch.setattr(_xla_threads, "_jax_num_cpu_devices", lambda: 5)
    assert _xla_threads.expected_pool_threads() == 5
    monkeypatch.setattr(_xla_threads, "_jax_num_cpu_devices", lambda: -1)
    monkeypatch.setenv("NPROC", "8")
    assert _xla_threads.expected_pool_threads() == 8
    monkeypatch.setenv("PJRT_NPROC", "1")  # read before NPROC, as XLA does
    monkeypatch.delenv("XLA_FLAGS")
    assert _xla_threads.expected_pool_threads() == 1
    monkeypatch.setenv("PJRT_NPROC", "many")  # not an integer: XLA moves on to NPROC
    assert _xla_threads.expected_pool_threads() == 8


def test_reserve_threads_sets_pool_for_devices(clean_env):
    monkeypatch = clean_env
    monkeypatch.setattr(_xla_threads, "_backend_started", lambda: False)
    _xla_threads.reserve_callback_threads()
    assert os.environ["PJRT_NPROC"] == "2"
    assert _xla_threads.callback_is_safe()
    assert not _xla_threads.callback_is_safe(concurrent_callbacks=2)

    monkeypatch.delenv("PJRT_NPROC")
    monkeypatch.setenv("NPROC", "1")
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=3")
    _xla_threads.reserve_callback_threads()
    assert os.environ["PJRT_NPROC"] == "4"
    assert _xla_threads.callback_is_safe(concurrent_callbacks=3)


def test_reserve_threads_respects_explicit_pjrt_nproc(clean_env):
    monkeypatch = clean_env
    monkeypatch.setattr(_xla_threads, "_backend_started", lambda: False)
    monkeypatch.setenv("PJRT_NPROC", "1")
    _xla_threads.reserve_callback_threads()
    assert os.environ["PJRT_NPROC"] == "1"
    assert not _xla_threads.callback_is_safe()  # NUTS then falls back with a warning


def test_reserve_threads_too_late_records_pool(clean_env):
    monkeypatch = clean_env
    monkeypatch.setattr(_xla_threads, "_backend_started", lambda: True)
    _xla_threads.reserve_callback_threads()
    assert "PJRT_NPROC" not in os.environ
    assert not _xla_threads.callback_is_safe()
    monkeypatch.setenv("PJRT_NPROC", "4")  # changing the environment later changes nothing
    assert not _xla_threads.callback_is_safe()


def test_pool_snapshot_on_first_check_after_backend_starts(clean_env):
    monkeypatch = clean_env
    started = {"value": False}
    monkeypatch.setattr(_xla_threads, "_backend_started", lambda: started["value"])
    _xla_threads.reserve_callback_threads()
    assert os.environ["PJRT_NPROC"] == "2"
    started["value"] = True
    assert _xla_threads.callback_is_safe()
    monkeypatch.setenv("PJRT_NPROC", "1")
    assert _xla_threads.callback_is_safe()  # the pool the backend started with


def test_backend_check_failure_is_treated_as_started(monkeypatch):
    import jax._src.xla_bridge as xla_bridge

    def broken():
        raise RuntimeError("API changed")

    monkeypatch.setattr(xla_bridge, "backends_are_initialized", broken)
    assert _xla_threads._backend_started()


class _FakeMCMC:
    def __init__(self, kernel, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    def print_summary(self):
        pass

    def get_samples(self):
        return {"log_stellar_mass": np.array([10.0, 10.2])}

    def get_extra_fields(self, group_by_chain=False):
        return {
            "diverging": np.array([[False, False]]),
            "num_steps": np.array([[7, 15]]),
            "accept_prob": np.array([[0.85, 0.9]]),
            "potential_energy": np.array([[8.0, 8.25]]),
            "energy": np.array([[10.0, 10.5]]),
        }


@pytest.mark.parametrize("safe", [True, False])
def test_fit_nuts_falls_back_when_pool_is_unsafe(monkeypatch, safe):
    calls, seen = [], []
    fit_globals = JAXSEDFit.fit_nuts.__globals__
    monkeypatch.setitem(fit_globals, "NUTS", lambda model, **kwargs: "kernel")
    monkeypatch.setitem(fit_globals, "MCMC", _FakeMCMC)
    monkeypatch.setitem(fit_globals, "_defer_nuts_postprocessing", lambda kernel: None)
    monkeypatch.setitem(fit_globals, "_use_callback_gradient", lambda *args: calls.append(args))
    monkeypatch.setattr(_xla_threads, "callback_is_safe", lambda n=1: (seen.append(n), safe)[1])
    fitter = JAXSEDFit.__new__(JAXSEDFit)
    fitter.config = _mock_config()
    fitter.config.inference.num_chains = 1
    fitter.config.inference.nuts_gradient_callback = True
    fitter.map_result = None
    fitter.predictive = {}
    fitter._model = lambda: None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        JAXSEDFit.fit_nuts(fitter, use_map_init=False, progress_bar=False)
    pool_warnings = [
        w for w in caught if issubclass(w.category, RuntimeWarning) and "thread pool" in str(w.message)
    ]
    assert seen == [1]
    assert len(calls) == (1 if safe else 0)
    assert len(pool_warnings) == (0 if safe else 1)
