"""NUTS with the gradient callback must not hang on a one-CPU XLA thread pool."""

import os
import subprocess
import sys
import textwrap

# A model whose NUTS step and gradient each have enough independent operations that
# XLA:CPU spreads them over its thread pool. With a one-thread pool, the callback's
# gradient waits for the thread the callback itself blocks, and NUTS hangs.
_SCRIPT = textwrap.dedent(
    """
    import os
    import jaxsedfit  # first, before JAX starts, as in a normal fit
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
    print("PJRT_NPROC", os.environ["PJRT_NPROC"], "DONE", flush=True)
    """
)


def test_callback_nuts_completes_on_a_one_cpu_pool():
    env = {k: v for k, v in os.environ.items() if k not in ("PJRT_NPROC", "NPROC")}
    env.update(NPROC="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", _SCRIPT], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "PJRT_NPROC 2 DONE" in result.stdout


def test_explicit_pjrt_nproc_is_kept():
    env = {k: v for k, v in os.environ.items() if k != "PJRT_NPROC"}
    env["PJRT_NPROC"] = "5"
    code = "import os, jaxsedfit; print(os.environ['PJRT_NPROC'])"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "5"
