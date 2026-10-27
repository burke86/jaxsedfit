import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

from jaxsedfit.batched import BatchedInputs, make_batched_predictive, map_draws, trim_padding


def toy_model(wave, flux, scale, z_qso):
    amp = numpyro.sample("amp", dist.Normal(0.0, 10.0))
    slope = numpyro.sample("slope", dist.Normal(0.0, 1.0))
    model = numpyro.deterministic("model", scale * amp * (wave / 5000.0) ** slope * (1.0 + z_qso))
    numpyro.deterministic("total", jnp.sum(model))
    numpyro.sample("obs", dist.Normal(model, 1.0), obs=flux)


def _inputs(n=3, n_pix=7):
    out = []
    for i in range(n):
        wave = np.linspace(4000.0, 6000.0, n_pix) * (1.0 + 0.1 * i)
        out.append(dict(wave=wave, flux=np.ones(n_pix), scale=2.0, z_qso=0.1 * i))
    return out


@pytest.mark.parametrize("block", [1, 2, 3, 10])
def test_map_draws_matches_vmapping_every_draw(block):
    tree = {"a": jnp.arange(2 * 5 * 3.0).reshape(2, 5, 3), "b": jnp.arange(10.0).reshape(2, 5)}
    fn = lambda t: t["a"].sum(-1) * t["b"]  # one draw for all objects -> (objects,)
    expected = jax.vmap(fn, in_axes=1, out_axes=1)(tree)
    assert np.allclose(map_draws(fn, tree, block), expected)


def test_from_stream_stacks_leaves_that_vary_anywhere():
    inputs = _inputs(4)
    inputs[3]["scale"] = 3.0  # constant in the first chunk, varies later
    streamed = BatchedInputs.from_stream(toy_model, iter(inputs))
    assert streamed.varying == BatchedInputs(toy_model, inputs).varying
    assert len(streamed.varying) > len(BatchedInputs(toy_model, inputs[:2]).varying)
    streamed.stack(inputs)  # no recalibration needed


@pytest.mark.parametrize("max_parallel", [2, 256])
def test_batched_predictive_matches_numpyro_predictive(max_parallel):
    inputs = _inputs(3)
    layout = BatchedInputs(toy_model, inputs)
    arrays, redshift = layout.stack(inputs)
    rng = np.random.default_rng(0)
    samples = {"amp": jnp.asarray(rng.normal(size=(3, 5))), "slope": jnp.asarray(rng.normal(size=(3, 5)))}
    predict = make_batched_predictive(layout, ["model", "total", "missing_site"], max_parallel=max_parallel, flux=None)
    out = predict(samples, arrays, redshift, jnp.stack([jax.random.PRNGKey(i) for i in range(3)]))
    assert set(out) == {"model", "total"}
    for j, x in enumerate(inputs):
        expected = Predictive(toy_model, posterior_samples={k: v[j] for k, v in samples.items()},
                              return_sites=["model", "total"])(jax.random.PRNGKey(0), **{**x, "flux": None})
        for site in ("model", "total"):
            assert np.allclose(out[site][j], expected[site])


def test_trim_padding_only_trims_pixel_axes():
    assert trim_padding(np.zeros((5, 10)), 10, 7).shape == (5, 7)
    assert trim_padding(np.zeros((5, 3)), 10, 7).shape == (5, 3)
    assert trim_padding(np.zeros((5, 10)), 10, 10).shape == (5, 10)


def test_host_basis_cache_keeps_only_recent_grids(monkeypatch):
    from types import SimpleNamespace

    import jaxsedfit.preload as preload

    monkeypatch.setattr(preload, "_HOST_BASIS_CACHE", {})
    monkeypatch.setattr(preload, "_HOST_BASIS_CACHE_SIZE", 3)
    monkeypatch.setattr(preload, "_ssp_lyman_basis_np", lambda wave, llambda: (np.zeros(2), np.zeros(2)))
    monkeypatch.setattr(preload, "_surviving_fraction_for_imf", lambda ages, imf: np.ones_like(ages))
    ssp = SimpleNamespace(ssp_wave=np.linspace(1000.0, 10000.0, 50), ssp_flux=np.ones((2, 2, 50)),
                          ssp_lg_age_gyr=np.zeros(2))
    grids = [np.linspace(2000.0, 8000.0, 20) + i for i in range(6)]
    first = preload._build_host_basis(grids[0], ssp)
    for grid in grids[1:3]:
        preload._build_host_basis(grid, ssp)
    assert preload._build_host_basis(grids[0], ssp) is first  # a hit makes it most recent
    for grid in grids[3:]:
        preload._build_host_basis(grid, ssp)
    assert len(preload._HOST_BASIS_CACHE) == 3
    assert preload._build_host_basis(grids[0], ssp) is not first  # evicted by the three newer grids
