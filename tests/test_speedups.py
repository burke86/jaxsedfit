"""Exactness of the likelihood fast paths: collapsed metallicity basis, IGM table, uniform filter grids."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from jaxsedfit.config import AGNConfig, EmissionLineTemplate, FeIITemplate, FilterCurve, FilterSet, GalaxyConfig
from jaxsedfit.model import (
    _contract_host_basis,
    _fixed_metallicity_host_basis,
    _igm_transmission,
    _igm_transmission_at_redshift,
    _packed_filter_transmission,
    _project_local_grid_filters,
)
from jaxsedfit.preload import build_model_context
from jaxsedfit.simulation import simulate, simulation_config

jax.config.update("jax_enable_x64", True)
FILTERS = ["f1", "f2", "f3"]


@pytest.fixture
def fake_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.random.default_rng(0).uniform(0.5, 2.0, (4, 7, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})


def _config(redshift_range=None, **likelihood):
    rng = np.random.default_rng(1)
    wave = np.linspace(3000.0, 9000.0, 241)
    filters = FilterSet(curves=[
        FilterCurve(name=f"f{i + 1}", wave=wave.tolist(),
                    transmission=np.clip(np.exp(-0.5 * ((wave - c) / 500.0) ** 2) + 0.05 * rng.random(wave.size), 0, None).tolist())
        for i, c in enumerate((4000.0, 6000.0, 8000.0))
    ])
    kw = dict(redshift_range=redshift_range) if redshift_range else dict(redshift=0.5)
    cfg = simulation_config(
        FILTERS, filters=filters,
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", rest_wave_max=20000.0, n_wave=256, sfh_n_steps=64),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[1.0, 0.5], wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(
                wave=[121.6, 486.1], lumin_blagn=[1.0, 0.5], lumin_sy2=[0.2, 0.1], lumin_liner=[0.1, 0.05],
                wavelength_unit="angstrom",
            ),
        ),
        **kw,
    )
    for key, value in likelihood.items():
        setattr(cfg.likelihood, key, value)
    cfg.prior_config.agn.log_amp = dist.Uniform(np.log(1e34), np.log(1e35))
    return cfg


def test_collapsed_metallicity_basis_matches_full_contraction(fake_ssp):
    ctx = build_model_context(_config())
    fixed = _fixed_metallicity_host_basis(ctx, {})
    basis = ctx.host_basis_jax.rest_llambda
    assert fixed.shape == basis.shape[1:]
    rng = np.random.default_rng(2)
    age_weights = jnp.asarray(rng.uniform(0.0, 1.0, basis.shape[1]))
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf
    from jaxsedfit.model import _host_metallicity_parameters, _resolve_tied_metallicity

    _, shared = _resolve_tied_metallicity(ctx, {})
    lgmet, scatter = _host_metallicity_parameters(ctx, {}, shared)
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(lgmet, scatter, ctx.host_basis_jax.ssp_lgmet)
    host_weights = lgmet_weights[:, None] * age_weights[None, :]
    host_weights = host_weights / jnp.sum(host_weights)
    full = _contract_host_basis(host_weights, age_weights, basis, None)
    np.testing.assert_allclose(_contract_host_basis(host_weights, age_weights, basis, fixed), full, rtol=1e-12)


@pytest.mark.parametrize("key", ["gal_lgmet", "gal_lgmet_scatter", "log_gal_lgmet_scatter", "nebular_zgas"])
def test_collapsed_basis_is_skipped_when_metallicity_is_sampled(fake_ssp, key):
    ctx = build_model_context(_config())
    assert _fixed_metallicity_host_basis(ctx, {key: dist.Normal(0.0, 1.0)}) is None


def test_igm_redshift_table_matches_exact_model(fake_ssp):
    ctx = build_model_context(_config(redshift_range=(0.05, 4.5)))
    table = ctx.igm_redshift_table_jax
    assert table is not None and table.table.shape[1] < ctx.rest_wave_jax.size  # only blueward of Ly-alpha
    assert np.all(np.asarray(ctx.rest_wave_jax)[table.columns] < 1216.0)
    for z in np.random.default_rng(3).uniform(0.05, 4.5, 25):
        exact = np.asarray(_igm_transmission(ctx.igm_cache_jax, z))
        np.testing.assert_allclose(np.asarray(_igm_transmission_at_redshift(ctx, z)), exact, atol=2e-3)
    off = build_model_context(_config(redshift_range=(0.05, 4.5), igm_redshift_table_n_grid=0))
    assert off.igm_redshift_table_jax is None
    np.testing.assert_array_equal(_igm_transmission_at_redshift(off, 2.0), _igm_transmission(off.igm_cache_jax, 2.0))


def test_igm_redshift_table_gradient_is_finite(fake_ssp):
    ctx = build_model_context(_config(redshift_range=(0.05, 4.5)))
    g = jax.grad(lambda z: jnp.sum(_igm_transmission_at_redshift(ctx, z)))(2.3)
    assert np.isfinite(float(g)) and float(g) < 0.0  # more absorption at higher z


def test_uniform_filter_grid_matches_native_interpolation(fake_ssp):
    ctx = build_model_context(_config())
    curves = ctx.packed_filter_curves_jax
    obs = jnp.asarray(np.random.default_rng(4).uniform(2500.0, 9500.0, (40, 9)))
    native = jax.vmap(lambda w, t: jnp.interp(obs, w, t, left=0.0, right=0.0))(curves.wave, curves.transmission)
    fast = _packed_filter_transmission(curves, obs)
    assert fast.shape == native.shape
    np.testing.assert_allclose(fast, native, rtol=1e-9, atol=1e-12)  # uniformly sampled curves: exact
    flux = jnp.asarray(np.random.default_rng(5).uniform(0.5, 1.5, obs.shape))
    obs_sorted = jnp.sort(obs, axis=1)
    from dataclasses import replace

    slow = _project_local_grid_filters(replace(ctx, packed_filter_curves_jax=replace(curves, uniform_transmission=None)), obs_sorted, flux)
    np.testing.assert_allclose(_project_local_grid_filters(ctx, obs_sorted, flux), slow, rtol=1e-9)


def test_uniform_grid_follows_non_uniformly_sampled_curves():
    from types import SimpleNamespace

    from jaxsedfit.preload import _uniform_filter_grids

    wave = np.cumsum(np.random.default_rng(6).uniform(2.0, 20.0, 400)) + 4000.0
    trans = np.exp(-0.5 * ((wave - wave.mean()) / 600.0) ** 2)
    wave0, dwave, last, table = _uniform_filter_grids([SimpleNamespace(wave=wave, native_transmission=trans)])
    q = np.random.default_rng(7).uniform(wave[0], wave[-1], 5000)
    pos = (q - wave0[0]) / dwave[0]
    k = np.clip(np.floor(pos).astype(int), 0, int(last[0]) - 1)
    fast = table[0, k] + (table[0, k + 1] - table[0, k]) * (pos - k)
    assert np.abs(fast - np.interp(q, wave, trans)).max() < 1e-4
    assert table[0, int(last[0])] == trans[-1]  # the grid ends on the last native sample


def test_simulate_reuses_the_compiled_simulator(fake_ssp):
    cfg = _config(redshift_range=(0.05, 2.0))
    ctx = build_model_context(cfg)
    params = {"redshift": np.full(16, 1.0), "log_stellar_mass": np.full(16, 10.0)}
    a = simulate(cfg, params, seed=1, context=ctx)
    sim = ctx._simulator_cache[(True, True)]
    b = simulate(cfg, params, seed=1, context=ctx)
    assert ctx._simulator_cache[(True, True)] is sim and len(ctx._simulator_cache) == 1
    np.testing.assert_array_equal(a.fluxes, b.fluxes)
    simulate(cfg, params, seed=1, context=ctx, include_components=False)
    assert len(ctx._simulator_cache) == 2
