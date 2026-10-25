"""Optional SFH prior uniform in log10 sSFR (GalaxyConfig.sfh_prior = "ssfr_uniform")."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from jaxsedfit.config import AGNConfig, EmissionLineTemplate, FeIITemplate, FilterCurve, FilterSet, GalaxyConfig
from numpyro import handlers

from jaxsedfit.model import _build_delayed_host, _burst_ssfr_per_yr, _delayed_ssfr_age_product, _solve_delayed_x
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
        ssp_flux = np.ones((4, 7, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})


def _config(sfh_prior="ssfr_uniform", host_sfh_model="delayed", ssfr_prior_range=(-13.0, -8.0)):
    filters = FilterSet(curves=[
        FilterCurve(name="f1", wave=[3000.0, 4000.0, 5000.0], transmission=[0.0, 1.0, 0.0]),
        FilterCurve(name="f2", wave=[5000.0, 6000.0, 7000.0], transmission=[0.0, 1.0, 0.0]),
        FilterCurve(name="f3", wave=[7000.0, 8000.0, 9000.0], transmission=[0.0, 1.0, 0.0]),
    ])
    cfg = simulation_config(
        FILTERS,
        redshift=0.3,
        filters=filters,
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", rest_wave_max=20000.0, n_wave=256, sfh_n_steps=64,
                            host_sfh_model=host_sfh_model, sfh_prior=sfh_prior, ssfr_prior_range=ssfr_prior_range),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[1.0, 0.5], wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(
                wave=[121.6, 486.1], lumin_blagn=[1.0, 0.5], lumin_sy2=[0.2, 0.1], lumin_liner=[0.1, 0.05],
                wavelength_unit="angstrom",
            ),
        ),
    )
    cfg.prior_config.agn.log_amp = dist.Uniform(np.log(1e34), np.log(1e35))
    return cfg


def _host_draws(cfg, n, seed):
    """Draw the host model n times: sampled sites and the model's current SFR / formed mass."""
    cfg.validate() if hasattr(cfg, "validate") else None
    ctx = build_model_context(cfg)
    prior = cfg.prior_config.to_mapping()

    def one(key):
        tr = handlers.trace(handlers.seed(
            lambda: _build_delayed_host(ctx, prior, full_output=False), key)).get_trace()
        state = _build_delayed_host_state_from(tr, ctx, prior, key)
        sites = {k: v["value"] for k, v in tr.items() if v["type"] in ("sample", "deterministic") and jnp.ndim(v["value"]) == 0}
        return sites, jnp.log10(state["current_sfr"]) - jnp.log10(state["formed_mass"])

    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    sites, achieved = jax.jit(jax.vmap(one))(keys)
    return {k: np.asarray(v) for k, v in sites.items()}, np.asarray(achieved)


def _build_delayed_host_state_from(tr, ctx, prior, key):
    # re-run with the traced values substituted to get the returned state
    data = {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}
    return handlers.substitute(handlers.seed(
        lambda: _build_delayed_host(ctx, prior, full_output=False), key), data=data)()


def test_delayed_ssfr_age_product_limits_and_solver_round_trip():
    x = jnp.array([1e-6, 1e-3, 0.1, 1.0, 3.0, 10.0, 50.0])
    h = np.asarray(_delayed_ssfr_age_product(x))
    assert abs(h[0] - 2.0) < 1e-5 and np.all(np.diff(h) < 0) and h[-1] < 1e-15
    targets = jnp.array([1.99, 1.5, 0.7, 0.1, 1e-3, 1e-8])
    xs = jax.vmap(_solve_delayed_x)(targets)
    np.testing.assert_allclose(np.asarray(jax.vmap(_delayed_ssfr_age_product)(xs)), np.asarray(targets), rtol=1e-8)


def test_solver_gradient_is_the_implicit_derivative():
    t = 0.37
    g = float(jax.grad(_solve_delayed_x)(t))
    eps = 1e-6
    fd = (float(_solve_delayed_x(t + eps)) - float(_solve_delayed_x(t - eps))) / (2 * eps)
    assert np.isfinite(g) and abs(g - fd) < 1e-4 * abs(fd)
    # composite identity: the sSFR reached by the solved x has unit derivative w.r.t. the target
    comp = jax.grad(lambda tt: _delayed_ssfr_age_product(_solve_delayed_x(tt)))(t)
    assert abs(float(comp) - 1.0) < 1e-6


def test_config_validation():
    with pytest.raises(ValueError, match="sfh_prior"):
        GalaxyConfig(sfh_prior="nope").validate()
    with pytest.raises(ValueError, match="ssfr_prior_range"):
        GalaxyConfig(sfh_prior="ssfr_uniform", ssfr_prior_range=(-9.0, -10.0)).validate()
    with pytest.raises(ValueError, match="diffstar"):
        GalaxyConfig(sfh_prior="ssfr_uniform", host_sfh_model="diffstar").validate()
    assert GalaxyConfig().sfh_prior == "age_tau"  # not the default


def test_default_prior_is_unchanged(fake_ssp):
    res = simulate(_config(sfh_prior="age_tau"), {"log_stellar_mass": np.full(64, 10.0)}, seed=3)
    assert "log_sfh_ssfr" not in res.params
    tau = np.exp(res.params["log_sfh_tau_gyr"])
    assert (tau >= 0.1 - 1e-9).all() and (tau <= 10.0 + 1e-9).all()  # GRAHSP-like tau prior


def test_delayed_ssfr_uniform_prior_is_uniform_and_reproduced_by_the_model(fake_ssp):
    n = 3000
    sites, achieved = _host_draws(_config(), n, seed=4)
    s = sites["log_sfh_ssfr"]
    assert s.min() >= -13.0 - 1e-9 and s.max() <= -8.0 + 1e-9
    frac = np.histogram(s, np.linspace(-13.0, -8.0, 6))[0] / n
    assert np.all(np.abs(frac - 0.2) < 0.03), frac
    # the model's own current SFR per formed mass equals the sampled sSFR
    np.testing.assert_allclose(achieved, s, atol=1e-6)
    np.testing.assert_allclose(sites["log_sfh_tau_over_age"], sites["log_sfh_tau_gyr"] - sites["log_sfh_age_gyr"], atol=1e-12)


def test_burst_ssfr_uniform_prior_sets_the_total_ssfr(fake_ssp):
    n = 2000
    sites, achieved = _host_draws(_config(host_sfh_model="delayed_burst"), n, seed=5)
    d = achieved - sites["log_sfh_ssfr"]
    f = np.exp(sites["log_sfh_burst_fraction"])
    sb = np.asarray(jax.vmap(_burst_ssfr_per_yr)(jnp.exp(sites["log_sfh_age_gyr"]), jnp.exp(sites["log_sfh_burst_age_gyr"]),
                                                  jnp.exp(sites["log_sfh_burst_tau_gyr"])))
    infeasible = f * sb >= 10 ** sites["log_sfh_ssfr"]  # the burst alone already exceeds the target
    assert 0.05 < infeasible.mean() < 0.95
    # feasible draws: the total sSFR (main + burst) equals the sampled value
    assert np.mean(np.abs(d[~infeasible]) < 1e-3) > 0.99
    # infeasible draws can only overshoot
    assert np.all(d[infeasible] > -1e-9)
    assert np.isfinite(f).all() and (f <= 0.2 + 1e-9).all()


def test_user_prior_on_log_sfh_ssfr_and_conflicting_priors(fake_ssp):
    cfg = _config()
    cfg.prior_config.host.log_sfh_ssfr = dist.Uniform(-11.0, -10.0)
    sites, achieved = _host_draws(cfg, 200, seed=6)
    assert (sites["log_sfh_ssfr"] >= -11.0).all() and (sites["log_sfh_ssfr"] <= -10.0).all()
    np.testing.assert_allclose(achieved, sites["log_sfh_ssfr"], atol=1e-6)
    bad = _config()
    bad.prior_config.host.log_sfh_tau_gyr = dist.Normal(0.0, 1.0)
    with pytest.raises(ValueError, match="log_sfh_ssfr"):
        simulate(bad, {"log_stellar_mass": np.full(4, 10.0)}, seed=7)
