import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from jaxsedfit.config import AGNConfig, EmissionLineTemplate, FeIITemplate, FilterCurve, FilterSet, GalaxyConfig
from jaxsedfit.model import _flat_lcdm_luminosity_distance_m_jax, _luminosity_distance_m_jax
from jaxsedfit.simulation import SimulationResult, simulate, simulation_config

FILTERS = ["f1", "f2", "f3"]


@pytest.fixture
def fake_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-1.0, -0.5, 0.0, 0.5])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})


def _config(redshift_range=None, agn_prior=True):
    filters = FilterSet(curves=[
        FilterCurve(name="f1", wave=[3000.0, 4000.0, 5000.0], transmission=[0.0, 1.0, 0.0]),
        FilterCurve(name="f2", wave=[5000.0, 6000.0, 7000.0], transmission=[0.0, 1.0, 0.0]),
        FilterCurve(name="f3", wave=[7000.0, 8000.0, 9000.0], transmission=[0.0, 1.0, 0.0]),
    ])
    cfg = simulation_config(
        FILTERS,
        redshift=0.3,
        redshift_range=redshift_range,
        n_redshift_grid=32,
        filters=filters,
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", rest_wave_max=20000.0, n_wave=512, sfh_n_steps=16),
        agn=AGNConfig(
            feii_template=FeIITemplate(name="fe", wave=[1000.0, 2000.0], lumin=[1.0, 0.5], wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(
                wave=[121.6, 486.1], lumin_blagn=[1.0, 0.5], lumin_sy2=[0.2, 0.1], lumin_liner=[0.1, 0.05],
                wavelength_unit="angstrom",
            ),
        ),
    )
    if agn_prior:
        cfg.prior_config.agn.log_amp = dist.Uniform(np.log(1e34), np.log(1e37))
    return cfg


def test_simulate_shapes_fixed_params_and_components(fake_ssp):
    mass = np.linspace(8.0, 11.0, 7)
    res = simulate(_config(), {"log_stellar_mass": mass}, seed=1)
    assert isinstance(res, SimulationResult) and len(res) == 7
    assert res.filter_names == FILTERS
    assert res.fluxes.shape == (7, 3) and np.isfinite(res.fluxes).all()
    np.testing.assert_allclose(res.params["log_stellar_mass"], mass)
    for comp in ("host_fluxes", "agn_fluxes", "variable_agn_fluxes", "constant_agn_fluxes"):
        assert res.components[comp].shape == (7, 3)
    np.testing.assert_allclose(
        res.components["variable_agn_fluxes"] + res.components["constant_agn_fluxes"], res.components["agn_fluxes"]
    )
    # drawn parameters follow the configured prior
    amp = res.params["log_agn_amp"]
    assert (amp >= np.log(1e34) - 1e-9).all() and (amp <= np.log(1e37) + 1e-9).all()


def test_simulate_is_reproducible_and_independent_of_batch_size(fake_ssp):
    cfg = _config()
    a = simulate(cfg, n=10, seed=3, batch_size=4)
    b = simulate(cfg, n=10, seed=3, batch_size=10)
    c = simulate(cfg, n=10, seed=4)
    np.testing.assert_allclose(a.fluxes, b.fluxes)
    np.testing.assert_allclose(a.params["log_agn_amp"], b.params["log_agn_amp"])
    assert not np.allclose(a.params["log_agn_amp"], c.params["log_agn_amp"])


def test_simulate_with_sampled_redshift_and_repeated_traces(fake_ssp):
    cfg = _config(redshift_range=(0.1, 1.5))
    z = np.linspace(0.2, 1.4, 6)
    first = simulate(cfg, {"redshift": z}, seed=0)
    second = simulate(cfg, {"redshift": z[:3]}, seed=0)  # new context and a separate jit trace
    np.testing.assert_allclose(first.params["redshift"], z)
    assert np.isfinite(first.fluxes).all() and np.isfinite(second.fluxes).all()
    drawn = simulate(cfg, n=20, seed=2).params["redshift"]
    assert (drawn >= 0.1 - 1e-9).all() and (drawn <= 1.5 + 1e-9).all()


def test_simulate_warns_about_photometry_seeded_agn_prior(fake_ssp):
    with pytest.warns(UserWarning, match="seeded from the configured"):
        simulate(_config(agn_prior=False), n=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        simulate(_config(agn_prior=False), {"log_agn_amp": np.full(2, np.log(1e35))})
        simulate(_config(agn_prior=True), n=2)


def test_simulate_validates_inputs(fake_ssp):
    cfg = _config()
    with pytest.raises(ValueError, match="different lengths"):
        simulate(cfg, {"log_stellar_mass": np.ones(3), "redshift": np.ones(4)})
    with pytest.raises(ValueError, match="give n"):
        simulate(cfg)
    with pytest.raises(ValueError, match="does not match"):
        simulate(cfg, {"log_stellar_mass": np.ones(3)}, n=5)
    with pytest.raises(ValueError, match="redshift_range"):
        simulation_config(FILTERS, redshift_range=(1.0, 0.5))


def test_simulation_result_dataframe_and_no_components(fake_ssp):
    cfg = _config()
    res = simulate(cfg, n=4, seed=0)
    df = res.to_dataframe()
    assert len(df) == 4 and "log_agn_amp" in df and "flux_f2" in df and "host_fluxes_f3" in df
    bare = simulate(cfg, n=4, seed=0, include_components=False)
    assert bare.components == {}
    np.testing.assert_allclose(bare.fluxes, res.fluxes)


def test_luminosity_distance_survives_separate_jit_traces():
    """jax_cosmo memoizes a distance table on the cached Cosmology; a tracer stored there by one
    jit trace must not leak into another."""
    f1 = jax.jit(lambda z: _luminosity_distance_m_jax(z, 70.0, 0.3))
    f2 = jax.jit(lambda z: 2.0 * _luminosity_distance_m_jax(z, 70.0, 0.3))
    d1 = float(f1(jnp.asarray(0.5)))
    d2 = float(f2(jnp.asarray([0.5, 1.0]))[0]) / 2.0
    ref = float(_flat_lcdm_luminosity_distance_m_jax(jnp.asarray(0.5), 70.0, 0.3))
    assert d1 == pytest.approx(d2, rel=1e-10)
    assert d1 == pytest.approx(ref, rel=1e-3)
