"""Optional Noll et al. (2009) galaxy attenuation curve (GalaxyConfig.attenuation_curve = "noll2009")."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from jaxsedfit.config import AGNConfig, EmissionLineTemplate, FeIITemplate, FilterCurve, FilterSet, GalaxyConfig
from jaxsedfit.model import (
    GRAHSP_BIATTENUATION_BREAK_A,
    DustScreen,
    _apply_biattenuation,
    _attenuation_curve,
    _calzetti_k,
    _noll2009_attenuation_curve,
    _noll2009_bump_strength,
    _project_fixed_cached_local_line_filters,
    _screen_magnitudes,
)
from jaxsedfit.simulation import simulate, simulation_config

jax.config.update("jax_enable_x64", True)
FILTERS = ["f1", "f2", "f3"]
WAVE = np.geomspace(100.0, 3.0e6, 2000)


def _grahsp(wave):
    return np.asarray(_attenuation_curve(wave, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A))


def _drude(wave, strength):
    width = (wave * 350.0) ** 2
    return strength * width / ((wave**2 - 2175.0**2) ** 2 + width)


def test_calzetti_k_matches_published_fit_and_is_continuous():
    k = lambda w: float(_calzetti_k(jnp.asarray(w)))
    assert k(5500.0) == pytest.approx(4.05, abs=0.01)  # R_V
    assert k(4400.0) - k(5500.0) == pytest.approx(1.0, abs=0.02)  # k is per unit E(B-V)
    assert k(1500.0) == pytest.approx(2.659 * (-2.156 + 1.509 / 0.15 - 0.198 / 0.15**2 + 0.011 / 0.15**3) + 4.05)
    for edge in (1200.0, 22000.0):  # the extrapolations join continuously
        assert k(edge * (1 - 1e-9)) == pytest.approx(k(edge * (1 + 1e-9)), rel=1e-6)
    assert abs(k(6300.0 * (1 - 1e-9)) - k(6300.0 * (1 + 1e-9))) < 0.03  # the published two-piece fit's own step
    values = np.asarray(_calzetti_k(WAVE))
    assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
    for piece in (WAVE < 6300.0, WAVE > 6300.0):  # declining with wavelength on each side of the step
        assert np.all(np.diff(values[piece]) <= 1e-12)


def test_delta_zero_is_calzetti_plus_milky_way_bump_at_2175():
    curve = np.asarray(_noll2009_attenuation_curve(WAVE, 0.0))
    raw = np.asarray(_calzetti_k(WAVE)) + _drude(WAVE, 0.85)
    norm = (float(_calzetti_k(4400.0)) + _drude(4400.0, 0.85)) - (float(_calzetti_k(5500.0)) + _drude(5500.0, 0.85))
    np.testing.assert_allclose(curve, raw / norm, rtol=1e-12)
    excess = curve * norm - np.asarray(_calzetti_k(WAVE))
    assert WAVE[np.argmax(excess)] == pytest.approx(2175.0, rel=0.005)
    assert excess.max() == pytest.approx(0.85, rel=1e-3)


@pytest.mark.parametrize("delta", [-0.8, -0.4, -0.2, 0.0, 0.2, 0.4])
def test_curve_is_normalized_to_unit_colour_excess(delta):
    curve = lambda w: float(_noll2009_attenuation_curve(jnp.asarray(w), delta))
    assert curve(4400.0) - curve(5500.0) == pytest.approx(1.0, rel=1e-12)
    values = np.asarray(_noll2009_attenuation_curve(WAVE, delta))
    assert np.all(np.isfinite(values)) and np.all(values >= 0.0)


def test_kriek_conroy_bump_and_steeper_uv_for_negative_delta():
    assert float(_noll2009_bump_strength(0.0)) == pytest.approx(0.85)
    assert float(_noll2009_bump_strength(-0.5)) == pytest.approx(1.80)
    assert float(_noll2009_bump_strength(0.45)) == 0.0  # clipped: no negative bumps
    uv_over_v = [float(_noll2009_attenuation_curve(1500.0, d) / _noll2009_attenuation_curve(5500.0, d)) for d in (-0.6, -0.2, 0.2)]
    assert uv_over_v[0] > uv_over_v[1] > uv_over_v[2]


def test_curve_gradient_in_delta_is_finite():
    grad = jax.vmap(jax.grad(lambda d, w: _noll2009_attenuation_curve(w, d)), in_axes=(None, 0))
    for delta in (-0.8, -0.1, 0.4, 0.45):
        g = np.asarray(grad(jnp.asarray(delta), jnp.asarray(WAVE)))
        assert np.all(np.isfinite(g))


def test_biattenuation_default_is_unchanged_and_noll_splits_host_and_agn_screens():
    wave = np.geomspace(912.0, 5.0e4, 300)
    gal, agn = np.ones_like(wave), 2.0 * np.ones_like(wave)
    curve = _grahsp(wave)
    gal_att, agn_att, _, _ = _apply_biattenuation(wave, gal, agn, 0.3, 0.5, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A)
    np.testing.assert_allclose(gal_att, gal * 10 ** (-0.4 * 0.3 * curve), rtol=1e-12)
    np.testing.assert_allclose(agn_att, agn * 10 ** (-0.4 * 0.8 * curve), rtol=1e-12)

    noll = np.asarray(_noll2009_attenuation_curve(wave, -0.3))
    gal_att, agn_att, _, dust = _apply_biattenuation(
        wave, gal, agn, 0.3, 0.5, -1.2, -3.0, 1.2, GRAHSP_BIATTENUATION_BREAK_A, noll_delta=-0.3
    )
    np.testing.assert_allclose(gal_att, gal * 10 ** (-0.4 * 0.3 * noll), rtol=1e-12)
    np.testing.assert_allclose(agn_att, agn * 10 ** (-0.4 * (0.3 * noll + 0.5 * curve)), rtol=1e-12)
    assert float(dust) > 0.0


def test_screen_magnitudes_bare_ebv_and_dust_screen():
    wave = jnp.asarray(np.geomspace(912.0, 5.0e4, 50))
    np.testing.assert_allclose(_screen_magnitudes(0.4, wave), 0.4 * _grahsp(np.asarray(wave)), rtol=1e-12)
    # An AGN-only screen reduces to the GRAHSP curve; a host screen to the Noll curve.
    np.testing.assert_allclose(_screen_magnitudes(DustScreen(0.0, 0.4, -0.2), wave), _screen_magnitudes(0.4, wave), rtol=1e-12)
    np.testing.assert_allclose(
        _screen_magnitudes(DustScreen(0.4, 0.0, -0.2), wave), 0.4 * _noll2009_attenuation_curve(wave, -0.2), rtol=1e-12
    )


def test_cached_line_projection_uses_cached_rest_wavelengths_for_noll_screens():
    rng = np.random.default_rng(0)
    n_width, n_filters, n_lines, n_local = 3, 2, 4, 5
    rest_wave = np.sort(rng.uniform(1500.0, 7000.0, (n_width, n_lines, n_local)), axis=-1)
    cache = SimpleNamespace(
        log_width_grid=jnp.log(jnp.asarray([100.0, 1000.0, 10000.0])),
        profile_norm=jnp.asarray(rng.uniform(0.5, 1.0, (n_width, n_lines, n_local))),
        attenuation_curve=jnp.asarray(_grahsp(rest_wave)),
        projection_weight=jnp.asarray(rng.uniform(0.0, 1.0, (n_width, n_filters, n_lines, n_local))),
        rest_line_wave=jnp.asarray(rest_wave),
    )
    context = SimpleNamespace(fixed_local_line_projection_cache_jax=cache)
    lumin = jnp.asarray(rng.uniform(1.0, 2.0, n_lines))
    width = 1000.0  # on a grid node: no width interpolation
    bare = _project_fixed_cached_local_line_filters(context, lumin, width, 0.3)
    agn_only = _project_fixed_cached_local_line_filters(context, lumin, width, DustScreen(0.0, 0.3, -0.2))
    np.testing.assert_allclose(agn_only, bare, rtol=1e-12)
    host = _project_fixed_cached_local_line_filters(context, lumin, width, DustScreen(0.3, 0.0, -0.2))
    att = 10 ** (-0.4 * 0.3 * np.asarray(_noll2009_attenuation_curve(rest_wave[1], -0.2)))
    expected = np.sum(np.asarray(cache.projection_weight[1]) * (np.asarray(lumin)[:, None] * np.asarray(cache.profile_norm[1]) * att)[None], axis=(1, 2))
    np.testing.assert_allclose(host, expected, rtol=1e-12)


def test_config_validates_attenuation_curve():
    assert GalaxyConfig().attenuation_curve == "grahsp"
    cfg = GalaxyConfig(attenuation_curve=" Noll2009 ")
    cfg.validate()
    assert cfg.attenuation_curve == "noll2009"
    with pytest.raises(ValueError, match="attenuation_curve"):
        GalaxyConfig(attenuation_curve="calzetti").validate()


@pytest.fixture
def fake_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.5, -1.0, -0.5])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 7, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})


def _config(attenuation_curve):
    filters = FilterSet(curves=[
        FilterCurve(name="f1", wave=[1500.0, 2000.0, 2500.0], transmission=[0.0, 1.0, 0.0]),
        FilterCurve(name="f2", wave=[5000.0, 6000.0, 7000.0], transmission=[0.0, 1.0, 0.0]),
        FilterCurve(name="f3", wave=[7000.0, 8000.0, 9000.0], transmission=[0.0, 1.0, 0.0]),
    ])
    cfg = simulation_config(
        FILTERS,
        redshift=0.01,
        filters=filters,
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", rest_wave_max=20000.0, n_wave=256, sfh_n_steps=64,
                            attenuation_curve=attenuation_curve),
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


def test_default_simulation_has_no_delta_site(fake_ssp):
    res = simulate(_config("grahsp"), {"log_stellar_mass": np.full(8, 10.0)}, seed=1)
    assert "dust_delta" not in res.params
    assert np.all(np.isfinite(res.fluxes))


def test_noll2009_simulation_samples_delta_and_it_moves_the_uv(fake_ssp):
    res = simulate(_config("noll2009"), {"log_stellar_mass": np.full(256, 10.0)}, seed=2)
    delta = res.params["dust_delta"]
    assert delta.min() >= -0.8 - 1e-9 and delta.max() <= 0.4 + 1e-9
    assert -0.35 < float(np.mean(delta)) < -0.05  # truncated N(-0.2, 0.3)
    assert np.all(np.isfinite(res.fluxes))

    fixed = {"log_stellar_mass": np.full(2, 10.0), "log_ebv_gal": np.log(np.full(2, 0.5)),
             "dust_delta": np.array([-0.6, 0.3])}
    for key in ("log_sfh_age_gyr", "log_sfh_tau_gyr", "log_agn_amp", "log_ebv_agn"):
        if key in res.params:
            fixed[key] = np.full(2, float(np.median(res.params[key])))
    out = simulate(_config("noll2009"), fixed, seed=3)
    uv_to_red = out.fluxes[:, 0] / out.fluxes[:, 2]
    assert uv_to_red[0] < uv_to_red[1]  # the steeper curve reddens the UV more at fixed E(B-V)
