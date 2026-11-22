import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

import jaxsedfit.model as model_module
from jaxsedfit.config import (
    AGNConfig,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    InferenceConfig,
    LikelihoodConfig,
    Observation,
    PhotometryData,
    SpectroscopyData,
)
from jaxsedfit.model import (
    GRAHSP_PL_BEND_LOC_A,
    GRAHSP_PL_BEND_WIDTH,
    GRAHSP_PL_CUTOFF_A,
    _consistent_line_coverage,
    _photometric_agn_line_mask,
    _project_spectral_line_state_filters,
    grahsp_photometric_model,
)
from jaxsedfit.preload import build_model_context


REDSHIFT = 2.0
# Spectrum starts at rest 1250.5 A: the tied-line table activates Ly-alpha (its
# support reaches ~1294 A) while the centre rule calls Ly-alpha uncovered.
SPEC_REST = (1250.5, 2900.0)
BANDS_REST = {"lya_band": (1150.0, 1290.0), "civ_band": (1480.0, 1620.0), "lyb_band": (1000.0, 1060.0)}


@pytest.fixture
def fake_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-1.0, 0.0])
        ssp_lg_age_gyr = np.array([-1.0, 0.0])
        ssp_wave = np.array([900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((2, 2, 4))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())


def _config(consistent=None):
    curves = []
    for name, (lo, hi) in BANDS_REST.items():
        wave = np.linspace(lo * (1 + REDSHIFT) - 1.0, hi * (1 + REDSHIFT) + 1.0, 600)
        trans = ((wave >= lo * (1 + REDSHIFT)) & (wave <= hi * (1 + REDSHIFT))).astype(float)
        curves.append(FilterCurve(name=name, wave=wave.tolist(), transmission=trans.tolist()))
    spec_wave = np.geomspace(*SPEC_REST, 400) * (1.0 + REDSHIFT)
    agn_kwargs = {} if consistent is None else {"consistent_line_coverage": consistent}
    return FitConfig(
        observation=Observation(object_id="obj", redshift=REDSHIFT),
        photometry=PhotometryData(filter_names=list(BANDS_REST), fluxes=[1.0] * 3, errors=[0.1] * 3),
        filters=FilterSet(curves=curves),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=512, fit_host=False),
        agn=AGNConfig(
            fit_lines=True,
            tied_lines=True,
            fit_feii=False,
            fit_balmer_continuum=False,
            line_flux_scale_mjy=0.1,
            **agn_kwargs,
        ),
        likelihood=LikelihoodConfig(variability_uncertainty=False, fit_spectrum_scale=False),
        spectroscopy=SpectroscopyData(
            wave_obs=spec_wave.tolist(),
            fluxes=np.full(spec_wave.size, 0.1).tolist(),
            errors=np.full(spec_wave.size, 0.02).tolist(),
            instrument="sdss",
        ),
        inference=InferenceConfig(map_steps=2),
    )


_PARAMS = {
    "log_agn_amp": np.array(85.0),
    "pl_slope": np.array(-1.5),
    "log_ebv_agn": np.array(np.log(1.0e-12)),
}


def _run(cfg, monkeypatch):
    captured = {}
    original = model_module._spectral_family_extrapolation

    def spy(*args, **kwargs):
        captured["state"] = args[2]
        captured["kwargs"] = kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "_spectral_family_extrapolation", spy)
    context = build_model_context(cfg)
    tr = trace(seed(substitute(grahsp_photometric_model, data=_PARAMS), jax.random.PRNGKey(3))).get_trace(
        context, include_components=True
    )
    return context, tr, captured


def _nominal(state):
    return np.exp(np.asarray(state["line_ln_lambda0_per_component"]))


def _site(tr, name):
    return np.asarray(tr[name]["value"])


def test_default_is_off_and_bit_identical_to_explicit_false(fake_ssp, monkeypatch):
    assert AGNConfig().consistent_line_coverage is False
    _, tr_default, captured = _run(_config(), monkeypatch)
    assert captured["kwargs"] == {}
    _, tr_off, _ = _run(_config(consistent=False), monkeypatch)
    for name in ("pred_fluxes", "spectral_line_photometry", "spectral_extrapolated_broad_photometry"):
        np.testing.assert_array_equal(_site(tr_default, name), _site(tr_off, name))


def test_lya_counted_once_when_spectrum_starts_at_1250(fake_ssp, monkeypatch):
    context, tr_off, cap_off = _run(_config(consistent=False), monkeypatch)
    state = cap_off["state"]
    nominal = _nominal(state)
    lya = np.abs(nominal - 1215.67) < 1.0
    assert lya.sum() == 3  # Lya_br is a 3-Gaussian fitted line here
    legacy_outside = np.asarray(
        _photometric_agn_line_mask(context, context.fit_config, np.asarray(context.templates.line_wave), REDSHIFT)
    )
    line_wave = np.asarray(context.templates.line_wave)
    template_lya = np.abs(line_wave - 1215.0) < 1.0
    # Legacy: Ly-alpha is fitted AND extrapolated.
    assert np.all(legacy_outside[template_lya] == 1.0)

    covered, outside = (np.asarray(x) for x in _consistent_line_coverage(context, context.fit_config, state, REDSHIFT, line_wave))
    assert np.all(covered[lya] == 0.0)
    assert np.all(outside[template_lya] == 1.0)

    context_on, tr_on, cap_on = _run(_config(consistent=True), monkeypatch)
    band = list(BANDS_REST).index("lya_band")
    # Fitted Ly-alpha no longer enters the photometry: the fitted-line photometry
    # equals the projection of the covered components only.
    weights = np.where(lya, 0.0, 1.0) * covered
    expected = _project_spectral_line_state_filters(
        context_on, cap_on["state"], REDSHIFT, broad_only=True, component_weight=jnp.asarray(weights)
    )
    expected_narrow = _project_spectral_line_state_filters(
        context_on, cap_on["state"], REDSHIFT, broad_only=False, component_weight=jnp.asarray(weights)
    )
    np.testing.assert_allclose(
        _site(tr_on, "spectral_line_photometry"), np.asarray(expected + expected_narrow), rtol=1e-12
    )
    lya_only = _project_spectral_line_state_filters(
        context_on, cap_on["state"], REDSHIFT, component_weight=jnp.asarray(lya.astype(float))
    )
    assert float(lya_only[band]) > 0.0
    np.testing.assert_allclose(
        _site(tr_off, "spectral_line_photometry")[band] - _site(tr_on, "spectral_line_photometry")[band],
        float(lya_only[band]),
        rtol=1e-9,
    )
    # Ly-alpha is still extrapolated once from the template.
    assert _site(tr_on, "spectral_extrapolated_broad_photometry")[band] > 0.0


def test_lines_fully_inside_or_outside_are_unchanged(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(consistent=False), monkeypatch)
    state = cap["state"]
    line_wave = np.asarray(context.templates.line_wave)
    covered, outside = (np.asarray(x) for x in _consistent_line_coverage(context, context.fit_config, state, REDSHIFT, line_wave))
    legacy_outside = np.asarray(_photometric_agn_line_mask(context, context.fit_config, line_wave, REDSHIFT))
    nominal = _nominal(state)
    civ = np.abs(nominal - 1549.06) < 1.0
    assert civ.any() and np.all(covered[civ] == 1.0)
    for rest in (1549.0, 1909.0):  # inside: fitted, not extrapolated, as before
        index = int(np.flatnonzero(np.abs(line_wave - rest) < 1.0)[0])
        assert outside[index] == legacy_outside[index] == 0.0
    for rest in (1026.0, 1035.0):  # outside: extrapolated, as before
        index = int(np.flatnonzero(np.abs(line_wave - rest) < 1.0)[0])
        assert outside[index] == legacy_outside[index] == 1.0
    _, tr_off, _ = _run(_config(consistent=False), monkeypatch)
    _, tr_on, _ = _run(_config(consistent=True), monkeypatch)
    civ_band = list(BANDS_REST).index("civ_band")
    lyb_band = list(BANDS_REST).index("lyb_band")
    # The C IV band sees only covered, fitted lines in both modes.
    assert _site(tr_on, "spectral_line_photometry")[civ_band] == pytest.approx(
        _site(tr_off, "spectral_line_photometry")[civ_band], rel=1e-12
    )
    assert _site(tr_on, "spectral_extrapolated_broad_photometry")[lyb_band] > 0.0


def test_covered_template_line_without_fitted_component_is_extrapolated(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(consistent=False), monkeypatch)
    state = cap["state"]
    line_wave = np.asarray(context.templates.line_wave)
    nominal = _nominal(state)
    covered, outside = (np.asarray(x) for x in _consistent_line_coverage(context, context.fit_config, state, REDSHIFT, line_wave))
    legacy_outside = np.asarray(_photometric_agn_line_mask(context, context.fit_config, line_wave, REDSHIFT))
    has_component = np.array([np.any(np.abs(nominal / w - 1.0) < 0.005) for w in line_wave])
    neither = (legacy_outside == 0.0) & ~has_component
    # Legacy: such lines are neither fitted nor extrapolated; now extrapolated once.
    assert neither.any()
    assert np.all(outside[neither] == 1.0)
    both = (legacy_outside == 0.0) & has_component
    assert both.any() and np.all(outside[both] == 0.0)


def test_coverage_is_jittable_and_extrapolation_differentiable(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(consistent=True), monkeypatch)
    state = {k: jnp.asarray(v) for k, v in cap["state"].items() if k.startswith("line_") and k.endswith("_per_component")}
    line_wave = jnp.asarray(context.templates.line_wave)

    @jax.jit
    def coverage(amps):
        return _consistent_line_coverage(context, context.fit_config, {**state, "line_amp_per_component": amps}, REDSHIFT, line_wave)

    covered, outside = coverage(state["line_amp_per_component"])
    assert covered.shape == state["line_amp_per_component"].shape

    def total_broad(amps):
        s = {**state, "line_amp_per_component": amps}
        cov, out = _consistent_line_coverage(context, context.fit_config, s, REDSHIFT, line_wave)
        broad, _, _, _ = model_module._spectral_family_extrapolation(
            context, context.fit_config, s, REDSHIFT, 1.0e26, 0.0, jnp.ones_like(context.rest_wave_jax),
            line_wave, jnp.asarray(context.templates.line_blagn), jnp.asarray(context.templates.line_sy2),
            component_covered=cov, template_outside_mask=out,
        )
        return jnp.sum(broad)

    grad = jax.grad(total_broad)(state["line_amp_per_component"] + 1.0e-3)
    assert np.all(np.isfinite(np.asarray(grad)))
