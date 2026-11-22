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
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0])
        ssp_wave = np.array([100.0, 500.0, 900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 6))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())


def _config(consistent=None, host_capture=False):
    curves = []
    for name, (lo, hi) in BANDS_REST.items():
        wave = np.linspace(lo * (1 + REDSHIFT) - 1.0, hi * (1 + REDSHIFT) + 1.0, 600)
        trans = ((wave >= lo * (1 + REDSHIFT)) & (wave <= hi * (1 + REDSHIFT))).astype(float)
        curves.append(FilterCurve(name=name, wave=wave.tolist(), transmission=trans.tolist()))
    spec_wave = np.geomspace(*SPEC_REST, 400) * (1.0 + REDSHIFT)
    agn_kwargs = {} if consistent is None else {"consistent_line_coverage": consistent}
    return FitConfig(
        observation=Observation(object_id="obj", redshift=REDSHIFT),
        photometry=PhotometryData(
            filter_names=list(BANDS_REST),
            fluxes=[1.0] * 3,
            errors=[0.1] * 3,
            **({"psf_fwhm_arcsec": [1.4] * 3, "photometry_method": ["psf"] * 3} if host_capture else {}),
        ),
        filters=FilterSet(curves=curves),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake.h5", n_wave=512, fit_host=host_capture),
        agn=AGNConfig(
            fit_lines=True,
            tied_lines=True,
            fit_feii=False,
            fit_balmer_continuum=False,
            line_flux_scale_mjy=0.1,
            **agn_kwargs,
        ),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False, fit_spectrum_scale=False, use_host_capture_model=host_capture
        ),
        spectroscopy=SpectroscopyData(
            **({"aperture_diameter_arcsec": 2.0} if host_capture else {}),
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


def _coverage(context, state):
    return _consistent_line_coverage(
        context,
        context.fit_config,
        state,
        REDSHIFT,
        np.asarray(context.templates.line_wave),
        np.asarray(context.templates.line_blagn),
        np.asarray(context.templates.line_sy2),
    )


def _coverage_arrays(context, state):
    cov = _coverage(context, state)
    return np.asarray(cov.component_covered), np.asarray(cov.template_outside)


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

    covered, outside = _coverage_arrays(context, state)
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
    covered, outside = _coverage_arrays(context, state)
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


def test_template_lines_without_fitted_counterpart(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(consistent=False), monkeypatch)
    state = cap["state"]
    line_wave = np.asarray(context.templates.line_wave)
    nominal = _nominal(state)
    covered, outside = _coverage_arrays(context, state)
    legacy_outside = np.asarray(_photometric_agn_line_mask(context, context.fit_config, line_wave, REDSHIFT))
    has_component = np.array([np.any(np.abs(nominal / w - 1.0) < 0.005) for w in line_wave])
    obs = line_wave * (1.0 + REDSHIFT)
    spec_obs = np.array(SPEC_REST) * (1.0 + REDSHIFT)
    inside = (obs >= spec_obs[0]) & (obs <= spec_obs[1])
    # Inside the data and not in the spectral model: not extrapolated (as before).
    in_data = inside & ~has_component
    assert in_data.any() and np.all(outside[in_data] == 0.0)
    # In the coverage margin but outside the data and not fitted: legacy counts
    # it nowhere; now it is extrapolated once.
    margin = (legacy_outside == 0.0) & ~inside & ~has_component
    assert np.all(outside[margin] == 1.0)
    both = (legacy_outside == 0.0) & has_component
    assert both.any() and np.all(outside[both] == 0.0)


def _unmatched_amplitude_scan(context, state, select):
    line_wave = jnp.asarray(context.templates.line_wave)
    out = []
    for scale in (1.0, 5.0):
        amps = np.asarray(state["line_amp_per_component"], dtype=float).copy()
        amps[select] *= scale
        s = {**state, "line_amp_per_component": jnp.asarray(amps)}
        cov = _coverage(context, s)
        broad, narrow, _, _ = model_module._spectral_family_extrapolation(
            context, context.fit_config, s, REDSHIFT, 1.0e26, 0.0, jnp.ones_like(context.rest_wave_jax),
            line_wave, jnp.asarray(context.templates.line_blagn), jnp.asarray(context.templates.line_sy2),
            coverage=cov,
        )
        out.append((np.asarray(broad), np.asarray(narrow)))
    return out


@pytest.mark.parametrize("family", ["narrow", "broad"])
def test_unmatched_fitted_components_do_not_set_the_anchor(fake_ssp, monkeypatch, family):
    context, _, cap = _run(_config(consistent=False), monkeypatch)
    state = {k: np.asarray(v) for k, v in cap["state"].items() if k.endswith("_per_component")}
    nominal = _nominal(state)
    line_wave = np.asarray(context.templates.line_wave)
    weight = np.asarray(context.templates.line_blagn if family == "broad" else context.templates.line_sy2)
    is_broad = state["line_broad_mask_per_component"] > 0.5
    family_mask = is_broad if family == "broad" else ~is_broad
    matched = np.array([np.any((np.abs(nominal_c / line_wave - 1.0) < 0.005) & (weight > 0.0)) for nominal_c in nominal])
    covered, _ = _coverage_arrays(context, state)
    unmatched = family_mask & ~matched & (covered > 0.0)
    if not unmatched.any():
        # Make one: move a covered component of this family to a wavelength
        # with no template line (e.g. H-delta-like 4102.89 A has none).
        index = int(np.flatnonzero(family_mask & (covered > 0.0))[0])
        state["line_ln_lambda0_per_component"] = state["line_ln_lambda0_per_component"].copy()
        state["line_mu_per_component"] = state["line_mu_per_component"].copy()
        state["line_ln_lambda0_per_component"][index] = np.log(2300.0)
        state["line_mu_per_component"][index] = np.log(2300.0)
        unmatched = np.zeros_like(unmatched)
        unmatched[index] = True
    (b1, n1), (b5, n5) = _unmatched_amplitude_scan(context, state, unmatched)
    np.testing.assert_allclose(b5, b1, rtol=1e-12)
    np.testing.assert_allclose(n5, n1, rtol=1e-12)


def test_coverage_is_jittable_and_extrapolation_differentiable(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(consistent=True), monkeypatch)
    state = {k: jnp.asarray(v) for k, v in cap["state"].items() if k.startswith("line_") and k.endswith("_per_component")}
    line_wave = jnp.asarray(context.templates.line_wave)

    @jax.jit
    def coverage(amps):
        return _consistent_line_coverage(
            context, context.fit_config, {**state, "line_amp_per_component": amps}, REDSHIFT, line_wave,
            jnp.asarray(context.templates.line_blagn), jnp.asarray(context.templates.line_sy2),
        )

    cov = coverage(state["line_amp_per_component"])
    assert cov.component_covered.shape == state["line_amp_per_component"].shape

    def total_broad(amps):
        s = {**state, "line_amp_per_component": amps}
        cov = _consistent_line_coverage(
            context, context.fit_config, s, REDSHIFT, line_wave,
            jnp.asarray(context.templates.line_blagn), jnp.asarray(context.templates.line_sy2),
        )
        broad, _, _, _ = model_module._spectral_family_extrapolation(
            context, context.fit_config, s, REDSHIFT, 1.0e26, 0.0, jnp.ones_like(context.rest_wave_jax),
            line_wave, jnp.asarray(context.templates.line_blagn), jnp.asarray(context.templates.line_sy2),
            coverage=cov,
        )
        return jnp.sum(broad)

    grad = jax.grad(total_broad)(state["line_amp_per_component"] + 1.0e-3)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_plotted_line_sed_omits_uncovered_fitted_components(fake_ssp, monkeypatch):
    _, tr_off, _ = _run(_config(consistent=False), monkeypatch)
    context, tr_on, cap = _run(_config(consistent=True), monkeypatch)
    obs = np.asarray(context.obs_wave_jax)
    lya_obs = 1215.67 * (1.0 + REDSHIFT)
    near = np.abs(obs - lya_obs) < 15.0
    assert near.any()
    off = np.asarray(tr_off["spectral_line_obs_sed"]["value"])[near]
    on = np.asarray(tr_on["spectral_line_obs_sed"]["value"])[near]
    assert np.max(on) < 0.2 * np.max(off)


def test_native_narrow_lines_removed_as_captured(fake_ssp, monkeypatch):
    """agn_fluxes keeps only the captured native narrow lines; the switch removes exactly that."""
    _, tr_off, _ = _run(_config(consistent=False, host_capture=True), monkeypatch)
    _, tr_on, _ = _run(_config(consistent=True, host_capture=True), monkeypatch)
    total = _site(tr_on, "agn_narrow_line_fluxes_total")
    captured = _site(tr_on, "captured_agn_narrow_line_fluxes")
    if np.allclose(total, captured):
        pytest.skip("host capture inactive in this configuration")
    # The extra continuum is exactly the uncaptured native narrow flux (other
    # terms differ through the coverage rule; compare the replaced part only).
    agn_minus_spectral = lambda tr: _site(tr, "agn_fluxes") - sum(
        _site(tr, name) for name in (
            "spectral_line_photometry", "spectral_extrapolated_broad_photometry",
            "spectral_extrapolated_narrow_photometry", "spectral_feii_photometry",
            "spectral_extrapolated_feii_photometry", "spectral_balmer_photometry",
        )
    )
    np.testing.assert_allclose(
        agn_minus_spectral(tr_on) - agn_minus_spectral(tr_off), total - captured, rtol=1e-9, atol=1e-15
    )
