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


def _config(consistent=None, host_capture=False, spec_rest=SPEC_REST, agn_extra=None, local_lines=True):
    curves = []
    for name, (lo, hi) in BANDS_REST.items():
        wave = np.linspace(lo * (1 + REDSHIFT) - 1.0, hi * (1 + REDSHIFT) + 1.0, 600)
        trans = ((wave >= lo * (1 + REDSHIFT)) & (wave <= hi * (1 + REDSHIFT))).astype(float)
        curves.append(FilterCurve(name=name, wave=wave.tolist(), transmission=trans.tolist()))
    spec_wave = np.geomspace(*spec_rest, 400) * (1.0 + REDSHIFT)
    agn_kwargs = {} if consistent is None else {"consistent_line_coverage": consistent}
    agn_kwargs.update(agn_extra or {})
    return FitConfig(
        observation=Observation(object_id="obj", redshift=REDSHIFT),
        photometry=PhotometryData(
            filter_names=list(BANDS_REST),
            fluxes=[1.0] * 3,
            errors=[0.1] * 3,
            **({"psf_fwhm_arcsec": [1.4] * 3, "photometry_method": ["psf"] * 3} if host_capture else {}),
        ),
        filters=FilterSet(curves=curves),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake_consistent_coverage.h5", n_wave=512, fit_host=host_capture),
        agn=AGNConfig(
            fit_lines=True,
            tied_lines=True,
            fit_feii=False,
            fit_balmer_continuum=False,
            line_flux_scale_mjy=0.1,
            **agn_kwargs,
        ),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False,
            fit_spectrum_scale=False,
            use_host_capture_model=host_capture,
            use_local_line_photometry=local_lines,
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
        captured["args"] = args
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
    # Left out: Ly-alpha and N V (template 1240 A lies outside the data that
    # start at 1250.5 A, inside only the 3000 km/s margin).
    nv = np.abs(nominal - 1240.14) < 1.0
    assert nv.any() and np.all(covered[nv] == 0.0)
    assert np.array_equal(covered == 0.0, lya | nv)
    dropped = _project_spectral_line_state_filters(
        context_on, cap_on["state"], REDSHIFT, component_weight=jnp.asarray((lya | nv).astype(float))
    )
    assert float(dropped[band]) > 0.0
    np.testing.assert_allclose(
        _site(tr_off, "spectral_line_photometry")[band] - _site(tr_on, "spectral_line_photometry")[band],
        float(dropped[band]),
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


def _inside(line_wave, spec_rest=SPEC_REST):
    obs = np.asarray(line_wave) * (1.0 + REDSHIFT)
    lo, hi = np.array(spec_rest) * (1.0 + REDSHIFT)
    return (obs >= lo) & (obs <= hi)


def test_template_lines_without_fitted_counterpart(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(consistent=False), monkeypatch)
    state = cap["state"]
    line_wave = np.asarray(context.templates.line_wave)
    nominal = _nominal(state)
    covered, outside = _coverage_arrays(context, state)
    legacy_outside = np.asarray(_photometric_agn_line_mask(context, context.fit_config, line_wave, REDSHIFT))
    has_component = np.array([np.any(np.abs(nominal / w - 1.0) < 0.005) for w in line_wave])
    inside = _inside(line_wave)
    # Inside the data and not in the spectral model: not extrapolated.
    in_data = inside & ~has_component
    assert in_data.any() and np.all(outside[in_data] == 0.0)
    # Centre within the 3000 km/s margin but outside the data (N V 1240 for a
    # spectrum starting at 1250.5 A): extrapolated, and any fitted counterpart
    # is left out of the photometry.
    margin = (legacy_outside == 0.0) & ~inside
    assert margin.any(), "test spectrum must leave a template line in the margin"
    assert 1240.0 in set(np.round(line_wave[margin]))
    assert np.all(outside[margin] == 1.0)
    for w in line_wave[margin]:
        assert np.all(covered[np.abs(nominal / w - 1.0) < 0.005] == 0.0)


def _template_counts(context, state, spec_rest):
    """Return per template line (counted as fitted, counted as extrapolated)."""
    cov = _coverage(context, state)
    line_wave = np.asarray(context.templates.line_wave)
    nominal = _nominal(state)
    live = np.asarray(state["line_amp_per_component"]) != 0.0
    rel = np.abs(nominal[None, :] / line_wave[:, None] - 1.0)
    nearest = np.argmin(rel, axis=0)
    within = np.min(rel, axis=0) < 0.005
    fitted = np.zeros(line_wave.size)
    for c in range(nominal.size):
        if within[c] and live[c]:
            fitted[nearest[c]] = max(fitted[nearest[c]], float(np.asarray(cov.component_covered)[c]))
    return fitted, np.asarray(cov.template_outside), within, nearest


@pytest.mark.parametrize(
    "spec_rest",
    [(start, 2900.0) for start in (1200.0, 1210.0, 1215.0, 1216.0, 1227.5, 1236.0, 1241.0, 1250.5, 1300.0, 1396.0, 1401.0, 1405.0, 1416.0)]
    + [(1300.0, end) for end in (2790.0, 2798.5, 2800.0, 2826.4, 4858.0, 4862.0, 4910.5)],
)
def test_every_template_line_counted_once_across_spectrum_edges(fake_ssp, monkeypatch, spec_rest):
    context, _, cap = _run(_config(consistent=True, spec_rest=spec_rest), monkeypatch)
    state = cap["state"]
    fitted, extrapolated, within, nearest = _template_counts(context, state, spec_rest)
    line_wave = np.asarray(context.templates.line_wave)
    has_counterpart = np.zeros(line_wave.size, dtype=bool)
    has_counterpart[nearest[within]] = True
    total = fitted + extrapolated
    # Exactly once for every line with a fitted counterpart; template-only
    # lines are extrapolated outside the data and absent inside it.
    np.testing.assert_array_equal(total[has_counterpart], 1.0)
    np.testing.assert_array_equal(extrapolated[~has_counterpart], (~_inside(line_wave, spec_rest)[~has_counterpart]).astype(float))


def test_anchor_numerator_and_denominator_use_the_same_lines(fake_ssp, monkeypatch):
    """If the counted fitted lines follow the template ratio, every extrapolated line does too."""
    context, _, cap = _run(_config(consistent=True), monkeypatch)
    line_wave = np.asarray(context.templates.line_wave)
    blagn = np.asarray(context.templates.line_blagn)
    # Fitted broad components only at C IV, C III] and Mg II (inside the data);
    # N IV] 1486, He II 1640, N III] 1750 are template lines inside the data with
    # no fitted counterpart and must not enter the denominator.
    picks = [1549.0, 1909.0, 2798.0]
    idx = [int(np.flatnonzero(np.abs(line_wave - w) < 1.0)[0]) for w in picks]
    nominal = line_wave[idx]
    sigs = np.full(len(idx), 0.004)
    amps = blagn[idx] * nominal * (1.0 + REDSHIFT) / sigs  # L_c proportional to blagn
    state = {
        "line_amp_per_component": jnp.asarray(amps),
        "line_mu_per_component": jnp.asarray(np.log(nominal)),
        "line_ln_lambda0_per_component": jnp.asarray(np.log(nominal)),
        "line_sig_per_component": jnp.asarray(sigs),
        "line_broad_mask_per_component": jnp.ones(len(idx)),
    }
    cov = _coverage(context, state)
    broad, _, _, _ = model_module._spectral_family_extrapolation(
        context, context.fit_config, state, REDSHIFT, 1.0e26, 0.0, jnp.ones_like(context.rest_wave_jax),
        jnp.asarray(line_wave), jnp.asarray(blagn), jnp.asarray(context.templates.line_sy2), coverage=cov,
    )
    broad = np.asarray(broad)
    extrapolated = (np.asarray(cov.template_outside) > 0.0) & (blagn > 0.0)
    assert extrapolated.any()
    ratios = broad[extrapolated] / blagn[extrapolated]
    conversion = 1.0e-10 / 299792458.0 * 1.0e29
    lumin = amps * np.sqrt(2.0 * np.pi) * sigs / (conversion * nominal * (1.0 + REDSHIFT)) * 4.0 * np.pi * 1.0e52
    np.testing.assert_allclose(ratios, lumin[0] / blagn[idx[0]], rtol=1e-10)


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
    switch = {"subtract_captured_native_narrow_lines": True}
    _, tr_off, _ = _run(_config(host_capture=True), monkeypatch)
    _, tr_on, _ = _run(_config(host_capture=True, agn_extra=switch), monkeypatch)
    total = _site(tr_on, "agn_narrow_line_fluxes_total")
    captured = _site(tr_on, "captured_agn_narrow_line_fluxes")
    assert np.any(total > 0.0) and np.any(captured < 0.99 * total), "host capture must be active here"
    np.testing.assert_allclose(_site(tr_on, "agn_fluxes") - _site(tr_off, "agn_fluxes"), total - captured, rtol=1e-9)
    # pred_fluxes is dominated by the (fake) host here; compare to its precision.
    pred_scale = np.max(np.abs(_site(tr_off, "pred_fluxes")))
    np.testing.assert_allclose(
        _site(tr_on, "pred_fluxes") - _site(tr_off, "pred_fluxes"), total - captured, rtol=0, atol=1e-14 * pred_scale
    )


def test_native_narrow_fix_inactive_without_local_line_photometry(fake_ssp, monkeypatch):
    switch = {"subtract_captured_native_narrow_lines": True}
    _, tr_off, _ = _run(_config(host_capture=True, local_lines=False), monkeypatch)
    _, tr_on, _ = _run(_config(host_capture=True, local_lines=False, agn_extra=switch), monkeypatch)
    for site in ("agn_fluxes", "pred_fluxes"):
        np.testing.assert_array_equal(_site(tr_on, site), _site(tr_off, site))


def test_narrow_line_extrapolation_modes(fake_ssp, monkeypatch):
    import copy

    from jaxsedfit.model import NARROW_EXTRAPOLATION_OPTICAL_MIN_A

    assert AGNConfig().narrow_line_extrapolation == "template"
    with pytest.raises(ValueError, match="narrow_line_extrapolation"):
        AGNConfig(narrow_line_extrapolation="uv")
    context, tr_t, cap = _run(_config(consistent=True), monkeypatch)
    band = list(BANDS_REST).index("lya_band")
    assert _site(tr_t, "spectral_extrapolated_narrow_photometry")[band] > 0.0
    args = list(cap["args"])
    line_wave = np.asarray(args[7])
    results = {}
    for mode in ("template", "optical_only", "none"):
        cfg = copy.deepcopy(args[1])
        cfg.agn.narrow_line_extrapolation = mode
        broad, narrow, _, _ = model_module._spectral_family_extrapolation(
            args[0], cfg, *args[2:], **cap["kwargs"]
        )
        results[mode] = (np.asarray(broad), np.asarray(narrow))
    broad_t, narrow_t = results["template"]
    optical = line_wave >= NARROW_EXTRAPOLATION_OPTICAL_MIN_A
    assert narrow_t[~optical].sum() > 0.0 and narrow_t[optical].sum() > 0.0
    np.testing.assert_array_equal(results["optical_only"][1][optical], narrow_t[optical])
    assert np.all(results["optical_only"][1][~optical] == 0.0)
    assert np.all(results["none"][1] == 0.0)
    for mode in ("optical_only", "none"):
        np.testing.assert_array_equal(results[mode][0], broad_t)
    cfg = _config(consistent=True)
    cfg.agn.narrow_line_extrapolation = "none"
    _, tr_none, _ = _run(cfg, monkeypatch)
    assert np.all(_site(tr_none, "spectral_extrapolated_narrow_photometry") == 0.0)


def test_optical_only_threshold_on_fixed_lines(fake_ssp, monkeypatch):
    import copy

    # The spectrum stops at rest 2600 A so Mg II 2798 is extrapolated too.
    context, _, cap = _run(_config(consistent=True, spec_rest=(1250.5, 2600.0)), monkeypatch)
    args = list(cap["args"])
    line_wave = np.asarray(args[7])
    out = {}
    for mode in ("template", "optical_only"):
        cfg = copy.deepcopy(args[1])
        cfg.agn.narrow_line_extrapolation = mode
        out[mode] = np.asarray(model_module._spectral_family_extrapolation(args[0], cfg, *args[2:], **cap["kwargs"])[1])
    for rest, kept in ((1215.0, False), (2798.0, False), (3426.0, True), (3727.0, True), (5007.0, True)):
        index = np.flatnonzero(np.abs(line_wave - rest) < 2.0)
        assert index.size, rest
        index = int(index[0])
        assert out["template"][index] > 0.0, rest
        assert (out["optical_only"][index] == out["template"][index]) == kept, rest
        assert (out["optical_only"][index] == 0.0) == (not kept), rest


def test_narrow_mode_is_normalized_in_validate():
    cfg = _config(consistent=True)
    cfg.agn.narrow_line_extrapolation = " Optical_Only "
    cfg.validate()
    assert cfg.agn.narrow_line_extrapolation == "optical_only"
    cfg.agn.narrow_line_extrapolation = "uv"
    with pytest.raises(ValueError, match="narrow_line_extrapolation"):
        cfg.validate()
