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


def _config(host_capture=False, spec_rest=SPEC_REST, agn_extra=None, local_lines=True):
    curves = []
    for name, (lo, hi) in BANDS_REST.items():
        wave = np.linspace(lo * (1 + REDSHIFT) - 1.0, hi * (1 + REDSHIFT) + 1.0, 600)
        trans = ((wave >= lo * (1 + REDSHIFT)) & (wave <= hi * (1 + REDSHIFT))).astype(float)
        curves.append(FilterCurve(name=name, wave=wave.tolist(), transmission=trans.tolist()))
    spec_wave = np.geomspace(*spec_rest, 400) * (1.0 + REDSHIFT)
    agn_kwargs = {}
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


def _template_outside(cov):
    """Extrapolation weight per template line; broad and narrow parts agree on a contiguous spectrum."""
    broad = np.asarray(cov.broad_template_outside)
    narrow = np.asarray(cov.narrow_template_outside)
    np.testing.assert_array_equal(broad, narrow)
    return broad


def _coverage_arrays(context, state):
    cov = _coverage(context, state)
    return np.asarray(cov.component_covered), _template_outside(cov)


def _site(tr, name):
    return np.asarray(tr[name]["value"])


def test_lya_counted_once_when_spectrum_starts_at_1250(fake_ssp, monkeypatch):
    context, tr, cap = _run(_config(), monkeypatch)
    state = cap["state"]
    nominal = _nominal(state)
    lya = np.abs(nominal - 1215.67) < 1.0
    nv = np.abs(nominal - 1240.14) < 1.0
    # The tied-line table fits Ly-alpha (3 Gaussians) and N V from their red
    # wings because the spectrum starts at 1250.5 A, within their support.
    assert lya.sum() == 3 and nv.any()
    line_wave = np.asarray(context.templates.line_wave)
    template_lya = np.abs(line_wave - 1215.0) < 1.0
    covered, outside = _coverage_arrays(context, state)
    # Both centres lie outside the data: the template lines are extrapolated
    # and the fitted Gaussians are left out of the photometry.
    assert np.all(covered[lya | nv] == 0.0)
    assert np.array_equal(covered == 0.0, lya | nv)
    assert np.all(outside[template_lya] == 1.0)
    band = list(BANDS_REST).index("lya_band")
    expected = _project_spectral_line_state_filters(
        context, state, REDSHIFT, broad_only=True, component_weight=jnp.asarray(covered)
    ) + _project_spectral_line_state_filters(
        context, state, REDSHIFT, broad_only=False, component_weight=jnp.asarray(covered)
    )
    np.testing.assert_allclose(_site(tr, "spectral_line_photometry"), np.asarray(expected), rtol=1e-12)
    dropped = _project_spectral_line_state_filters(
        context, state, REDSHIFT, component_weight=jnp.asarray((lya | nv).astype(float))
    )
    full = _project_spectral_line_state_filters(context, state, REDSHIFT)
    assert float(dropped[band]) > 0.0
    np.testing.assert_allclose(
        np.asarray(full - dropped), np.asarray(expected), rtol=1e-9, atol=1e-12 * float(np.max(np.abs(full)))
    )
    # Ly-alpha is extrapolated once from the template.
    assert _site(tr, "spectral_extrapolated_broad_photometry")[band] > 0.0


def test_lines_fully_inside_or_outside_are_unchanged(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(), monkeypatch)
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
    _, tr, _ = _run(_config(), monkeypatch)
    lyb_band = list(BANDS_REST).index("lyb_band")
    assert _site(tr, "spectral_extrapolated_broad_photometry")[lyb_band] > 0.0


def _inside(line_wave, spec_rest=SPEC_REST):
    obs = np.asarray(line_wave) * (1.0 + REDSHIFT)
    lo, hi = np.array(spec_rest) * (1.0 + REDSHIFT)
    return (obs >= lo) & (obs <= hi)


def test_template_lines_without_fitted_counterpart(fake_ssp, monkeypatch):
    context, _, cap = _run(_config(), monkeypatch)
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
    return fitted, _template_outside(cov), within, nearest


@pytest.mark.parametrize(
    "spec_rest",
    [(start, 2900.0) for start in (1200.0, 1210.0, 1215.0, 1216.0, 1227.5, 1236.0, 1241.0, 1250.5, 1300.0, 1396.0, 1401.0, 1405.0, 1416.0)]
    + [(1300.0, end) for end in (2790.0, 2798.5, 2800.0, 2826.4, 4858.0, 4862.0, 4910.5)]
    # H-alpha / [N II] / [S II] edges.
    + [(1300.0, end) for end in (6530.0, 6550.0, 6556.0, 6562.5, 6563.5, 6570.0, 6583.5, 6590.0, 6716.5, 6725.0, 6731.5, 6740.0)],
)
def test_every_template_line_counted_once_across_spectrum_edges(fake_ssp, monkeypatch, spec_rest):
    context, _, cap = _run(_config(spec_rest=spec_rest), monkeypatch)
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
    # Unmatched fitted components follow their own centre, without a margin.
    nominal = _nominal(state)
    covered = np.asarray(_coverage(context, state).component_covered)
    own_inside = _inside(nominal, spec_rest)
    np.testing.assert_array_equal(covered[~within], own_inside[~within].astype(float))
    # Matched components follow their template line exactly.
    np.testing.assert_array_equal(covered[within], _inside(line_wave[nearest[within]], spec_rest).astype(float))


def test_anchor_numerator_and_denominator_use_the_same_lines(fake_ssp, monkeypatch):
    """If the counted fitted lines follow the template ratio, every extrapolated line does too."""
    context, _, cap = _run(_config(), monkeypatch)
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
    extrapolated = (_template_outside(cov) > 0.0) & (blagn > 0.0)
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
    context, _, cap = _run(_config(), monkeypatch)
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
    context, _, cap = _run(_config(), monkeypatch)
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
    context, tr, _ = _run(_config(), monkeypatch)
    obs = np.asarray(context.obs_wave_jax)
    sed = np.asarray(tr["spectral_line_obs_sed"]["value"])
    # The SED grid is coarse (about 2% spacing): look within +-2%.
    near_lya = np.abs(np.log(obs / (1215.67 * (1.0 + REDSHIFT)))) < 0.02
    near_civ = np.abs(np.log(obs / (1549.06 * (1.0 + REDSHIFT)))) < 0.02
    assert near_lya.any() and near_civ.any()
    # The fitted Ly-alpha is not drawn (the template Ly-alpha is the one used).
    assert np.max(sed[near_lya]) < 0.05 * np.max(sed[near_civ])


_LINE_SITES = (
    "spectral_line_photometry",
    "spectral_extrapolated_broad_photometry",
    "spectral_extrapolated_narrow_photometry",
    "spectral_feii_photometry",
    "spectral_extrapolated_feii_photometry",
    "spectral_balmer_photometry",
)


def _continuum_minus_disc(tr):
    lines = sum(_site(tr, name) for name in _LINE_SITES)
    return _site(tr, "agn_fluxes") - lines - _site(tr, "disk_fluxes") - _site(tr, "torus_fluxes")


def test_agn_continuum_equals_disc_under_host_capture(fake_ssp, monkeypatch):
    """With host capture the native narrow lines are removed exactly as included."""
    _, tr, _ = _run(_config(host_capture=True), monkeypatch)
    total = _site(tr, "agn_narrow_line_fluxes_total")
    captured = _site(tr, "captured_agn_narrow_line_fluxes")
    assert np.any(total > 0.0) and np.any(captured < 0.99 * total), "host capture must be active here"
    scale = np.max(np.abs(_site(tr, "disk_fluxes")))
    np.testing.assert_allclose(_continuum_minus_disc(tr), 0.0, atol=1e-12 * scale)


def test_native_lines_not_corrected_without_local_line_photometry(fake_ssp, monkeypatch):
    """Without local line photometry no native line is subtracted, so nothing is added back."""
    _, tr, _ = _run(_config(host_capture=True, local_lines=False), monkeypatch)
    total = _site(tr, "agn_narrow_line_fluxes_total")
    captured = _site(tr, "captured_agn_narrow_line_fluxes")
    assert np.any(total > 0.0) and np.any(captured < 0.99 * total)
    # The AGN flux keeps the coarse native lines (broad plus captured narrow);
    # it must not gain the uncaptured narrow flux on top.
    native_in_agn = _site(tr, "line_bl_fluxes") + captured
    scale = np.max(np.abs(_site(tr, "disk_fluxes")))
    np.testing.assert_allclose(_continuum_minus_disc(tr), native_in_agn, atol=1e-9 * scale)


def test_removed_switches_are_rejected():
    with pytest.raises(TypeError, match="consistent_line_coverage"):
        from jaxsedfit.config import _coerce_dataclass

        _coerce_dataclass(AGNConfig, {"consistent_line_coverage": True})


def test_fitted_redshift_weights_sum_to_one(fake_ssp, monkeypatch):
    """With a fitted redshift the coverage step is smoothed; each transition still counts once."""
    import copy

    spec_rest = (1216.0, 2900.0)  # Ly-alpha sits on the smoothed blue edge
    context, _, cap = _run(_config(spec_rest=spec_rest), monkeypatch)
    cfg = copy.deepcopy(context.fit_config)
    cfg.observation.redshift_mode = "fit"
    assert cfg.observation.fits_redshift
    state = cap["state"]
    line_wave = np.asarray(context.templates.line_wave)
    cov = _consistent_line_coverage(
        context, cfg, state, REDSHIFT, line_wave,
        np.asarray(context.templates.line_blagn), np.asarray(context.templates.line_sy2),
    )
    covered = np.asarray(cov.component_covered)
    broad = np.asarray(state["line_broad_mask_per_component"]) > 0.5
    nominal = _nominal(state)
    rel = np.abs(nominal[None, :] / line_wave[:, None] - 1.0)
    nearest, within = np.argmin(rel, axis=0), np.min(rel, axis=0) < 0.005
    weights = covered[within]
    assert np.any((weights > 0.01) & (weights < 0.99)), "need a partially covered line"
    for c in np.flatnonzero(within):
        outside = cov.broad_template_outside if broad[c] else cov.narrow_template_outside
        assert covered[c] + float(np.asarray(outside)[nearest[c]]) == pytest.approx(1.0, abs=1e-12)
    blagn = np.asarray(context.templates.line_blagn)
    live = np.asarray(state["line_amp_per_component"]) != 0.0
    anchored = within & broad & live & (blagn[nearest] > 0.0)
    # The anchor weight equals the counted weight (not its square).
    np.testing.assert_allclose(np.asarray(cov.broad_anchor_weight)[anchored], covered[anchored], rtol=1e-12)


def test_fitted_redshift_coverage_is_one_half_at_the_data_edges(fake_ssp, monkeypatch):
    """The smoothed step is centred on the first and last valid pixel, for both line families."""
    import copy

    from jaxsedfit.model import _inside_spectral_data

    spec_rest = (1300.0, 2900.0)
    context, _, _ = _run(_config(spec_rest=spec_rest), monkeypatch)
    cfg = copy.deepcopy(context.fit_config)
    cfg.observation.redshift_mode = "fit"
    spec_wave = np.asarray(context.spec_wave_obs)
    spec_min, spec_max = float(spec_wave.min()), float(spec_wave.max())
    rest = np.array([spec_min, spec_max, 0.9 * spec_min, 1.1 * spec_max, 2.0 * spec_min]) / (1.0 + REDSHIFT)
    for family in (0, 1):  # broad, narrow
        w = np.asarray(_inside_spectral_data(context, cfg, rest, REDSHIFT, family))
        # The tabulated profile fraction is 0.5 at the edges to within the grid spacing.
        np.testing.assert_allclose(w[:2], 0.5, atol=0.02)
        np.testing.assert_allclose(w[2:4], 0.0, atol=1e-6)
        np.testing.assert_allclose(w[4], 1.0, atol=1e-6)


def test_zero_amplitude_component_leaves_its_line_out_of_the_denominator(fake_ssp, monkeypatch):
    context, _, _ = _run(_config(), monkeypatch)
    line_wave = np.asarray(context.templates.line_wave)
    blagn = np.asarray(context.templates.line_blagn)
    picks = [1549.0, 1909.0, 2798.0]
    idx = [int(np.flatnonzero(np.abs(line_wave - w) < 1.0)[0]) for w in picks]
    nominal = line_wave[idx]
    sigs = np.full(len(idx), 0.004)
    amps = blagn[idx] * nominal * (1.0 + REDSHIFT) / sigs
    amps[1] = 0.0  # C III] fitted with zero amplitude: absent
    state = {
        "line_amp_per_component": jnp.asarray(amps),
        "line_mu_per_component": jnp.asarray(np.log(nominal)),
        "line_ln_lambda0_per_component": jnp.asarray(np.log(nominal)),
        "line_sig_per_component": jnp.asarray(sigs),
        "line_broad_mask_per_component": jnp.ones(len(idx)),
    }
    cov = _coverage(context, state)
    assert float(np.asarray(cov.broad_template_covered)[idx[1]]) == 0.0
    broad, _, _, _ = model_module._spectral_family_extrapolation(
        context, context.fit_config, state, REDSHIFT, 1.0e26, 0.0, jnp.ones_like(context.rest_wave_jax),
        jnp.asarray(line_wave), jnp.asarray(blagn), jnp.asarray(context.templates.line_sy2), coverage=cov,
    )
    broad = np.asarray(broad)
    extrapolated = (_template_outside(cov) > 0.0) & (blagn > 0.0)
    conversion = 1.0e-10 / 299792458.0 * 1.0e29
    lumin0 = amps[0] * np.sqrt(2.0 * np.pi) * sigs[0] / (conversion * nominal[0] * (1.0 + REDSHIFT)) * 4.0 * np.pi * 1.0e52
    np.testing.assert_allclose(broad[extrapolated] / blagn[extrapolated], lumin0 / blagn[idx[0]], rtol=1e-10)


def test_unmatched_components_follow_their_own_centre(fake_ssp, monkeypatch):
    """Fitted lines without a template counterpart are counted only when their centre is in the data."""
    context, _, _ = _run(_config(spec_rest=(1300.0, 2900.0)), monkeypatch)
    # Si III] 1892 and Al III 1857 (no template line) lie inside the data;
    # H-delta 4102.89 and a 1250 A line lie outside it, far from the edges;
    # 2905 A lies outside the data but within the 3000 km/s (1%) margin of the
    # red edge, so a margin applied to unmatched lines would count it.
    nominal = np.array([1892.03, 1857.40, 4102.89, 1250.0, 2905.0])
    n = nominal.size
    state = {
        "line_amp_per_component": jnp.ones(n),
        "line_mu_per_component": jnp.asarray(np.log(nominal)),
        "line_ln_lambda0_per_component": jnp.asarray(np.log(nominal)),
        "line_sig_per_component": jnp.full(n, 0.004),
        "line_broad_mask_per_component": jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0]),
    }
    cov = _coverage(context, state)
    line_wave = np.asarray(context.templates.line_wave)
    rel = np.abs(nominal[None, :] / line_wave[:, None] - 1.0)
    assert np.all(np.min(rel, axis=0) >= 0.005)  # all are unmatched
    assert abs(2905.0 / 2900.0 - 1.0) < 3000.0 / 299792.458  # inside the old margin
    np.testing.assert_array_equal(np.asarray(cov.component_covered), [1.0, 1.0, 0.0, 0.0, 0.0])
    # Unmatched components never enter the anchors.
    assert np.all(np.asarray(cov.broad_anchor_weight) == 0.0)
    assert np.all(np.asarray(cov.narrow_anchor_weight) == 0.0)


# --------------------------------------------------------------------------- #
# Gaps and masks inside the spectrum
# --------------------------------------------------------------------------- #
CIV_REST = 1549.0


def _civ_index(context):
    line_wave = np.asarray(context.templates.line_wave)
    return int(np.argmin(np.abs(line_wave - CIV_REST)))


def _civ_components(context, state):
    nominal = _nominal(state)
    near = np.abs(nominal / CIV_REST - 1.0) < 0.005
    broad = np.asarray(state["line_broad_mask_per_component"]) > 0.5
    return near & broad, near & ~broad


def _masked_config(half_width_rest, *, drop_pixels=False):
    cfg = _config()
    wave = np.asarray(cfg.spectroscopy.wave_obs)
    hole = np.abs(wave / (1.0 + REDSHIFT) - CIV_REST) < half_width_rest
    if drop_pixels:  # a gap in the wavelength grid
        keep = ~hole
        cfg.spectroscopy.wave_obs = wave[keep].tolist()
        cfg.spectroscopy.fluxes = np.asarray(cfg.spectroscopy.fluxes)[keep].tolist()
        cfg.spectroscopy.errors = np.asarray(cfg.spectroscopy.errors)[keep].tolist()
    else:
        cfg.spectroscopy.mask = (~hole).tolist()
    return cfg


def test_line_in_a_masked_region_is_extrapolated(fake_ssp, monkeypatch):
    context, _, cap = _run(_masked_config(40.0), monkeypatch)
    state = cap["state"]
    cov = _coverage(context, state)
    civ = _civ_index(context)
    broad_c, narrow_c = _civ_components(context, state)
    assert broad_c.any()
    assert np.asarray(cov.broad_template_outside)[civ] == 1.0
    assert np.asarray(cov.narrow_template_outside)[civ] == 1.0
    assert np.all(np.asarray(cov.component_covered)[broad_c | narrow_c] == 0.0)
    # A line well away from the mask (Mg II 2798) is still fitted.
    line_wave = np.asarray(context.templates.line_wave)
    mgii = int(np.argmin(np.abs(line_wave - 2798.0)))
    assert np.asarray(cov.broad_template_outside)[mgii] == 0.0


def test_line_in_a_wavelength_gap_is_extrapolated(fake_ssp, monkeypatch):
    context, _, cap = _run(_masked_config(40.0, drop_pixels=True), monkeypatch)
    state = cap["state"]
    cov = _coverage(context, state)
    civ = _civ_index(context)
    broad_c, narrow_c = _civ_components(context, state)
    assert np.asarray(cov.broad_template_outside)[civ] == 1.0
    assert np.all(np.asarray(cov.component_covered)[broad_c | narrow_c] == 0.0)


def test_small_mask_at_the_peak_keeps_the_broad_line_and_drops_the_narrow_core(fake_ssp, monkeypatch):
    # +-3 A rest masks most of a 500 km/s narrow line but about a third of a
    # 3000 km/s broad line, whose wings still sample the profile.
    context, _, cap = _run(_masked_config(3.0), monkeypatch)
    state = cap["state"]
    cov = _coverage(context, state)
    civ = _civ_index(context)
    broad_c, narrow_c = _civ_components(context, state)
    assert np.asarray(cov.broad_template_outside)[civ] == 0.0
    assert np.asarray(cov.narrow_template_outside)[civ] == 1.0
    assert np.all(np.asarray(cov.component_covered)[broad_c] == 1.0)
    assert np.all(np.asarray(cov.component_covered)[narrow_c] == 0.0)
    # Exactly once per family: the narrow C IV is extrapolated, not fitted.
    blagn = np.asarray(context.templates.line_blagn)
    narrow_t = np.asarray(context.templates.line_sy2)
    assert blagn[civ] > 0.0 and narrow_t[civ] > 0.0
    assert np.asarray(cov.broad_template_covered)[civ] == 1.0
    assert np.asarray(cov.narrow_template_covered)[civ] == 0.0


def test_a_second_unmasked_spectrum_covers_the_line(fake_ssp, monkeypatch):
    cfg = _masked_config(40.0)
    first = cfg.spectroscopy
    second = SpectroscopyData(
        wave_obs=list(first.wave_obs), fluxes=list(first.fluxes), errors=list(first.errors), instrument="boss"
    )
    cfg.spectroscopy = [first, second]
    context, _, cap = _run(cfg, monkeypatch)
    cov = _coverage(context, cap["state"])
    assert np.asarray(cov.broad_template_outside)[_civ_index(context)] == 0.0


def test_pixel_intervals_tile_the_data_and_stop_at_edges_and_gaps():
    from jaxsedfit.preload import spectral_pixel_intervals

    wave = np.array([100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 200.0, 201.0, 202.0])
    mask = np.ones(wave.size, dtype=bool)
    mask[5] = False
    lo, hi = spectral_pixel_intervals(wave, mask, np.array([0] * 7 + [1] * 3))
    np.testing.assert_allclose(lo[:4], [100.0, 100.5, 101.5, 102.5])  # first pixel does not extend outward
    np.testing.assert_allclose(hi[:4], [100.5, 101.5, 102.5, 103.0])  # 103 sits before a 7 A gap
    np.testing.assert_allclose([lo[4], hi[4]], [110.0, 110.5])  # 110 sits after the gap
    assert lo[5] == hi[5] == 111.0  # masked pixel: empty interval
    np.testing.assert_allclose([lo[7], hi[7]], [200.0, 200.5])  # second spectrum starts fresh
    np.testing.assert_allclose([lo[9], hi[9]], [201.5, 202.0])


def test_coverage_table_matches_a_direct_profile_integral():
    from scipy.special import ndtr

    from jaxsedfit.preload import LINE_COVERAGE_FWHM_KMS, spectral_line_coverage_table, spectral_pixel_intervals

    wave = np.geomspace(4000.0, 6000.0, 1500)
    mask = np.ones(wave.size, dtype=bool)
    mask[(wave > 4950.0) & (wave < 4970.0)] = False
    index = np.zeros(wave.size, dtype=int)
    grid, table = spectral_line_coverage_table(wave, mask, index)
    lo, hi = spectral_pixel_intervals(wave, mask, index)
    for centre in (4000.0, 4500.0, 4960.0, 4975.0, 6000.0, 6010.0):
        for family, fwhm in enumerate(LINE_COVERAGE_FWHM_KMS):
            sigma = centre * fwhm / 299792.458 / 2.354820045
            direct = np.sum(ndtr((hi - centre) / sigma) - ndtr((lo - centre) / sigma))
            log_grid = grid[0] + grid[1] * np.arange(table.shape[1])
            tabulated = np.interp(np.log(centre), log_grid, table[family])
            np.testing.assert_allclose(tabulated, direct, atol=2e-3)
