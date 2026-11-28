"""Fixed-definition ("reference") chi-square diagnostics."""

import numpy as np
import pytest
import jax
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace

from jaxsedfit.config import (
    AGNConfig,
    FilterCurve,
    FilterSet,
    FitConfig,
    GalaxyConfig,
    InferenceConfig,
    LikelihoodConfig,
    NebularConfig,
    Observation,
    PhotometryData,
    PriorConfig,
    SpectroscopyData,
)
from jaxsedfit.model import grahsp_photometric_model, reference_spectroscopic_chi2
from jaxsedfit.preload import build_model_context

REF_SITES = (
    "sed_chi2_ref",
    "sed_reduced_chi2_ref",
    "spectroscopy_chi2_ref",
    "spectroscopy_n_ref",
    "spectroscopy_reduced_chi2_ref",
    "joint_chi2_ref",
    "joint_n_ref",
    "joint_reduced_chi2_ref",
)
LEGACY_SITES = (
    "sed_chi2",
    "sed_n_eff",
    "sed_reduced_chi2",
    "spectroscopy_chi2",
    "spectroscopy_n_eff",
    "spectroscopy_reduced_chi2",
    "joint_chi2",
    "joint_n_eff",
    "joint_reduced_chi2",
)


def test_reference_spectroscopic_chi2_formula():
    pred = np.array([1.0, 2.0, -3.0, 4.0, 5.0, 6.0])
    obs = np.array([1.5, 1.0, -2.0, np.nan, 5.0, 7.0])
    err = np.array([0.1, 0.2, 0.3, 0.1, 0.0, 0.5])
    mask = np.array([True, True, True, True, True, False])
    floor = 0.08
    out = reference_spectroscopic_chi2(pred, obs, err, mask, floor)
    # Valid: masked in, finite and positive error -> pixels 0, 1, 2.
    valid = np.array([0, 1, 2])
    var = err[valid] ** 2 + (floor * np.abs(pred[valid])) ** 2
    expected = np.sum((obs[valid] - pred[valid]) ** 2 / var)
    assert float(out["n_eff"]) == 3.0
    assert float(out["chi2"]) == pytest.approx(expected, rel=1e-12)
    assert float(out["reduced_chi2"]) == pytest.approx(expected / 3.0, rel=1e-12)

    empty = reference_spectroscopic_chi2(pred, obs, err, np.zeros(6, dtype=bool), floor)
    assert float(empty["n_eff"]) == 0.0
    assert np.isnan(float(empty["chi2"]))
    assert np.isnan(float(empty["reduced_chi2"]))


def _cfg(**likelihood_kwargs):
    return FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(filter_names=["f1", "f2"], fluxes=[1.0, 1.2], errors=[0.1, 0.1]),
        filters=FilterSet(
            curves=[
                FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0]),
                FilterCurve(name="f2", wave=[3000.0, 4000.0, 5000.0], transmission=[0.0, 1.0, 0.0]),
            ]
        ),
        galaxy=GalaxyConfig(dsps_ssp_fn="fake_refchi2.h5", n_wave=16, fit_host=True, fit_host_kinematics=True),
        agn=AGNConfig(fit_agn=False, fit_lines=False),
        nebular=NebularConfig(enabled=False),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False,
            fit_spectrum_scale=False,
            **likelihood_kwargs,
        ),
        spectroscopy=SpectroscopyData(
            wave_obs=[5000.0, 5001.0, 5002.0, 5003.0],
            fluxes=[2.0, 4.0, 5.0, 3.0],
            errors=[0.1, 0.2, 0.1, 0.3],
            instrument="sdss",
            resolving_power=2000.0,
        ),
        inference=InferenceConfig(map_steps=2),
        prior_config=PriorConfig(stellar_mass=dist.Normal(8.0, 1.0e-6)),
    )


@pytest.fixture
def _fake_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0])
        ssp_wave = np.array([100.0, 500.0, 900.0, 2000.0, 5000.0, 10000.0])
        ssp_flux = np.ones((4, 4, 6))

    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._HOST_BASIS_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr(
        "jaxsedfit.model._shift_and_broaden_single_spectrum_lnlam",
        lambda lnwave, spectrum, v_kms, sigma_kms: spectrum,
    )


def _trace(context, data):
    return trace(
        substitute(seed(grahsp_photometric_model, jax.random.PRNGKey(3)), data=data)
    ).get_trace(context, include_components=False)


def _v(tr, name):
    return float(np.asarray(tr[name]["value"]))


BASE_DATA = {
    "gal_v_kms": np.array(0.0),
    "gal_sigma_kms": np.array(120.0),
    "log_ebv_gal": np.array(np.log(1.0e-12)),
    "dust_alpha": np.array(2.0),
    "log_systematics_width": np.array(np.log(0.12)),
}


def test_reference_sites_match_fixed_variance_definition(_fake_ssp):
    cfg = _cfg(reference_spectrum_floor=0.05)
    context = build_model_context(cfg)
    tr = _trace(context, {**BASE_DATA, "frac_jitter": np.array(0.03)})
    assert set(REF_SITES) <= set(tr)
    for name in REF_SITES:
        assert tr[name]["type"] == "deterministic"

    pred = np.asarray(tr["pred_spectrum_fluxes"]["value"])
    obs = np.asarray(cfg.spectroscopy.fluxes)
    err = np.asarray(cfg.spectroscopy.errors)
    expected_spec = np.sum((obs - pred) ** 2 / (err**2 + (0.05 * pred) ** 2))
    assert _v(tr, "spectroscopy_n_ref") == 4.0
    assert _v(tr, "spectroscopy_chi2_ref") == pytest.approx(expected_spec, rel=1e-10)
    assert _v(tr, "spectroscopy_reduced_chi2_ref") == pytest.approx(expected_spec / 4.0, rel=1e-10)
    assert _v(tr, "joint_n_ref") == pytest.approx(_v(tr, "sed_n_eff") + 4.0)
    assert _v(tr, "joint_chi2_ref") == pytest.approx(
        _v(tr, "sed_chi2_ref") + _v(tr, "spectroscopy_chi2_ref")
    )
    assert _v(tr, "joint_reduced_chi2_ref") == pytest.approx(
        _v(tr, "joint_chi2_ref") / _v(tr, "joint_n_ref")
    )


def test_reference_chi2_is_invariant_to_fitted_noise(_fake_ssp):
    context = build_model_context(_cfg())
    low = _trace(context, {**BASE_DATA, "frac_jitter": np.array(0.001)})
    high = _trace(
        context,
        {**BASE_DATA, "frac_jitter": np.array(0.3), "log_systematics_width": np.array(np.log(0.08))},
    )
    # The fitted-noise chi-square moves with the jitter and the systematic width ...
    assert _v(low, "spectroscopy_chi2") > 2.0 * _v(high, "spectroscopy_chi2")
    assert _v(low, "sed_chi2") != pytest.approx(_v(high, "sed_chi2"))
    # ... the reference chi-square does not.
    for name in REF_SITES:
        assert _v(low, name) == pytest.approx(_v(high, name), rel=1e-12)


def test_reference_floors_do_not_change_likelihood_or_existing_sites(_fake_ssp):
    data = {**BASE_DATA, "frac_jitter": np.array(0.02)}
    tr_a = _trace(build_model_context(_cfg()), data)
    tr_b = _trace(
        build_model_context(
            _cfg(reference_spectrum_floor=0.5, reference_photometry_floor=0.4, reference_agn_floor=0.9)
        ),
        data,
    )
    assert set(tr_a) == set(tr_b)
    assert set(LEGACY_SITES) <= set(tr_a)
    # Every site other than the reference diagnostics is bit-for-bit identical,
    # including the existing chi-square sites and all likelihood terms.
    for name, site in tr_a.items():
        if name in REF_SITES or site["type"] not in ("sample", "deterministic"):
            continue
        np.testing.assert_array_equal(
            np.asarray(site["value"]), np.asarray(tr_b[name]["value"]), err_msg=name
        )
        if site["type"] == "sample":
            np.testing.assert_array_equal(
                np.asarray(site["fn"].log_prob(site["value"])),
                np.asarray(tr_b[name]["fn"].log_prob(tr_b[name]["value"])),
                err_msg=name,
            )
    assert _v(tr_a, "spectroscopy_chi2_ref") != pytest.approx(_v(tr_b, "spectroscopy_chi2_ref"))


def _agn_scale_cfg(
    phot_fluxes=(1.0, 1.2, 0.9),
    phot_errors=(0.1, 0.1, 0.05),
    fit_host=True,
    spec_fluxes=None,
    spec_errors=None,
    **likelihood_kwargs,
):
    rng = np.random.default_rng(0)
    wave = np.linspace(5000.0, 5040.0, 41)
    return FitConfig(
        observation=Observation(object_id="obj", redshift=0.1),
        photometry=PhotometryData(
            filter_names=["f1", "f2", "f3"], fluxes=list(phot_fluxes), errors=list(phot_errors)
        ),
        filters=FilterSet(
            curves=[
                FilterCurve(name="f1", wave=[1000.0, 2000.0, 3000.0], transmission=[0.0, 1.0, 0.0]),
                FilterCurve(name="f2", wave=[3000.0, 4000.0, 5000.0], transmission=[0.0, 1.0, 0.0]),
                FilterCurve(name="f3", wave=[5000.0, 6000.0, 7000.0], transmission=[0.0, 1.0, 0.0]),
            ]
        ),
        galaxy=GalaxyConfig(
            dsps_ssp_fn="fake_refchi2_agn.h5", n_wave=16, fit_host=fit_host, fit_host_kinematics=fit_host
        ),
        agn=AGNConfig(fit_agn=True, fit_lines=False),
        nebular=NebularConfig(enabled=False),
        likelihood=LikelihoodConfig(
            variability_uncertainty=False, fit_spectrum_scale=True, **likelihood_kwargs
        ),
        spectroscopy=SpectroscopyData(
            wave_obs=wave.tolist(),
            fluxes=list(spec_fluxes) if spec_fluxes is not None else (2.0 + 0.3 * rng.standard_normal(41)).tolist(),
            errors=list(spec_errors) if spec_errors is not None else (0.1 + 0.05 * rng.random(41)).tolist(),
            instrument="sdss",
            resolving_power=2000.0,
        ),
        inference=InferenceConfig(map_steps=2),
        prior_config=PriorConfig(stellar_mass=dist.Normal(8.0, 1.0e-6)),
    )


def _agn_latents(context, **overrides):
    tr = trace(seed(grahsp_photometric_model, jax.random.PRNGKey(7))).get_trace(
        context, include_components=False
    )
    latents = {
        k: np.asarray(v["value"])
        for k, v in tr.items()
        if v["type"] == "sample" and not v.get("is_observed", False)
    }
    latents.update({k: np.asarray(v) for k, v in overrides.items()})
    return latents


def test_reference_sed_matches_independent_calculation_with_agn(_fake_ssp):
    """Fixed floors multiply the total and the AGN band flux; fitted widths do not enter."""
    # AGN-only model, with photometry placed near the model so that every
    # variance term matters.
    probe_context = build_model_context(_agn_scale_cfg(fit_host=False))
    latents = _agn_latents(
        probe_context,
        log_agn_systematics_width=np.log(0.4),
        log_systematics_width=np.log(0.12),
    )
    probe = np.asarray(_trace(probe_context, latents)["pred_fluxes"]["value"])
    cfg = _agn_scale_cfg(
        phot_fluxes=probe * np.array([1.3, 0.8, 1.1]),
        phot_errors=0.05 * probe,
        fit_host=False,
    )
    context = build_model_context(cfg)
    tr = _trace(context, latents)
    pred = np.asarray(tr["pred_fluxes"]["value"])
    np.testing.assert_allclose(pred, probe, rtol=1e-12)
    agn = np.asarray(tr["variable_agn_fluxes"]["value"])
    assert np.all(np.abs(agn) > 0.0), "test needs a non-zero AGN band flux"
    obs = np.asarray(cfg.photometry.fluxes)
    err = np.asarray(cfg.photometry.errors)
    variance = err**2 + (0.10 * pred) ** 2 + (0.15 * agn) ** 2
    expected = np.sum((obs - pred) ** 2 / variance)
    assert _v(tr, "sed_chi2_ref") == pytest.approx(expected, rel=1e-10)
    assert _v(tr, "sed_reduced_chi2_ref") == pytest.approx(expected / 3.0, rel=1e-10)
    fitted = err**2 + (0.12 * pred) ** 2 + (0.4 * agn) ** 2
    assert _v(tr, "sed_chi2") == pytest.approx(np.sum((obs - pred) ** 2 / fitted), rel=1e-10)
    # Changing only the fitted AGN width leaves the reference unchanged.
    other = _trace(context, {**latents, "log_agn_systematics_width": np.log(0.05)})
    assert _v(other, "sed_chi2_ref") == pytest.approx(_v(tr, "sed_chi2_ref"), rel=1e-12)
    assert _v(other, "sed_chi2") != pytest.approx(_v(tr, "sed_chi2"), rel=1e-6)


def test_reference_spectrum_uses_scaled_model_and_raw_errors(_fake_ssp, monkeypatch):
    """The reference uses the scaled spectral model and the quoted errors only."""
    import jaxsedfit.model as model_module

    calls = []
    original = model_module.reference_spectroscopic_chi2

    def _capture(pred, obs, err, mask, floor):
        calls.append(
            {"pred": np.asarray(pred), "obs": np.asarray(obs), "err": np.asarray(err),
             "mask": np.asarray(mask), "floor": float(floor)}
        )
        return original(pred, obs, err, mask, floor)

    monkeypatch.setattr(model_module, "reference_spectroscopic_chi2", _capture)
    probe_context = build_model_context(_agn_scale_cfg(reference_spectrum_floor=0.06))
    base = _agn_latents(probe_context, log_spectrum_scale=np.asarray(0.0))
    probe = np.asarray(_trace(probe_context, base)["pred_spectrum_fluxes"]["value"])
    noise = 1.0 + 0.05 * np.random.default_rng(1).standard_normal(probe.size)
    cfg = _agn_scale_cfg(
        reference_spectrum_floor=0.06, spec_fluxes=1.6 * probe * noise, spec_errors=0.05 * probe
    )
    context = build_model_context(cfg)
    calls.clear()
    unit = _trace(context, {**base, "log_spectrum_scale": np.asarray(0.0)})
    scaled = _trace(
        context,
        {**base, "log_spectrum_scale": np.asarray(np.log(2.0)), "frac_jitter_std": np.asarray(3.0)},
    )
    assert len(calls) == 2
    for call in calls:
        np.testing.assert_array_equal(call["err"], np.asarray(cfg.spectroscopy.errors))
        np.testing.assert_array_equal(call["obs"], np.asarray(cfg.spectroscopy.fluxes))
        assert call["floor"] == pytest.approx(0.06)
    np.testing.assert_allclose(calls[1]["pred"], 2.0 * calls[0]["pred"], rtol=1e-12)
    pred = np.asarray(scaled["pred_spectrum_fluxes"]["value"])
    np.testing.assert_allclose(pred, calls[1]["pred"], rtol=1e-12)
    obs = np.asarray(cfg.spectroscopy.fluxes)
    err = np.asarray(cfg.spectroscopy.errors)
    expected = np.sum((obs - pred) ** 2 / (err**2 + (0.06 * pred) ** 2))
    assert _v(scaled, "spectroscopy_chi2_ref") == pytest.approx(expected, rel=1e-10)
    assert _v(unit, "spectroscopy_chi2_ref") != pytest.approx(expected, rel=1e-6)


def test_reference_spectrum_ignores_every_jitter_term(_fake_ssp):
    """add_jitter, frac_fe_jitter and frac_jitter do not enter the reference."""
    cfg = _agn_scale_cfg()
    context = build_model_context(cfg)
    base = _agn_latents(context)
    ref = _trace(context, base)
    for extra in ({"frac_jitter_std": np.asarray(4.0)},):
        tr = _trace(context, {**base, **extra})
        assert _v(tr, "spectroscopy_chi2") != pytest.approx(_v(ref, "spectroscopy_chi2"), rel=1e-6)
        assert _v(tr, "spectroscopy_chi2_ref") == pytest.approx(_v(ref, "spectroscopy_chi2_ref"), rel=1e-12)
    # The total jitter variance (frac, Fe II and additive terms) is visible as
    # spectrum_sigma_tot; the reference equals the formula with the quoted
    # errors, so none of those terms can leak in.
    sigma_tot = np.asarray(ref["spectrum_sigma_tot"]["value"])
    err = np.asarray(cfg.spectroscopy.errors)
    assert np.all(sigma_tot >= err)
    pred = np.asarray(ref["pred_spectrum_fluxes"]["value"])
    obs = np.asarray(cfg.spectroscopy.fluxes)
    expected = np.sum((obs - pred) ** 2 / (err**2 + (0.08 * pred) ** 2))
    assert _v(ref, "spectroscopy_chi2_ref") == pytest.approx(expected, rel=1e-10)
