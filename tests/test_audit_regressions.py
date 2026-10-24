from types import SimpleNamespace

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from numpyro.handlers import seed

from jaxsedfit.config import FilterCurve
from jaxsedfit.model import _luminosity_distance_m_jax
from jaxsedfit.preload import _filter_response_cache_key, _prepare_loaded_filter
from jaxsedfit.spectral_components import _evaluate_simple_line_components
from jaxsedfit.spectral_model import (
    _custom_component_eval_wave,
    _custom_fnu_mjy_to_rest_flambda_1e17,
)
from jaxsedfit.spectroscopy import SpectralComponentConfig, make_custom_component


def test_filter_outside_model_grid_is_rejected():
    obs_wave = np.geomspace(1000.0, 1.0e4, 256)
    curve = FilterCurve(name="far", wave=[2.0e4, 2.5e4, 3.0e4], transmission=[0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="not covered"):
        _prepare_loaded_filter(obs_wave, curve)


def test_narrow_filter_interpolation_weights_stay_bounded():
    obs_wave = np.geomspace(1000.0, 1.0e4, 64)
    curve = FilterCurve(name="narrow", wave=[5000.0, 5001.0, 5002.0], transmission=[0.0, 1.0, 0.0])
    loaded = _prepare_loaded_filter(obs_wave, curve)
    assert np.all((loaded.interp_weight >= 0.0) & (loaded.interp_weight <= 1.0))


def test_filter_cache_key_tracks_array_contents_not_identity():
    wave = np.array([4000.0, 5000.0, 6000.0])
    curve = FilterCurve(name="inline", wave=wave, transmission=np.array([0.0, 1.0, 0.0]))
    cfg = SimpleNamespace(
        photometry=SimpleNamespace(filter_names=["inline"]),
        filters=SimpleNamespace(curves=[curve]),
    )
    before = _filter_response_cache_key(cfg)
    wave += 4000.0
    assert _filter_response_cache_key(cfg) != before


@pytest.mark.parametrize("z", [0.001, 0.02, 0.3, 3.0])
def test_luminosity_distance_matches_astropy(z):
    expected = FlatLambdaCDM(H0=70.0, Om0=0.3).luminosity_distance(z).to_value(u.m)
    assert float(_luminosity_distance_m_jax(z, 70.0, 0.3)) == pytest.approx(expected, rel=1.0e-5)


def test_untied_lines_expose_per_component_state_for_photometry():
    wave_rest = np.linspace(4600.0, 5100.0, 256)
    cfg = SpectralComponentConfig(
        tied_lines=False,
        line_centers_rest=(4862.68, 5008.24),
        line_names=("Hb_br", "OIII5007"),
        broad_line_names=("Hb_br",),
    )
    _, _, _, diagnostics = seed(_evaluate_simple_line_components, jax.random.PRNGKey(0))(
        jnp.asarray(wave_rest), jnp.ones_like(jnp.asarray(wave_rest)), cfg, site_prefix="spectral"
    )
    assert np.asarray(diagnostics["line_amp_per_component"]).shape == (2,)
    assert np.allclose(np.asarray(diagnostics["line_broad_mask_per_component"]), [1.0, 0.0])
    assert np.allclose(np.exp(np.asarray(diagnostics["line_mu_per_component"])), [4862.68, 5008.24], rtol=0.02)


def test_standalone_custom_component_honors_fnu_mjy_and_frame():
    z = 0.5
    wave_rest = np.array([3000.0, 5000.0])
    flam = np.asarray(_custom_fnu_mjy_to_rest_flambda_1e17(wave_rest, np.ones(2), z))
    expected_obs = (1.0 * u.mJy).to(
        u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(wave_rest * (1.0 + z) * u.AA)
    ).value / 1.0e-17
    assert np.allclose(flam, expected_obs * (1.0 + z))

    observed = make_custom_component("obs", {}, lambda w, p, m: w, frame="observed")
    rest = make_custom_component("rest", {}, lambda w, p, m: w)
    assert np.allclose(_custom_component_eval_wave(wave_rest, observed, z), wave_rest * (1.0 + z))
    assert np.allclose(_custom_component_eval_wave(wave_rest, rest, z), wave_rest)


def _module_level_component(wave, params, metadata):
    return jnp.zeros_like(jnp.asarray(wave)) + params["amplitude"]


def test_config_with_custom_component_and_one_sided_prior_saves_and_restores(tmp_path):
    import h5py
    import numpyro.distributions as dist

    from jaxsedfit import FitConfig, Observation, SpectroscopyData
    from jaxsedfit.config import AGNConfig, fit_config_from_mapping, serialize_config
    from jaxsedfit.core import JAXSEDFit

    component = make_custom_component(
        "extra",
        {"amplitude": dist.TruncatedNormal(0.0, 1.0, low=0.0)},
        _module_level_component,
    )
    cfg = FitConfig(
        observation=Observation(redshift=0.1),
        spectroscopy=SpectroscopyData(wave_obs=[5000.0, 5001.0], fluxes=[1.0, 1.0], errors=[0.1, 0.1]),
        agn=AGNConfig(custom_components=(component,)),
    )

    state = serialize_config(cfg)
    with h5py.File(tmp_path / "cfg.h5", "w") as handle:
        JAXSEDFit._write_hdf5_node(handle, "config", state)

    restored = fit_config_from_mapping(state).agn.custom_components[0]
    assert restored.name == "extra"
    assert restored.evaluate is _module_level_component
    prior = restored.parameter_priors["amplitude"]
    assert float(prior.low) == 0.0 and not np.isfinite(float(getattr(prior, "high", np.inf)))


def test_inline_feii_template_is_used_without_inline_line_template():
    from jaxsedfit.config import EmissionLineTemplate, FeIITemplate
    from jaxsedfit.preload import _load_templates

    wave = np.linspace(4000.0, 6000.0, 50)
    cfg = SimpleNamespace(
        agn=SimpleNamespace(
            fit_agn=True,
            feii_template=FeIITemplate(wave=wave, lumin=np.ones_like(wave), wavelength_unit="angstrom"),
            emission_line_template=EmissionLineTemplate(),
        ),
        spectroscopy_list=[],
        galaxy=SimpleNamespace(fit_host=False, use_energy_balance=False, dust_alpha=2.0),
    )
    loaded = _load_templates(cfg)
    assert np.allclose(loaded.feii_wave, wave)
    assert loaded.line_wave.size > 1


def test_instrumental_broadening_conserves_ln_lambda_line_flux():
    from jaxsedfit.spectral_components import _split_instrument_broadened_gauss_lnlam
    from jaxsedfit.spectral_model import _split_many_gauss_lnlam

    lnwave = jnp.log(jnp.linspace(4700.0, 5100.0, 4000))
    amps = jnp.array([2.0, 1.0])
    mus = jnp.log(jnp.array([4862.68, 5008.24]))
    sigs = jnp.array([2000.0, 150.0]) / 299792.458
    broad_mask = jnp.array([1.0, 0.0])
    sigma_inst = jnp.full_like(lnwave, 1.0 / (2.354820045 * 2000.0))

    intrinsic, _, intrinsic_narrow, _ = _split_many_gauss_lnlam(lnwave, amps, mus, sigs, broad_mask)
    observed, _, observed_narrow = _split_instrument_broadened_gauss_lnlam(
        lnwave, amps, mus, sigs, broad_mask, sigma_inst
    )

    # f_nu line flux is the integral of f_nu / lambda over ln(lambda).
    def flux(profile):
        return float(jnp.trapezoid(profile * jnp.exp(-lnwave), lnwave))

    assert flux(observed) == pytest.approx(flux(intrinsic), rel=1.0e-6)
    narrow_sigma = float(sigs[1])
    narrow_sigma_eff = float(np.hypot(narrow_sigma, float(sigma_inst[0])))
    expected_peak_ratio = narrow_sigma / narrow_sigma_eff * np.exp(0.5 * (narrow_sigma**2 - narrow_sigma_eff**2))
    assert float(jnp.max(observed_narrow)) / float(jnp.max(intrinsic_narrow)) == pytest.approx(
        expected_peak_ratio, rel=1.0e-3
    )


def _simple_line_joint_spectrum(instrumental_sigma_ln):
    from numpyro.handlers import substitute

    from jaxsedfit.spectroscopy import evaluate_joint_spectral_components

    wave_obs = np.linspace(5150.0, 5350.0, 2000)
    cfg = SpectralComponentConfig(
        tied_lines=False,
        line_centers_rest=(5008.24,),
        line_names=("OIII5007",),
    )
    fixed = {
        "spectral_line_amp_OIII5007": 1.0,
        "spectral_line_fwhm_OIII5007": 200.0,
        "spectral_line_velocity_OIII5007": 0.0,
    }
    model = substitute(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(0)), data=fixed)
    result = model(
        wave_obs,
        0.05,
        np.ones_like(wave_obs),
        config=cfg,
        instrumental_sigma_ln=instrumental_sigma_ln,
    )
    return wave_obs, result


def test_joint_lines_are_convolved_with_resolving_power_only_on_the_spectrum():
    sigma_r2000 = 1.0 / (2.354820045 * 2000.0)
    wave, intrinsic = _simple_line_joint_spectrum(None)
    _, unchanged = _simple_line_joint_spectrum(np.zeros(2000))
    _, observed = _simple_line_joint_spectrum(np.full(2000, sigma_r2000))

    assert np.array_equal(np.asarray(unchanged["lines"]), np.asarray(intrinsic["lines"]))
    line_intrinsic = np.asarray(intrinsic["lines"])
    line_observed = np.asarray(observed["lines"])
    assert np.trapezoid(line_observed, wave) == pytest.approx(np.trapezoid(line_intrinsic, wave), rel=1.0e-4)

    def fwhm_kms(profile):
        center = np.trapezoid(profile * wave, wave) / np.trapezoid(profile, wave)
        variance = np.trapezoid(profile * (wave - center) ** 2, wave) / np.trapezoid(profile, wave)
        return 2.354820045 * np.sqrt(variance) / center * 299792.458

    assert fwhm_kms(line_observed) == pytest.approx(np.hypot(200.0, 299792.458 / 2000.0), rel=0.01)
    # The line state used for photometry and results stays intrinsic.
    assert np.allclose(
        np.asarray(observed["state"]["line_sig_per_component"]),
        np.asarray(intrinsic["state"]["line_sig_per_component"]),
    )


def test_spectral_instrumental_sigma_follows_each_spectrum_resolving_power():
    from jaxsedfit.model import _spectral_instrumental_sigma_ln

    context = SimpleNamespace(
        spec_resolving_power=np.array([2000.0, np.nan]),
        spec_spectrum_index=np.array([0, 0, 1, 1]),
    )
    sigma = _spectral_instrumental_sigma_ln(context)
    assert np.allclose(sigma, [1.0 / (2.354820045 * 2000.0)] * 2 + [0.0, 0.0])

    no_r = SimpleNamespace(spec_resolving_power=np.array([np.nan]), spec_spectrum_index=np.array([0, 0]))
    assert _spectral_instrumental_sigma_ln(no_r) is None


def test_joint_feii_and_balmer_broaden_only_with_resolving_power():
    from numpyro.handlers import substitute

    from jaxsedfit.spectroscopy import evaluate_joint_spectral_components

    wave_obs = np.linspace(3300.0, 5200.0, 3000)
    template_wave = np.linspace(3000.0, 6000.0, 6000)
    template_flux = np.exp(-0.5 * ((template_wave - 4570.0) / 3.0) ** 2)
    cfg = SpectralComponentConfig(use_lines=False, use_feii=True, use_balmer_continuum=True)
    fixed = {
        "spectral_feii_norm": 1.0,
        "spectral_feii_fwhm": 1200.0,
        "spectral_feii_shift": 0.0,
        "spectral_balmer_norm": 1.0,
        "spectral_balmer_tau": 1.0,
        "spectral_balmer_vel": 1500.0,
    }

    def run(sigma):
        model = substitute(seed(evaluate_joint_spectral_components, jax.random.PRNGKey(0)), data=fixed)
        return model(
            wave_obs,
            0.0,
            np.ones_like(wave_obs),
            config=cfg,
            feii_template_wave_rest=template_wave,
            feii_template_flux=template_flux,
            instrumental_sigma_ln=sigma,
        )

    intrinsic = run(None)
    unchanged = run(np.zeros_like(wave_obs))
    observed = run(np.full_like(wave_obs, 1.0 / (2.354820045 * 500.0)))
    for key in ("feii", "balmer"):
        assert np.array_equal(np.asarray(unchanged[key]), np.asarray(intrinsic[key]))
    assert float(np.max(observed["feii"])) < 0.9 * float(np.max(intrinsic["feii"]))


def test_host_kernel_accounts_for_instrument_and_template_resolution():
    from jaxsedfit.spectral_model import host_kernel_sigma_kms

    c_kms = 299792.458
    sigma_inst_r2000 = c_kms / (2.354820045 * 2000.0)
    # At R=2000 and 5000 A, the instrument and 2.5 A MILES resolution cancel almost exactly.
    sigma_template_5000 = c_kms * 2.5 / 2.354820045 / 5000.0
    kernel = float(host_kernel_sigma_kms(100.0, sigma_inst_r2000, 2.5, 5000.0))
    assert kernel == pytest.approx(np.sqrt(100.0**2 + sigma_inst_r2000**2 - sigma_template_5000**2))

    # Unknown resolving power: the fitted dispersion stays an effective broadening.
    assert float(host_kernel_sigma_kms(100.0, 0.0, 2.5, 5000.0)) == pytest.approx(100.0)
    # No template correction requested: plain quadrature with the instrument.
    assert float(host_kernel_sigma_kms(100.0, sigma_inst_r2000, None, 5000.0)) == pytest.approx(
        np.hypot(100.0, sigma_inst_r2000)
    )
    # Templates coarser than the data can resolve: kernel floors near zero, finite gradient.
    floored = host_kernel_sigma_kms(10.0, c_kms / (2.354820045 * 5000.0), 2.5, 5000.0)
    assert 0.0 < float(floored) < 1.0e-2
    grad = jax.grad(lambda s: host_kernel_sigma_kms(s, sigma_inst_r2000, 2.5, 5000.0))(100.0)
    assert np.isfinite(float(grad))


def test_joint_spectral_host_uses_per_spectrum_kernel(monkeypatch):
    from jaxsedfit import model as model_module
    from jaxsedfit.spectral_model import host_kernel_sigma_kms

    calls = []

    def fake_broaden(lnwave, spectrum, v_kms, sigma_kms):
        calls.append(float(sigma_kms))
        return spectrum * float(sigma_kms)

    monkeypatch.setattr(model_module, "_shift_and_broaden_single_spectrum_lnlam", fake_broaden)
    rest_wave = jnp.array([4900.0, 5000.0, 5100.0, 6400.0, 6500.0, 6600.0])
    context = SimpleNamespace(
        spec_resolving_power=np.array([2000.0, np.nan]),
        spec_spectrum_index=np.array([0, 0, 0, 1, 1, 1]),
        fit_config=SimpleNamespace(galaxy=SimpleNamespace(ssp_template_fwhm_angstrom=2.5)),
    )
    out = model_module._broaden_spectral_host(context, rest_wave, jnp.ones(6), 0.0, 120.0)

    expected_first = float(host_kernel_sigma_kms(120.0, 299792.458 / (2.354820045 * 2000.0), 2.5, 5000.0))
    assert calls == pytest.approx([expected_first, 120.0])
    assert np.allclose(np.asarray(out), [expected_first] * 3 + [120.0] * 3)
