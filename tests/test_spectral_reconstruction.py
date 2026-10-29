import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from jaxsedfit import spectral_model as sm


class _SSPData:
    ssp_lgmet = np.array([-2.5, -2.0, -1.7])
    ssp_lg_age_gyr = np.linspace(-2.5, 1.1, 12)
    ssp_wave = np.geomspace(100.0, 1.0e6, 2000)
    # Smooth spectra (no features), so host broadening leaves them unchanged.
    ssp_flux = (
        np.exp(-ssp_lg_age_gyr)[None, :, None]
        * (ssp_wave / 5000.0)[None, None, :] ** -1.5
        * np.ones((3, 1, 1))
        * 1.0e-6
    )


def _patch_ssp(monkeypatch):
    monkeypatch.setattr(sm, "load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._HOST_BASIS_CACHE", {})


def test_reconstruction_rebuilds_the_delayed_sfh_host(monkeypatch):
    _patch_ssp(monkeypatch)
    z = 0.1
    wave = np.linspace(4000.0, 6000.0, 200)
    age_grid, logz_grid = (0.1, 1.0, 10.0), (-0.5, 0.0)
    prior_config = {
        "z_qso": z,
        "host_sfh_model": "delayed",
        "log_host_aperture_scale": dist.Normal(np.log(2.0), 1.0e-3),
    }
    fsps_grid = sm.build_fsps_template_grid(
        sm.extend_loglam_grid(wave, pad_kms=3000.0), age_grid, logz_grid, dsps_ssp_fn="fake.h5", z_qso=z
    )

    # The fitted host for one posterior draw, as the spectral model builds it.
    with handlers.trace() as trace, handlers.seed(rng_seed=1):
        host, _, proxy_weights = sm._delayed_sfh_host_spectrum(fsps_grid, prior_config, np.nan, z)
        aperture_scale = np.exp(sm._sample_log_host_aperture_scale(prior_config))
    fitted_host = np.interp(wave, fsps_grid.wave, np.asarray(aperture_scale * host))
    samples = {
        name: np.asarray(site["value"])[None]
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    samples["gal_sigma_kms"] = np.array([1.0])

    recon = sm.reconstruct_posterior_components(
        wave_out=wave,
        samples=samples,
        pred_out={"fsps_weights": np.asarray(proxy_weights)[None]},
        age_grid_gyr=age_grid,
        logzsol_grid=logz_grid,
        dsps_ssp_fn="fake.h5",
        prior_config=prior_config,
        fit_poly=False,
        fit_poly_order=0,
        fit_reddening=False,
        fe_uv_wave=np.array([1000.0, 2000.0, 3000.0]),
        fe_uv_flux=np.zeros(3),
        fe_op_wave=np.array([4000.0, 5000.0, 6000.0]),
        fe_op_flux=np.zeros(3),
    )

    interior = slice(20, -20)
    np.testing.assert_allclose(recon["draws"]["host"][0][interior], fitted_host[interior], rtol=2e-3)
