import csv
from pathlib import Path

import jax
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.util import log_density

from jaxsedfit.batched import BatchedContext, fit_batch
from jaxsedfit.benchmark import build_chimera_fit_config
from jaxsedfit.model import grahsp_photometric_model
from jaxsedfit.preload import build_model_context

SAMPLE = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "chimera_validation_sample.csv"


def _patch_ssp(monkeypatch):
    class _SSPData:
        ssp_lgmet = np.array([-2.0, -1.0, -0.3, 0.0])
        ssp_lg_age_gyr = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        ssp_wave = np.geomspace(100.0, 1.0e6, 64)
        ssp_flux = np.ones((4, 5, 64))

    monkeypatch.setattr("jaxsedfit.preload._load_ssp_templates", lambda fn: _SSPData())
    monkeypatch.setattr("jaxsedfit.preload._SSP_DATA_CACHE", {})
    monkeypatch.setattr("jaxsedfit.preload._HOST_BASIS_CACHE", {})


def _chimera_configs(tmp_path, monkeypatch, n=3):
    _patch_ssp(monkeypatch)
    ssp_path = tmp_path / "fake.h5"
    ssp_path.write_bytes(b"")
    with SAMPLE.open() as fh:
        rows = list(csv.DictReader(fh))
    rows = sorted(rows, key=lambda r: float(r["redshift"]))
    picked = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, n)]  # span the redshift range
    configs = []
    for row in picked:
        row = {k: (v if k in ("id", "ID_COSMOS") else float(v)) for k, v in row.items()}
        cfg = build_chimera_fit_config(row, dsps_ssp_fn=str(ssp_path))
        cfg.prior_config.stellar_mass = dist.Uniform(6.0, 12.0)
        cfg.inference.map_steps = 6
        cfg.inference.svi_steps = 4
        cfg.inference.num_samples = 8
        configs.append(cfg)
    return configs


def _chimera_contexts(tmp_path, monkeypatch, n=3):
    return [build_model_context(cfg) for cfg in _chimera_configs(tmp_path, monkeypatch, n)]


def test_batched_log_density_matches_per_object_model(tmp_path, monkeypatch):
    contexts = _chimera_contexts(tmp_path, monkeypatch)
    bctx = BatchedContext(contexts[:2])
    arrays, redshift = bctx.stack(contexts)
    batched_model = bctx.model()
    batched_lp = jax.jit(lambda params, a, z: log_density(batched_model, (a, z), {}, params)[0])

    for i, ctx in enumerate(contexts):
        def model(ctx=ctx):
            return grahsp_photometric_model(ctx, include_components=False)

        tr = trace(seed(model, i)).get_trace()
        params = {k: s["value"] for k, s in tr.items() if s["type"] == "sample" and not s["is_observed"]}
        expected = float(log_density(model, (), {}, params)[0])
        got = float(batched_lp(params, [a[i] for a in arrays], redshift[i]))
        assert np.isfinite(expected)
        assert got == np.float64(expected) or abs(got - expected) < 1e-8 * max(1.0, abs(expected))


@pytest.mark.parametrize("method", ["map", "laplace", "svi"])
def test_fit_batch_returns_per_object_results(tmp_path, monkeypatch, method):
    configs = _chimera_configs(tmp_path, monkeypatch, n=3)
    result = fit_batch(configs, method=method, chunk_size=2, progress_bar=False)

    n_draws = 1 if method == "map" else 8
    assert len(result) == 3
    assert result.object_ids == [cfg.observation.object_id for cfg in configs]
    for samples in result.samples:
        logm = np.asarray(samples["log_stellar_mass"])
        assert logm.shape == (n_draws,)
        assert np.all((logm >= 6.0) & (logm <= 12.0))
    assert "log_stellar_mass_p50" in result.summary_table().colnames
    single = result[1]
    assert single.method == method
    assert np.asarray(single.samples["log_stellar_mass"]).shape == (n_draws,)


def test_fit_batch_does_not_depend_on_chunking(tmp_path, monkeypatch):
    configs = _chimera_configs(tmp_path, monkeypatch, n=3)
    a = fit_batch(configs, method="map", chunk_size=1, progress_bar=False)
    b = fit_batch(configs, method="map", chunk_size=3, progress_bar=False)
    for sa, sb in zip(a.samples, b.samples):
        np.testing.assert_allclose(sa["log_stellar_mass"], sb["log_stellar_mass"], rtol=1e-6)


def test_fit_batch_groups_objects_with_different_priors(tmp_path, monkeypatch):
    configs = _chimera_configs(tmp_path, monkeypatch, n=3)
    configs[2].prior_config.stellar_mass = dist.Uniform(9.0, 12.0)
    result = fit_batch(configs, method="map", chunk_size=2, progress_bar=False)
    assert float(result.samples[2]["log_stellar_mass"][0]) >= 9.0


def test_fit_batch_derived_sites(tmp_path, monkeypatch):
    configs = _chimera_configs(tmp_path, monkeypatch, n=3)
    sites = ["log_sfr_fit", "log_agn_bol_luminosity_fit", "log_agn_bol_luminosity_attenuated_fit"]
    result = fit_batch(configs, method="laplace", chunk_size=2, progress_bar=False, derived_sites=sites)
    for i, (samples, map_values) in enumerate(zip(result.samples, result.map_values)):
        for site in sites:
            assert np.asarray(samples[site]).shape == (8,)
            assert np.all(np.isfinite(samples[site]))
        # Derived values at the MAP match the per-object model evaluated there.
        ctx = build_model_context(configs[i])
        params = {k: v for k, v in map_values.items() if k not in sites}
        model = seed(lambda: grahsp_photometric_model(ctx, include_components=False), 0)
        sub = trace(substitute(model, data=params)).get_trace()
        for site in sites:
            np.testing.assert_allclose(map_values[site], sub[site]["value"], rtol=1e-6, atol=1e-8)
        # The attenuated luminosities are the dereddened ones times the AGN transmission at 5100 A.
        transmission = float(sub["agn_transmission_5100_fit"]["value"])
        assert 0.0 < transmission <= 1.0
        np.testing.assert_allclose(sub["log_agn_bol_luminosity_dereddened_fit"]["value"],
                                   sub["log_agn_bol_luminosity_fit"]["value"])
        for kind in ("l5100", "bol_luminosity"):
            np.testing.assert_allclose(
                sub[f"log_agn_{kind}_attenuated_fit"]["value"],
                sub[f"log_agn_{kind}_dereddened_fit"]["value"] + np.log10(transmission), rtol=1e-10)
    assert "log_sfr_fit_p50" in result.summary_table().colnames


def _spectral_configs(tmp_path, monkeypatch, photometry=True, n=3):
    from jaxsedfit.config import SpectroscopyData

    configs = _chimera_configs(tmp_path, monkeypatch, n)
    rng = np.random.default_rng(1)
    for k, cfg in enumerate(configs):
        n_pix = 500 + 40 * k  # different lengths, padded to a common one
        wave = 3800.0 * np.exp(np.arange(n_pix) * 1.0e-4 * np.log(10.0) * 8)
        flux = 0.05 * (wave / 5000.0) ** -1 * (1.0 + 0.05 * rng.normal(size=n_pix))
        mask = np.ones(n_pix, dtype=bool)
        mask[50:60] = False
        cfg.spectroscopy = SpectroscopyData(
            wave_obs=wave, fluxes=flux, errors=np.full(n_pix, 0.005), mask=mask, instrument="sdss",
            aperture_diameter_arcsec=3.0, resolving_power=[1800.0, 2000.0, 2500.0][k % 3],
        )
        cfg.agn.fit_lines = True
        cfg.agn.fit_feii = True
        cfg.likelihood.fit_spectrum_scale = True
        if not photometry:
            cfg.photometry = None
    return configs


@pytest.mark.parametrize("photometry", [True, False], ids=["joint", "spectrum_only"])
def test_batched_spectral_log_density_matches_per_object_model(tmp_path, monkeypatch, photometry):
    from jaxsedfit.batched import _share_line_tables

    configs = _share_line_tables(_spectral_configs(tmp_path, monkeypatch, photometry), None)
    pad = [max(len(cfg.spectroscopy.wave_obs) for cfg in configs)]
    contexts = [build_model_context(cfg, spec_pad_to=pad) for cfg in configs]
    bctx = BatchedContext(contexts[:2])
    arrays, redshift = bctx.stack(contexts)
    batched_model = bctx.model()
    batched_lp = jax.jit(lambda params, a, z: log_density(batched_model, (a, z), {}, params)[0])

    for i, (cfg, ctx) in enumerate(zip(configs, contexts)):
        def model(ctx=ctx):
            return grahsp_photometric_model(ctx, include_components=False)

        tr = trace(seed(model, i)).get_trace()
        params = {k: s["value"] for k, s in tr.items() if s["type"] == "sample" and not s["is_observed"]}
        expected = float(log_density(model, (), {}, params)[0])
        got = float(batched_lp(params, [a[i] for a in arrays], redshift[i]))
        assert np.isfinite(expected)
        assert abs(got - expected) < 1e-10 * max(1.0, abs(expected))

        # Padding with masked pixels leaves the model unchanged.
        unpadded = build_model_context(cfg)
        lp = float(log_density(lambda: grahsp_photometric_model(unpadded, include_components=False), (), {}, params)[0])
        assert abs(lp - expected) < 1e-10 * max(1.0, abs(expected))


def test_share_line_tables_groups_by_redshift_window(tmp_path, monkeypatch):
    from jaxsedfit.batched import _share_line_tables

    configs = _spectral_configs(tmp_path, monkeypatch)
    z = [cfg.observation.redshift for cfg in configs]
    tables = [cfg.agn.line_activation_rest for cfg in _share_line_tables(configs, 0.5)]
    assert all(cfg.agn.line_activation_rest is None for cfg in configs)  # inputs untouched
    lnz = np.log1p(z)
    for i in range(3):
        for j in range(3):
            if abs(lnz[i] - lnz[j]) > 0.5:
                assert tables[i] != tables[j]
    single = _share_line_tables(configs, None)
    assert len({cfg.agn.line_activation_rest for cfg in single}) == 1
    lo, hi = single[0].agn.line_activation_rest
    assert lo < 3800.0 / (1.0 + max(z)) and hi > 3800.0 / (1.0 + min(z))


def test_tied_line_coverage_mask_matches_coverage_table():
    from jaxsedfit.spectroscopy import SpectralComponentConfig, build_joint_tied_line_meta
    from jaxsedfit.spectral_model import tied_line_coverage_mask

    wide = build_joint_tied_line_meta(SpectralComponentConfig(line_coverage_rest=(1000.0, 9000.0)))
    for coverage in [(1200.0, 2000.0), (3500.0, 5200.0), (4700.0, 7000.0)]:
        narrow = build_joint_tied_line_meta(SpectralComponentConfig(line_coverage_rest=coverage))
        active = np.asarray(tied_line_coverage_mask(wide, coverage)) > 0
        assert [n for n, a in zip(wide["names"], active) if a] == narrow["names"]


def test_fit_batch_fits_spectra(tmp_path, monkeypatch):
    # Spectrum-only without Fe II and staging compiles fastest; the joint model is
    # covered by the log-density test above.
    configs = _spectral_configs(tmp_path, monkeypatch, photometry=False, n=2)
    for cfg in configs:
        cfg.agn.fit_feii = False
        cfg.inference.staged_map = False
    result = fit_batch(configs, method="map", chunk_size=2, progress_bar=False, redshift_window=None)
    assert len(result) == 2
    table = result.configs[0].agn.line_activation_rest
    assert table is not None and all(cfg.agn.line_activation_rest == table for cfg in result.configs)
    for samples, diag in zip(result.samples, result.diagnostics):
        assert np.isfinite(diag["loss"])
        assert np.all(np.isfinite(np.asarray(samples["log_agn_amp"])))
    assert result[0].predict().spectrum.lines


def test_qso_model_pixel_mask_ignores_padding():
    from jaxsedfit.spectral_model import FSPSTemplateGrid, qso_fsps_joint_model

    rng = np.random.default_rng(3)
    wave = np.geomspace(3000.0, 6000.0, 300)
    flux = 10.0 * (wave / 4000.0) ** -1.5 + rng.normal(scale=0.3, size=wave.size)
    err = np.full(wave.size, 0.3)
    pad = 40
    wave_p = np.concatenate([wave, wave[-1] * np.exp(np.log(wave[1] / wave[0]) * np.arange(1, pad + 1))])
    flux_p = np.concatenate([flux, np.zeros(pad)])
    err_p = np.concatenate([err, np.full(pad, 1.0e30)])

    def model(w, f, e, mask=None):
        grid = FSPSTemplateGrid(wave=w, templates=np.zeros((w.size, 1)), template_meta=[{}],
                                age_grid_gyr=np.ones(1), logzsol_grid=np.zeros(1))
        return qso_fsps_joint_model(
            w, f, e, {}, {"n_lines": 0, "n_fgroups": 0}, grid, wave, np.zeros_like(wave), wave, np.zeros_like(wave),
            use_lines=False, prior_config={"PL_pivot": 4000.0}, decompose_host=False, fit_fe=False, fit_bc=False,
            pixel_mask=mask,
        )

    tr = trace(seed(lambda: model(wave, flux, err), 0)).get_trace()
    params = {k: s["value"] for k, s in tr.items() if s["type"] == "sample" and not s["is_observed"]}
    expected = float(log_density(lambda: model(wave, flux, err), (), {}, params)[0])
    padded = float(log_density(lambda: model(wave_p, flux_p, err_p, np.arange(wave_p.size) < wave.size), (), {}, params)[0])
    assert np.isfinite(expected)
    assert abs(padded - expected) < 1e-10 * max(1.0, abs(expected))
