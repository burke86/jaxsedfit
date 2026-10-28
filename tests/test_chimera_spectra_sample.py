import csv
from pathlib import Path

import numpy as np
from astropy.table import Table

DATA = Path(__file__).resolve().parents[1] / "benchmarks" / "data"


def test_chimera_spectra_sample_matches_photometry_sample():
    spectra = Table.read(DATA / "chimera_validation_spectra.fits")
    with (DATA / "chimera_validation_sample.csv").open() as fh:
        photometry = {row["id"]: row for row in csv.DictReader(fh)}

    assert len(spectra) > 0
    assert spectra.meta["SPEC_RES"] > 0
    for row in spectra:
        phot = photometry[row["id"]]
        assert np.isclose(row["redshift"], float(phot["redshift"]))
        assert np.isclose(row["chimera_QSO_weight"], float(phot["chimera_QSO_weight"]))
        wave, flux = np.asarray(row["wave_obs"]), np.asarray(row["flux"])
        assert wave.shape == flux.shape == np.shape(row["flux_err"])
        assert np.all(np.diff(wave) > 0)
        ok = np.isfinite(flux)
        assert ok.mean() > 0.5  # LEGA-C hosts have masked pixels, so not all are finite
        assert np.all(~np.isfinite(np.asarray(row["flux_err"])[~ok]))
        np.testing.assert_allclose(flux[ok], np.asarray(row["flux_gal"])[ok] + np.asarray(row["flux_qso"])[ok],
                                   rtol=1e-5, atol=1e-9)


def test_joint_validation_attaches_each_objects_spectrum(tmp_path):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "chimera_validation.py"
    spec = importlib.util.spec_from_file_location("chimera_validation", path)
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)

    spectra = cv.load_spectra(DATA / "chimera_validation_spectra.fits")
    rows = [r for r in cv.load_sample(DATA / "chimera_validation_sample.csv") if r["id"] in spectra][:2]
    for s in spectra.values():
        assert np.all(np.isfinite(s["flux"])) and np.all(s["err"] > 0)  # masked pixels are finite placeholders
    ssp = tmp_path / "fake.h5"
    ssp.write_bytes(b"")
    for row, cfg in zip(rows, cv._configs(rows, str(ssp), 10, 5e-3, spectra)):
        spec = cfg.spectroscopy_list[0]
        np.testing.assert_array_equal(spec.mask, spectra[row["id"]]["mask"])
        assert spec.resolving_power == spectra[row["id"]]["resolving_power"]
        assert cfg.agn.fit_lines and cfg.agn.fit_feii and cfg.likelihood.fit_spectrum_scale is False
    assert all(not cfg.spectroscopy_list for cfg in cv._configs(rows, str(ssp), 10, 5e-3))
