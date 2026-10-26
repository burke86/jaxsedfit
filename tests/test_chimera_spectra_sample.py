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
