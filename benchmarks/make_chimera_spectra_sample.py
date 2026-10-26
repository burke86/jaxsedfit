"""Build the Chimera spectra fixture for the validation sample.

Collects the composite spectra (host galaxy + weighted quasar) of the objects in
``data/chimera_validation_sample.csv`` that have one, and writes them to
``data/chimera_validation_spectra.fits``: one row per Chimera, keyed by ``id``
like the photometry sample, with the spectra as variable-length float32 arrays.
The spectra come from the Chimeras ``make spectra`` target (burke86/chimeras,
branch ``agent/chimera-spectra``); only a Chimera whose host was observed by DESI
DR1 or LEGA-C has one.

Columns
-------
id, redshift, chimera_QSO_weight
    As in the photometry sample.
galaxy_survey, galaxy_snr_median, galaxy_scale, qso_scale
    Host spectrum source and median S/N, and the factors that scale the host and
    quasar spectra to their own photometry.
wave_obs
    Observed-frame vacuum wavelength [Angstrom], log-spaced at the resolving power
    in the ``SPEC_RES`` header keyword.
flux, flux_err, flux_gal, flux_qso
    Total, uncertainty, host and weighted-quasar flux density [mJy]
    (``flux = flux_gal + flux_qso``); ``flux_err`` is infinite where either
    component is missing.

Usage
-----
python benchmarks/make_chimera_spectra_sample.py --chimeras-dir <chimeras build directory>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SAMPLE = HERE / "data" / "chimera_validation_sample.csv"
DEFAULT_OUTPUT = HERE / "data" / "chimera_validation_spectra.fits"
ARRAY_COLUMNS = ("wave_obs", "flux", "flux_err", "flux_gal", "flux_qso")
INDEX_COLUMNS = ("redshift", "chimera_QSO_weight", "galaxy_survey", "galaxy_snr_median", "galaxy_scale", "qso_scale")


def main(argv=None) -> int:
    from astropy.io import fits
    from astropy.table import Table

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chimeras-dir", required=True,
                        help="Chimeras build directory with chimeras-spectra.fits and chimeras-spectra/.")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    chimeras_dir = Path(args.chimeras_dir)
    index = Table.read(chimeras_dir / "chimeras-spectra.fits")
    by_id = {str(index["id"][i]): i for i in range(len(index))}
    with Path(args.sample).open() as fh:
        ids = [row["id"] for row in csv.DictReader(fh)]

    rows = []
    for object_id in ids:
        if object_id not in by_id:
            continue
        entry = index[by_id[object_id]]
        spectrum = fits.getdata(chimeras_dir / str(entry["spectrum_file"]), 1)
        row = {"id": object_id, **{k: entry[k] for k in INDEX_COLUMNS}}
        row.update({k: np.asarray(spectrum[k], dtype=np.float32) for k in ARRAY_COLUMNS})
        rows.append(row)

    out = Table(rows=rows, names=["id", *INDEX_COLUMNS, *ARRAY_COLUMNS])
    out["galaxy_survey"] = out["galaxy_survey"].astype(str)
    out["wave_obs"].unit = "Angstrom"
    for name in ARRAY_COLUMNS[1:]:
        out[name].unit = "mJy"
    out.meta["SPEC_RES"] = float(index.meta["SPEC_RES"])
    out.meta["COMMENT"] = ["Chimera composite spectra (host + weighted quasar) for chimera_validation_sample.csv;",
                           "built with benchmarks/make_chimera_spectra_sample.py."]
    out.write(args.output, overwrite=True)
    print(f"Wrote {len(out)} spectra ({len(ids)} objects in the sample) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
