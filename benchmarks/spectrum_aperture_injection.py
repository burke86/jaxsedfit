"""Injection tests of spectral aperture systematics on Chimera joint fits.

Each scenario perturbs the Chimera spectra (photometry unchanged, total flux) and
fits them with the proposed joint default: gray spectrum scale plus the per-spectrum
AGN variability term. Scenarios:

- clean: original spectra.
- psf_blue_excess: quasar part times (lambda_obs/5400)^-0.34,
  +15% at 3600 A (BOSS quasar focal-plane offset residual, Margala+2016 scale).
- psf_blue_loss: quasar part times (lambda/5400)^+0.34 (seeing loss of a point source
  relative to the extended host, 1.5" FWHM in a 2" fiber gives about this slope).
- psf_blue_excess_tilt: as psf_blue_excess, fit with fit_spectrum_tilt.
- host_gradient: host part times (lambda/5400)^+0.33, a fiber that sees a center
  0.1 mag redder in g-r than the whole galaxy.
- fiber_capture: host part times the model's own capture fraction for a 3" fiber,
  1.5" seeing, point-source calibration and a 2" host scale (0.33); quasar part
  (including the narrow lines) unchanged. Fit with the
  aperture and seeing given, the photometry labeled total ("model") and the host
  capture model on.
- fiber_capture_default: the same spectra fit with the host capture model off
  (the default before it was turned on), so the aperture is ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))  # noqa: E402

import chimera_validation as cv  # noqa: E402
import spectrum_agn_variability_injection as inj  # noqa: E402

SCENARIOS = ("clean", "psf_blue_excess", "psf_blue_loss", "psf_blue_excess_tilt", "host_gradient", "fiber_capture", "fiber_capture_default")
FIBER_DIAMETER, SEEING_FWHM, HOST_SCALE_TRUE = 3.0, 1.5, 2.0


def _spectra(table, rows, scenario):
    by_id = {str(t["id"]): t for t in table}
    res = float(table.meta["SPEC_RES"])
    out = {}
    for r in rows:
        t = by_id[r["id"]]
        wave = np.asarray(t["wave_obs"], dtype=float)
        qso = np.nan_to_num(np.asarray(t["flux_qso"], dtype=float))
        gal = np.nan_to_num(np.asarray(t["flux_gal"], dtype=float))
        flux = np.asarray(t["flux"], dtype=float)
        err = np.asarray(t["flux_err"], dtype=float)
        good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        x = wave / 5400.0
        if scenario.startswith("psf_blue_excess"):
            flux = flux + (x**-0.34 - 1.0) * qso
        elif scenario == "psf_blue_loss":
            flux = flux + (x**0.34 - 1.0) * qso
        elif scenario == "host_gradient":
            flux = flux + (x**0.33 - 1.0) * gal
        elif scenario.startswith("fiber_capture"):
            from jaxsedfit.preload import SPEC_CAPTURE_LOG_R_GRID, spectral_aperture_capture_tables

            table, _ = spectral_aperture_capture_tables([FIBER_DIAMETER], [SEEING_FWHM], ["point_source"])
            f = float(np.interp(np.log(HOST_SCALE_TRUE), SPEC_CAPTURE_LOG_R_GRID, table[0]))
            flux = flux + (f - 1.0) * gal
        out[r["id"]] = dict(wave=wave, flux=np.where(good, flux, 0.0), err=np.where(good, err, 1.0), mask=good,
                            resolving_power=res)
    return out


def _fit(rows, spectra, scenario, args):
    from jaxsedfit import fit_batch

    configs = cv._configs(rows, args.dsps_ssp_fn, args.steps, args.learning_rate, spectra)
    for cfg in configs:
        cfg.likelihood.fit_spectrum_scale = True
        cfg.likelihood.spectrum_scale_prior_sigma_dex = args.scale_prior_dex
        cfg.likelihood.fit_spectrum_agn_variability = True
        cfg.agn.fit_spectrum_tilt = scenario.endswith("_tilt")
        cfg.likelihood.use_host_capture_model = scenario != "fiber_capture_default"
        if scenario.startswith("fiber_capture"):
            cfg.spectroscopy.aperture_diameter_arcsec = FIBER_DIAMETER
            cfg.spectroscopy.psf_fwhm_arcsec = SEEING_FWHM
            cfg.photometry.photometry_method = ["model"] * len(cfg.photometry.filter_names)
    derived = ["log_agn_bol_luminosity_fit", "spectrum_agn_log_amplitude", "spectrum_host_capture_fraction"]
    t0 = time.time()
    batch = fit_batch(configs, method="laplace", chunk_size=args.chunk, derived_sites=derived)
    out = []
    for row, mv in zip(rows, batch.map_values):
        rec = {"id": row["id"], "log_stellar_mass_truth": row["log_stellar_mass_truth"],
               "log_lbol_truth": row["log_lbol_chimera_truth"],
               "log_stellar_mass_fit": float(mv["log_stellar_mass"]),
               "log_lbol_fit": float(mv["log_agn_bol_luminosity_fit"]) + 7.0}
        for key, val in mv.items():
            if key in ("log_spectrum_scale", "spectrum_agn_log_amplitude", "agn_variability_index",
                       "spectrum_host_capture_fraction", "log_host_capture_scale_arcsec") or "tilt" in key:
                rec[key] = float(np.ravel(val)[0])
        out.append(rec)
    print(f"  {scenario}: {len(rows)} fits in {time.time() - t0:.0f} s", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsps-ssp-fn", required=True)
    p.add_argument("--n-objects", type=int, default=20)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p.add_argument("--scale-prior-dex", type=float, default=0.1, help="spectrum_scale_prior_sigma_dex.")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)

    from astropy.table import Table

    table = Table.read(cv.DEFAULT_SPECTRA)
    rows, ratio = inj._select(cv.load_sample(Path(cv.DEFAULT_SAMPLE)), table, args.n_objects, 0.2, 5.0)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results["ratio"] = {r["id"]: ratio[r["id"]] for r in rows}
    for scenario in args.scenarios:
        if scenario in results:
            continue
        results[scenario] = _fit(rows, _spectra(table, rows, scenario), scenario, args)
        path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
