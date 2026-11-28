"""Injection test of the AGN spectral-variability term on Chimera joint fits.

The Chimera composite spectra store the host and quasar parts separately. This
script multiplies only the quasar part of each spectrum by a bluer-when-brighter
factor, exp[ln A * (lambda_rest / 5100 A)**(-index)], and keeps the photometry at
the reference state. Each object is fit (batched MAP + Laplace, as in the Chimera
validation) in four modes.

- ``base``: original spectrum, the Chimera validation configuration.
- ``fixed``: injected spectrum, no spectrum scale (the Chimera validation setup).
- ``gray``: injected spectrum, gray per-spectrum scale.
- ``var``: injected spectrum, gray scale plus the AGN variability term.

Example
-------
python benchmarks/spectrum_agn_variability_injection.py --dsps-ssp-fn tempdata.h5 \
    --n-objects 20 --output-dir agnvar_out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chimera_validation as cv  # noqa: E402

MODES = ("base", "fixed", "gray", "var")


def _injection_factor(wave_obs, redshift, log_amplitude, index, pivot_rest_a=5100.0):
    rest = np.asarray(wave_obs, dtype=float) / (1.0 + redshift)
    return np.exp(log_amplitude * (rest / pivot_rest_a) ** (-index))


def _select(rows, table, n_objects, min_ratio, max_ratio):
    ratio = {}
    for t in table:
        q = np.asarray(t["flux_qso"], dtype=float)
        g = np.asarray(t["flux_gal"], dtype=float)
        ratio[str(t["id"])] = float(np.nanmedian(q / g))
    keep = [r for r in rows if r["id"] in ratio and min_ratio < ratio[r["id"]] < max_ratio]
    keep.sort(key=lambda r: ratio[r["id"]])
    if len(keep) > n_objects:
        keep = [keep[i] for i in np.linspace(0, len(keep) - 1, n_objects).round().astype(int)]
    return keep, ratio


def _spectra(table, rows, log_amplitude, index):
    by_id = {str(t["id"]): t for t in table}
    res = float(table.meta["SPEC_RES"])
    out = {}
    for r in rows:
        t = by_id[r["id"]]
        wave = np.asarray(t["wave_obs"], dtype=float)
        flux = np.asarray(t["flux"], dtype=float)
        qso = np.nan_to_num(np.asarray(t["flux_qso"], dtype=float))
        err = np.asarray(t["flux_err"], dtype=float)
        good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        if log_amplitude != 0.0:
            flux = flux + (_injection_factor(wave, r["redshift"], log_amplitude, index) - 1.0) * qso
        out[r["id"]] = dict(wave=wave, flux=np.where(good, flux, 0.0), err=np.where(good, err, 1.0), mask=good,
                            resolving_power=res)
    return out


def _fit(rows, spectra, mode, args):
    from jaxsedfit import fit_batch

    configs = cv._configs(rows, args.dsps_ssp_fn, args.steps, args.learning_rate, spectra)
    for cfg in configs:
        cfg.likelihood.fit_spectrum_scale = mode in ("gray", "var")
        cfg.likelihood.fit_spectrum_agn_variability = mode == "var"
        cfg.likelihood.spectrum_scale_prior_sigma_dex = args.scale_prior_dex
    derived = ["log_agn_bol_luminosity_fit"] + (["spectrum_agn_log_amplitude"] if mode == "var" else [])
    t0 = time.time()
    batch = fit_batch(configs, method="laplace", chunk_size=args.chunk, derived_sites=derived)
    out = []
    for row, mv in zip(rows, batch.map_values):
        rec = {"id": row["id"], "log_stellar_mass_truth": row["log_stellar_mass_truth"],
               "log_stellar_mass_fit": float(mv["log_stellar_mass"]),
               "log_lbol_fit": float(mv["log_agn_bol_luminosity_fit"]) + 7.0,
               "log_lbol_truth": row["log_lbol_chimera_truth"]}
        for key in ("log_spectrum_scale", "spectrum_agn_log_amplitude", "agn_variability_index"):
            if key in mv:
                rec[key] = float(np.ravel(mv[key])[0])
        out.append(rec)
    print(f"  {mode}: {len(rows)} fits in {time.time() - t0:.0f} s", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dsps-ssp-fn", required=True)
    p.add_argument("--sample", default=str(cv.DEFAULT_SAMPLE))
    p.add_argument("--spectra", default=str(cv.DEFAULT_SPECTRA))
    p.add_argument("--n-objects", type=int, default=20)
    p.add_argument("--min-ratio", type=float, default=0.2, help="Minimum median quasar/host spectral flux ratio.")
    p.add_argument("--max-ratio", type=float, default=5.0)
    p.add_argument("--log-amplitudes", type=float, nargs="+", default=[np.log(1.5), -np.log(1.5)])
    p.add_argument("--index", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--modes", nargs="+", default=list(MODES[1:]), choices=MODES[1:],
                   help="Fit modes for the injected spectra.")
    p.add_argument("--scale-prior-dex", type=float, default=0.1, help="spectrum_scale_prior_sigma_dex.")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)

    from astropy.table import Table

    table = Table.read(args.spectra)
    rows, ratio = _select(cv.load_sample(Path(args.sample)), table, args.n_objects, args.min_ratio, args.max_ratio)
    print(f"{len(rows)} objects, quasar/host ratio {min(ratio[r['id']] for r in rows):.2f}"
          f"-{max(ratio[r['id']] for r in rows):.2f}", flush=True)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {"index": args.index, "ratio": {r["id"]: ratio[r["id"]] for r in rows},
               "base": _fit(rows, _spectra(table, rows, 0.0, args.index), "base", args)}
    for log_amp in args.log_amplitudes:
        spectra = _spectra(table, rows, log_amp, args.index)
        key = f"lnA={log_amp:+.3f}"
        print(key, flush=True)
        results[key] = {mode: _fit(rows, spectra, mode, args) for mode in args.modes}
        (outdir / "results.json").write_text(json.dumps(results, indent=1))
    (outdir / "results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
