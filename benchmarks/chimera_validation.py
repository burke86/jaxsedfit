"""Chimera stellar-mass, AGN luminosity and SFR validation with batched MAP fits.

Fits the luminosity-stratified Chimera sample (COSMOS galaxies + SDSS quasars,
binned by the true injected AGN luminosity) with ``jaxsedfit.batched``, plots
the residuals of stellar mass, AGN bolometric luminosity and SFR against their
true values in each luminosity bin, and checks the per-bin outlier fraction and
RMS against ``chimera_validation_thresholds.json`` (quantities without
thresholds there are reported but not gated).

Subcommands
-----------
make-sample
    Build ``data/chimera_validation_sample.csv`` from the Chimera FITS tables
    (only needed when the sample definition changes). The truth masses are
    COSMOS2015 LePhare masses, computed at the photo-z, so hosts whose COSMOS2015
    photo-z disagrees with the spec-z used by the Chimeras are dropped; their
    "true" masses can be off by more than a dex. The SFR truths are COSMOS2015
    LePhare SFRs (SFR_MED).
run
    Fit the sample and write ``results.json``, ``metrics.json``, the residual plots
    and a Markdown summary (``output``) for the PR comment. With ``--spectra``, fit
    photometry and spectra jointly instead, for the objects that have a Chimera
    spectrum in ``data/chimera_validation_spectra.fits``; these joint results are
    reported, not gated, and ``--compare-photometry`` adds photometry-only fits of
    the same objects for comparison.
check
    Exit non-zero if ``metrics.json`` fails the thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SAMPLE = HERE / "data" / "chimera_validation_sample.csv"
DEFAULT_THRESHOLDS = HERE / "chimera_validation_thresholds.json"
DEFAULT_SPECTRA = HERE / "data" / "chimera_validation_spectra.fits"
LUMINOSITY_BIN_EDGES = [-np.inf, 42.0, 43.0, 44.0, 45.0]
OUTLIER_DEX = 0.5
# Recovered quantities: fitted site, truth column, offset from the site's units to the
# truth's, and plot. The AGN truth is the injected (QSO-weighted) SDSS DR7 bolometric
# luminosity; the host truths are COSMOS2015 LePhare medians.
QUANTITIES = {
    "log_stellar_mass": dict(
        site="log_stellar_mass", truth="log_stellar_mass_truth", offset=0.0, title="Stellar mass",
        label=r"\log_{10}(M_\star/M_\odot)", delta=r"$\Delta\log M_\star$ [dex]", plot="chimera_dmass_vs_mass.png",
    ),
    "log_lbol_agn": dict(
        site="log_agn_bol_luminosity_fit", truth="log_lbol_chimera_truth", offset=7.0,  # W -> erg/s
        title="AGN bolometric luminosity", label=r"\log_{10}(L_{\rm bol,AGN}/{\rm erg\,s^{-1}})",
        delta=r"$\Delta\log L_{\rm bol,AGN}$ [dex]", plot="chimera_dlbol_vs_lbol.png",
    ),
    "log_sfr": dict(
        site="log_sfr_fit", truth="log_sfr_truth", offset=0.0, title="Star formation rate",
        label=r"\log_{10}({\rm SFR}/M_\odot\,{\rm yr}^{-1})", delta=r"$\Delta\log {\rm SFR}$ [dex]",
        plot="chimera_dsfr_vs_sfr.png", unscored_label="quiescent host (not scored)",
        note="Scored on star-forming hosts (true log sSFR > −11); quiescent hosts are plotted as open circles.",
    ),
}
SFR_MIN_LOG_SSFR = -11.0
COMMENT_MARKER = "jaxsedfit Chimera validation"
# Not a superstring of COMMENT_MARKER, so the two PR comments are found separately.
JOINT_COMMENT_MARKER = "jaxsedfit joint Chimera validation"


def _bin_label(lo: float, hi: float) -> str:
    return f"L < {hi:g}" if np.isneginf(lo) else f"{lo:g} ≤ L < {hi:g}"


# --------------------------------------------------------------------------- #
# Sample
# --------------------------------------------------------------------------- #
def make_sample(args) -> int:
    from astropy.io import fits

    from jaxsedfit.benchmark import CHIMERA_FILTER_NAMES

    column_map = {"spitzer.irac.I1": "IRAC1", "spitzer.irac.I2": "IRAC2"}
    data_dir = Path(args.data_dir)
    phot = fits.getdata(data_dir / "chimeras-grahsp.fits", 1)
    truth = fits.getdata(data_dir / "chimeras-fullinfo.fits", 1)
    phot_by_id = {str(phot["id"][i]): phot[i] for i in range(len(phot))}
    truth_by_id = {str(truth["id"][i]): truth[i] for i in range(len(truth))}
    cosmos = fits.getdata(args.cosmos2015_catalog, 1, memmap=True)
    photoz_by_host = dict(zip(np.asarray(cosmos["NUMBER"]).tolist(), np.asarray(cosmos["PHOTOZ"], dtype=float).tolist()))

    rows = []
    for row_id in sorted(set(phot_by_id) & set(truth_by_id)):
        p, t = phot_by_id[row_id], truth_by_id[row_id]
        weight = float(p["chimera_QSO_weight"])
        row = {
            "id": row_id,
            "ID_COSMOS": str(p["ID_COSMOS"]),
            "redshift": float(p["redshift"]),
            "chimera_QSO_weight": weight,
            "resample_weight": float(t["resample_weight"]),
            "log_stellar_mass_truth": float(t["MASS_MED_GAL"]),
            "log_lbol_qso_truth": float(t["logLbol_QSO"]),
            "log_lbol_chimera_truth": float(t["logLbol_QSO"]) + np.log10(weight),
            "log_sfr_truth": float(t["SFR_MED_GAL"]),
            "photoz_cosmos2015": photoz_by_host.get(int(float(p["ID_COSMOS"])), np.nan),
        }
        if "QSO_host_contamination_dex" in truth.dtype.names:  # chimeras built with add_host_contamination.py
            row["qso_host_contamination_dex"] = float(t["QSO_host_contamination_dex"])
        for name in CHIMERA_FILTER_NAMES:
            row[name] = float(p[column_map.get(name, name)])
            row[f"{name}_err"] = float(p[f"{column_map.get(name, name)}_err"])
        rows.append(row)
    n_all = len(rows)
    rows = [r for r in rows if abs(r["photoz_cosmos2015"] - r["redshift"]) / (1.0 + r["redshift"]) < args.max_photoz_dz]
    if args.max_qso_host_contamination >= 0:
        # The quasar photometry contains the quasar's own host, which the chimera adds to the
        # COSMOS galaxy; neither the galaxy nor the AGN truth is clean for contaminated chimeras.
        if not rows or "qso_host_contamination_dex" not in rows[0]:
            raise SystemExit("--max-qso-host-contamination needs chimeras built with add_host_contamination.py")
        rows = [r for r in rows if r["qso_host_contamination_dex"] < args.max_qso_host_contamination]

    # Same selection as notebooks/03_chimera_benchmarks.ipynb: up to N objects per bin of
    # true injected AGN luminosity, drawn without selecting on the QSO weight.
    rng = np.random.default_rng(args.seed)
    selected = []
    for b, (lo, hi) in enumerate(zip(LUMINOSITY_BIN_EDGES[:-1], LUMINOSITY_BIN_EDGES[1:])):
        in_bin = [r for r in rows if np.isfinite(r["log_lbol_chimera_truth"]) and lo <= r["log_lbol_chimera_truth"] < hi]
        if len(in_bin) > args.per_bin:
            take = np.sort(rng.choice(len(in_bin), size=args.per_bin, replace=False))
            in_bin = [in_bin[i] for i in take]
        selected.extend({**r, "luminosity_bin_index": b} for r in in_bin)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(selected[0]))
        writer.writeheader()
        for r in selected:
            writer.writerow({k: (repr(float(v)) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {len(selected)} rows to {out} ({len(rows)} of {n_all} Chimeras pass the photo-z and contamination cuts)")
    return 0


def load_sample(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("id", "ID_COSMOS"):
                r[k] = int(v) if k == "luminosity_bin_index" else float(v)
    return rows


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def load_spectra(path: Path) -> dict[str, dict]:
    """Chimera spectra by id: observed wavelength [A], flux and error [mJy], good-pixel mask."""
    from astropy.table import Table

    table = Table.read(path)
    out = {}
    for row in table:
        wave = np.asarray(row["wave_obs"], dtype=float)
        flux = np.asarray(row["flux"], dtype=float)
        err = np.asarray(row["flux_err"], dtype=float)
        # flux_err is infinite where the host or the quasar spectrum is missing.
        good = np.isfinite(flux) & np.isfinite(err) & (err > 0)
        out[str(row["id"])] = dict(wave=wave, flux=np.where(good, flux, 0.0), err=np.where(good, err, 1.0), mask=good,
                                   resolving_power=float(table.meta["SPEC_RES"]))
    return out


def _configs(rows, dsps_ssp_fn, steps, learning_rate, spectra=None):
    import numpyro.distributions as dist

    from jaxsedfit import SpectroscopyData
    from jaxsedfit.benchmark import build_chimera_fit_config

    configs = []
    for row in rows:
        cfg = build_chimera_fit_config(row, dsps_ssp_fn=dsps_ssp_fn)
        cfg.prior_config.stellar_mass = dist.Uniform(6.0, 12.0)
        cfg.inference.map_steps = steps
        cfg.inference.learning_rate = learning_rate
        cfg.inference.num_samples = 400
        if spectra is not None:
            spec = spectra[row["id"]]
            cfg.spectroscopy = SpectroscopyData(wave_obs=spec["wave"], fluxes=spec["flux"], errors=spec["err"],
                                                mask=spec["mask"], resolving_power=spec["resolving_power"])
            # Quasar emission lines and Fe II in the spectrum; the composite spectra are scaled
            # to the total photometry of each component, so the spectrum scale is not fitted.
            cfg.agn.fit_lines = True
            cfg.agn.fit_feii = True
            cfg.likelihood.fit_spectrum_scale = False
        configs.append(cfg)
    return configs


def fit_sample(rows, *, dsps_ssp_fn, chunk, steps, learning_rate, spectra=None):
    """Batched MAP + Laplace fits (``jaxsedfit.fit_batch``); point estimate = MAP."""
    from jaxsedfit import fit_batch

    t0 = time.time()
    batch = fit_batch(_configs(rows, dsps_ssp_fn, steps, learning_rate, spectra), method="laplace", chunk_size=chunk,
                      derived_sites=[q["site"] for q in QUANTITIES.values() if q["site"] != "log_stellar_mass"])
    results = []
    for row, draws, map_values, diag in zip(rows, batch.samples, batch.map_values, batch.diagnostics):
        result = {
            "id": row["id"],
            "luminosity_bin_index": int(row["luminosity_bin_index"]),
            "redshift": row["redshift"],
            "chimera_QSO_weight": row["chimera_QSO_weight"],
        }
        for name, q in QUANTITIES.items():
            p02, p16, p84, p97 = np.percentile(np.asarray(draws[q["site"]]) + q["offset"], [2.275, 16.0, 84.0, 97.725])
            result.update({
                f"{name}_truth": row[q["truth"]],
                f"{name}_fit": float(map_values[q["site"]]) + q["offset"],
                f"{name}_p02275": float(p02),
                f"{name}_p16": float(p16),
                f"{name}_p84": float(p84),
                f"{name}_p97725": float(p97),
            })
        result.update({"map_loss": diag["loss"], "hessian_min_eig": diag["hessian_min_eigenvalue"]})
        results.append(result)
    return results, time.time() - t0


# --------------------------------------------------------------------------- #
# Metrics, plot, comment
# --------------------------------------------------------------------------- #
def _in_metrics(result, name) -> bool:
    # COSMOS2015 SFRs of quiescent hosts are only upper limits in practice, so SFR
    # recovery is scored on star-forming hosts (they are still plotted).
    if name == "log_sfr":
        return result["log_sfr_truth"] - result["log_stellar_mass_truth"] > SFR_MIN_LOG_SSFR
    return True


def compute_metrics(results, name) -> list[dict]:
    metrics = []
    for b, (lo, hi) in enumerate(zip(LUMINOSITY_BIN_EDGES[:-1], LUMINOSITY_BIN_EDGES[1:])):
        res = np.array([r[f"{name}_fit"] - r[f"{name}_truth"]
                        for r in results if r["luminosity_bin_index"] == b and _in_metrics(r, name)])
        finite = res[np.isfinite(res)]
        metrics.append({
            "bin": b,
            "label": _bin_label(lo, hi),
            "n": int(res.size),
            "n_failed": int(res.size - finite.size),
            "median": float(np.median(finite)),
            "nmad": float(1.4826 * np.median(np.abs(finite - np.median(finite)))),
            "rms": float(np.sqrt(np.mean(finite**2))),
            # Failed fits count as outliers.
            "outlier_fraction": float((np.sum(np.abs(finite) > OUTLIER_DEX) + res.size - finite.size) / max(res.size, 1)),
        })
    return metrics


def check_metrics(metrics, thresholds) -> list[str]:
    """Compare per-bin metrics with the thresholds; quantities without thresholds are not gated."""
    failures = []
    for name, bins in metrics.items():
        for m in bins:
            t = thresholds.get(name, {}).get(str(m["bin"]))
            if t is None:
                continue
            prefix = f"{QUANTITIES[name]['title']}, {m['label']}"
            if m["outlier_fraction"] > t["max_outlier_fraction"]:
                failures.append(f"{prefix}: outlier fraction {m['outlier_fraction']:.3f} > {t['max_outlier_fraction']:.3f}")
            if m["rms"] > t["max_rms"]:
                failures.append(f"{prefix}: RMS {m['rms']:.3f} > {t['max_rms']:.3f} dex")
    return failures


def plot_results(results, metrics, name, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from jaxsedfit.mplstyle import style_path

    plt.style.use(style_path())
    q = QUANTITIES[name]
    ylim = 3.0
    truth = np.array([r[f"{name}_truth"] for r in results])
    resid = np.array([r[f"{name}_fit"] for r in results]) - truth
    lo95 = np.array([r[f"{name}_p02275"] for r in results]) - truth
    hi95 = np.array([r[f"{name}_p97725"] for r in results]) - truth
    weight = np.array([r["chimera_QSO_weight"] for r in results])
    bins = np.array([r["luminosity_bin_index"] for r in results])
    scored = np.array([_in_metrics(r, name) for r in results])
    norm = LogNorm(vmin=weight[weight > 0].min(), vmax=weight.max())
    shown = np.clip(resid, -0.97 * ylim, 0.97 * ylim)  # off-scale points drawn at the edge as triangles
    clipped = np.abs(resid) > 0.97 * ylim

    fig, axes = plt.subplots(1, len(metrics), figsize=(17.0, 4.2), sharex=True, sharey=True, constrained_layout=True)
    scatter = None
    for ax, m in zip(np.atleast_1d(axes), metrics):
        sel = (bins == m["bin"]) & np.isfinite(resid)
        ax.axhline(0.0, color="black", lw=1.0, ls="--", zorder=0)
        for y in (OUTLIER_DEX, -OUTLIER_DEX):
            ax.axhline(y, color="0.55", lw=0.9, ls=":", zorder=0)
        bars = sel & ~clipped
        # A derived quantity's MAP can fall outside its Laplace draws (e.g. a few SFRs), so clip at zero.
        yerr = np.clip(np.vstack([resid[bars] - lo95[bars], hi95[bars] - resid[bars]]), 0.0, None)
        ax.errorbar(truth[bars], resid[bars], yerr=yerr,
                    fmt="none", ecolor="0.55", elinewidth=0.8, capsize=1.5, alpha=0.55, zorder=1)
        for subset, marker in ((sel & scored & ~clipped, "o"), (sel & scored & clipped, "^")):
            scatter = ax.scatter(truth[subset], np.where(clipped[subset], np.sign(resid[subset]) * 0.97 * ylim,
                                                         shown[subset]),
                                 c=weight[subset], cmap="viridis", norm=norm, s=42, alpha=0.9, marker=marker,
                                 edgecolor="black", linewidth=0.35, zorder=2)
        if not scored.all():
            ax.scatter(truth[sel & ~scored], shown[sel & ~scored], s=30, facecolor="none", edgecolor="0.45",
                       linewidth=0.7, zorder=1.5, label=q.get("unscored_label"))
        ax.set_title(f"{m['label']}  (N={m['n']})\noutliers={100 * m['outlier_fraction']:.1f}%, RMS={m['rms']:.2f} dex",
                     fontsize=11)
        ax.set_xlabel(f"True ${q['label']}$")
        ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=0.18)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel(q["delta"])
    if not scored.all():
        first.legend(loc="lower left", fontsize=9)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Chimera QSO weight", pad=0.01)
    fig.savefig(path, dpi=110)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_comment(path: Path, metrics, failures, thresholds, *, sha, elapsed, image_base_url, joint=False,
                  comparison=None) -> None:
    if joint:
        lines = [
            f"## {JOINT_COMMENT_MARKER} (photometry + spectra): reported, not gated",
            "",
            f"Commit `{sha}` · {sum(m['n'] for m in metrics['log_stellar_mass'])} Chimeras with a composite spectrum "
            f"(host from DESI DR1 or LEGA-C + weighted SDSS quasar) · joint MAP + Laplace (`fit_batch`) in "
            f"{elapsed / 60:.1f} min",
            "",
            "Same plots as the photometry-only validation, for the objects with spectra, fitted with their "
            "photometry and spectrum together (quasar lines and Fe II on, spectrum scale fixed).",
            "",
            f"Residual = MAP − truth in bins of injected AGN luminosity (log erg/s); bars are Laplace 95% intervals; "
            f"outlier: |Δ| > {OUTLIER_DEX} dex.",
        ]
    else:
        status = "✅ passed" if not failures else "❌ failed"
        lines = [
            f"## {COMMENT_MARKER}: {status}",
            "",
            f"Commit `{sha}` · {sum(m['n'] for m in metrics['log_stellar_mass'])} Chimeras · "
            f"MAP + Laplace (`fit_batch`) in {elapsed / 60:.1f} min",
            "",
            f"Residual = MAP − truth in bins of injected AGN luminosity (log erg/s); bars are Laplace 95% intervals; "
            f"outlier: |Δ| > {OUTLIER_DEX} dex.",
        ]
    if comparison is not None:
        lines += ["", "### Joint vs photometry only, same objects", "",
                  "Each cell: photometry only → photometry + spectra.", "",
                  "| Quantity | AGN luminosity bin | N | median Δ | NMAD | RMS | outliers |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name, bins in metrics.items():
            for m, c in zip(bins, comparison[name]):
                lines.append(f"| {QUANTITIES[name]['title']} | {m['label']} | {m['n']} | "
                             f"{c['median']:+.3f} → {m['median']:+.3f} | {c['nmad']:.3f} → {m['nmad']:.3f} | "
                             f"{c['rms']:.3f} → {m['rms']:.3f} | "
                             f"{100 * c['outlier_fraction']:.1f}% → {100 * m['outlier_fraction']:.1f}% |")
    for name, bins in metrics.items():
        q = QUANTITIES[name]
        gated = name in thresholds
        lines += ["", f"### {q['title']}" + ("" if gated else " (reported, not gated)"), ""]
        if q.get("note"):
            lines += [q["note"], ""]
        lines += ["| AGN luminosity bin | N | median Δ | NMAD | RMS" + (" (max)" if gated else "")
                  + " | outliers" + (" (max)" if gated else "") + " |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m in bins:
            t = thresholds.get(name, {}).get(str(m["bin"]))
            rms = f"{m['rms']:.3f}"
            out = f"{100 * m['outlier_fraction']:.1f}%"
            if t is not None:
                rms += f" ({t['max_rms']:.3f})" + ("" if m["rms"] <= t["max_rms"] else " ❌")
                out += (f" ({100 * t['max_outlier_fraction']:.1f}%)"
                        + ("" if m["outlier_fraction"] <= t["max_outlier_fraction"] else " ❌"))
            lines.append(f"| {m['label']} | {m['n']} | {m['median']:+.3f} | {m['nmad']:.3f} | {rms} | {out} |")
        lines += ["", f"![{q['title']} residuals]({image_base_url}/{q['plot']})" if image_base_url
                  else "_Plot attached to the workflow artifacts._"]
    if failures:
        lines += ["", "**Failures**", ""] + [f"- {f}" for f in failures]
    workflow_url = os.environ.get("GITHUB_SERVER_URL", "") and (
        f"{os.environ['GITHUB_SERVER_URL']}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    if workflow_url:
        lines += ["", f"[Workflow run]({workflow_url})"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> int:
    rows = load_sample(args.sample)
    if args.per_bin is not None:
        rng = np.random.default_rng(0)
        keep = []
        for b in sorted({r["luminosity_bin_index"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["luminosity_bin_index"] == b]
            keep += sorted(rng.choice(idx, size=min(args.per_bin, len(idx)), replace=False).tolist())
        rows = [rows[i] for i in sorted(keep)]
    spectra = None
    if args.spectra:
        spectra = load_spectra(args.spectra)
        rows = [r for r in rows if r["id"] in spectra]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[chimera] fitting {len(rows)} objects" + (" with spectra" if spectra else ""), flush=True)

    fit_kwargs = dict(dsps_ssp_fn=args.dsps_ssp_fn, chunk=args.chunk, steps=args.steps, learning_rate=args.learning_rate)
    results, elapsed = fit_sample(rows, spectra=spectra, **fit_kwargs)
    metrics = {name: compute_metrics(results, name) for name in QUANTITIES}
    comparison = None
    if spectra and args.compare_photometry:
        print(f"[chimera] fitting the same {len(rows)} objects with photometry only", flush=True)
        phot_results, _ = fit_sample(rows, **fit_kwargs)
        (out / "results_photometry_only.json").write_text(json.dumps(phot_results, indent=1))
        comparison = {name: compute_metrics(phot_results, name) for name in QUANTITIES}
    # Joint fits are reported, not gated: the thresholds are for the photometry-only sample.
    thresholds = {} if spectra else {
        k: v for k, v in json.loads(Path(args.thresholds).read_text()).items() if not k.startswith("_")}
    failures = check_metrics(metrics, thresholds)

    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "metrics.json").write_text(json.dumps({**metrics, "failures": failures, "elapsed_s": elapsed,
                                                  **({"photometry_only": comparison} if comparison else {})}, indent=2))
    for name, q in QUANTITIES.items():
        plot_results(results, metrics[name], name, out / q["plot"])
    write_comment(out / "output", metrics, failures, thresholds, sha=args.sha, elapsed=elapsed,
                  image_base_url=args.image_base_url, joint=spectra is not None, comparison=comparison)
    print((out / "output").read_text())
    return 0


def check(args) -> int:
    metrics = json.loads((Path(args.output_dir) / "metrics.json").read_text())
    if metrics["failures"]:
        print("Chimera validation failed:\n  " + "\n  ".join(metrics["failures"]))
        return 1
    print("Chimera validation passed.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-sample", help="Build the validation sample CSV from the Chimera FITS tables.")
    p.add_argument("--data-dir", required=True, help="Directory with chimeras-grahsp.fits and chimeras-fullinfo.fits.")
    p.add_argument("--cosmos2015-catalog", required=True, help="COSMOS2015 catalog with NUMBER and PHOTOZ (Laigle et al. 2016).")
    p.add_argument("--max-photoz-dz", type=float, default=0.15,
                   help="Keep hosts with |PHOTOZ - z_spec| / (1 + z_spec) below this.")
    p.add_argument("--max-qso-host-contamination", type=float, default=0.05,
                   help="Keep chimeras with QSO_host_contamination_dex below this (dex). "
                        "Requires the per-chimera contamination column; pass a negative value to skip.")
    p.add_argument("--output", default=str(DEFAULT_SAMPLE))
    p.add_argument("--per-bin", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=make_sample)

    p = sub.add_parser("run", help="Fit the sample and write metrics, plot and comment.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dsps-ssp-fn", required=True)
    p.add_argument("--sample", default=str(DEFAULT_SAMPLE))
    p.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    p.add_argument("--per-bin", type=int, default=None, help="Random subset per luminosity bin (default: all).")
    p.add_argument("--chunk", type=int, default=32, help="Objects per compiled batch.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=5e-3)
    p.add_argument("--sha", default="local")
    p.add_argument("--image-base-url", default=None, help="URL of the directory holding the plots to embed in the comment.")
    p.add_argument("--spectra", nargs="?", const=str(DEFAULT_SPECTRA), default=None,
                   help="Fit photometry + spectra jointly for the objects in this spectra file "
                        f"(default when given without a path: {DEFAULT_SPECTRA.name}); reported, not gated.")
    p.add_argument("--compare-photometry", action="store_true",
                   help="With --spectra, also fit the same objects with photometry only and compare.")
    p.set_defaults(func=run)

    p = sub.add_parser("check", help="Exit non-zero if metrics.json fails the thresholds.")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
