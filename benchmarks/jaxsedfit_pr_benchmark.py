"""Gated base-vs-PR performance benchmark of the jaxsedfit photometric likelihood.

The benchmark times the jit-compiled value and gradient of the full photometric
log density of the fixed-redshift Fairall 9 tutorial fit (14 bands), the
function every optimizer and sampler step evaluates.

Subcommands
-----------
measure
    Time the jaxsedfit importable in this process and write one JSON record:
    compile time, the median time per evaluation over several trials, and the
    log density at a fixed reference point (the prior medians, seed 0).
report
    Combine alternating base and PR ``measure`` records into a short Markdown
    comment and exit non-zero if the PR is slower than the limits.

The workflow alternates base and PR processes on one runner (base, PR, base,
PR, ...), so slow drifts of the machine affect both sides equally.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import jax
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import log_density

from jaxsedfit.config import AGNConfig, FilterSet, FitConfig, GalaxyConfig, InferenceConfig, LikelihoodConfig, Observation, PhotometryData, PriorConfig
from jaxsedfit.core import JAXSEDFit
from jaxsedfit.filters import load_filter_curves
from jaxsedfit.model import grahsp_photometric_model

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMENT_MARKER = "jaxsedfit PR benchmark"
# A slowdown fails the benchmark when it is larger than this fraction and larger than
# NOISE_SIGMA times the run-to-run scatter.
MAX_SLOWDOWN = 0.10
MAX_COMPILE_SLOWDOWN = 0.25
NOISE_SIGMA = 3.0


def build_fairall9_config(dsps_ssp_fn: str | Path) -> FitConfig:
    """The fixed-redshift Fairall 9 fit from the tutorials."""
    from astropy.table import Table

    phot = Table.read(REPO_ROOT / "notebooks" / "data" / "fairall9_vizier_photometry.ecsv")
    filters = [str(name) for name in phot["grahsp_filter"]]
    return FitConfig(
        observation=Observation(object_id="Fairall 9 benchmark", redshift=float(phot.meta["redshift"])),
        photometry=PhotometryData(
            filter_names=filters,
            fluxes=[float(x) for x in phot["flux_mjy"]],
            errors=[float(x) for x in phot["err_mjy"]],
            psf_fwhm_arcsec=[float(x) for x in phot["psf_fwhm_arcsec"]],
        ),
        filters=FilterSet(curves=load_filter_curves(filters)),
        galaxy=GalaxyConfig(dsps_ssp_fn=str(dsps_ssp_fn)),
        agn=AGNConfig(agn_type=1),
        likelihood=LikelihoodConfig(use_host_capture_model=False),
        inference=InferenceConfig(seed=0),
        prior_config=PriorConfig(
            stellar_mass=dist.Normal(10.5, 1.0),
            host={"ebv_gal": dist.HalfNormal(0.15)},
            agn={"ebv_agn": dist.HalfNormal(0.15)},
        ),
    )


def measure(args) -> int:
    fitter = JAXSEDFit(build_fairall9_config(args.dsps_ssp_fn))
    model = partial(grahsp_photometric_model, fitter.context, include_components=False)
    trace = handlers.trace(
        handlers.substitute(handlers.seed(model, 0), substitute_fn=lambda site: init_to_median(site, num_samples=15))
    ).get_trace()
    params = {name: site["value"] for name, site in trace.items() if site["type"] == "sample" and not site["is_observed"]}
    value_and_grad = jax.jit(jax.value_and_grad(lambda p: log_density(model, (), {}, p)[0]))

    start = time.perf_counter()
    value, _ = jax.block_until_ready(value_and_grad(params))
    compile_s = time.perf_counter() - start
    trial_ms = []
    for _ in range(args.trials):
        start = time.perf_counter()
        for _ in range(args.repeats):
            out = value_and_grad(params)
        jax.block_until_ready(out)
        trial_ms.append(1.0e3 * (time.perf_counter() - start) / args.repeats)

    record = {
        "label": args.label,
        "sha": args.sha,
        "compile_s": compile_s,
        "ms_per_eval": statistics.median(trial_ms),
        "trial_ms": trial_ms,
        "log_density": float(value),
        "jax": jax.__version__,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    return 0


def _summary(records, key):
    values = [r[key] for r in records]
    median = statistics.median(values)
    # Scaled median absolute deviation: a robust run-to-run standard deviation.
    mad = 1.4826 * statistics.median([abs(v - median) for v in values]) if len(values) > 1 else 0.0
    return median, mad


def render_report(base, pr) -> tuple[str, list[str]]:
    base_ms, base_mad = _summary(base, "ms_per_eval")
    pr_ms, pr_mad = _summary(pr, "ms_per_eval")
    base_compile, _ = _summary(base, "compile_s")
    pr_compile, _ = _summary(pr, "compile_s")
    n = min(len(base), len(pr))
    # Uncertainty of the difference of the two medians (each from n runs).
    noise_ms = (1.2533 * (base_mad**2 + pr_mad**2) ** 0.5 / n**0.5) if n > 1 else 0.0
    slowdown = pr_ms / base_ms - 1.0
    compile_slowdown = pr_compile / base_compile - 1.0

    failures = []
    if slowdown > MAX_SLOWDOWN and pr_ms - base_ms > NOISE_SIGMA * noise_ms:
        failures.append(f"value + gradient is {100 * slowdown:.0f}% slower (limit {100 * MAX_SLOWDOWN:.0f}%)")
    if compile_slowdown > MAX_COMPILE_SLOWDOWN:
        failures.append(f"compile is {100 * compile_slowdown:.0f}% slower (limit {100 * MAX_COMPILE_SLOWDOWN:.0f}%)")

    base_logp, pr_logp = base[0]["log_density"], pr[0]["log_density"]
    logp_change = "unchanged" if abs(pr_logp - base_logp) <= 1e-6 * max(1.0, abs(base_logp)) else f"{pr_logp - base_logp:+.4g}"
    status = "✅ no slowdown" if not failures else "❌ slower"
    lines = [
        f"### {COMMENT_MARKER}: {status}",
        "",
        f"`{base[0]['sha'][:7]}` → `{pr[0]['sha'][:7]}` · jit-compiled value and gradient of the Fairall 9 "
        f"photometric log density · {n} alternating runs per side on one runner",
        "",
        "| | base | PR | change |",
        "|---|---:|---:|---:|",
        f"| value + gradient | {base_ms:.3f} ms | {pr_ms:.3f} ms | {100 * slowdown:+.1f}% ± {100 * noise_ms / base_ms:.1f}% |",
        f"| compile | {base_compile:.2f} s | {pr_compile:.2f} s | {100 * compile_slowdown:+.0f}% |",
        f"| log density at a fixed point | {base_logp:.6g} | {pr_logp:.6g} | {logp_change} |",
        "",
        f"Fails if value + gradient is more than {100 * MAX_SLOWDOWN:.0f}% slower (and beyond {NOISE_SIGMA:.0f}σ of the "
        f"run-to-run scatter) or compile is more than {100 * MAX_COMPILE_SLOWDOWN:.0f}% slower. A changed log density "
        "means the model itself changed.",
    ]
    if failures:
        lines += ["", "**Failures**", ""] + [f"- {f}" for f in failures]
    server = os.environ.get("GITHUB_SERVER_URL")
    if server:
        lines += ["", f"[Workflow run]({server}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
                      f"{os.environ.get('GITHUB_RUN_ID', '')})"]
    return "\n".join(lines) + "\n", failures


def report(args) -> int:
    def load(pattern):
        return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]

    base, pr = load(args.base), load(args.pr)
    if not base or not pr:
        print("Missing base or PR measurements.")
        return 1
    text, failures = render_report(base, pr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("measure", help="Time this process's jaxsedfit and write one JSON record.")
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="local")
    p.add_argument("--sha", default=os.getenv("GITHUB_SHA", "local"))
    p.add_argument("--dsps-ssp-fn", default=str(REPO_ROOT / "tempdata.h5"))
    p.add_argument("--repeats", type=int, default=200, help="Evaluations per timing trial.")
    p.add_argument("--trials", type=int, default=5)
    p.set_defaults(func=measure)

    p = sub.add_parser("report", help="Write the PR comment; exit non-zero past the limits.")
    p.add_argument("--base", required=True, help="Glob of base measure records.")
    p.add_argument("--pr", required=True, help="Glob of PR measure records.")
    p.add_argument("--output", required=True)
    p.set_defaults(func=report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
