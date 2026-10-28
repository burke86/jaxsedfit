import importlib.util
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(rms, outliers):
    labels = ["L < 42", "42 ≤ L < 43", "43 ≤ L < 44", "44 ≤ L < 45"]
    bins = [
        {"bin": b, "label": labels[b], "n": 100, "n_failed": 0, "median": 0.0, "nmad": 0.1,
         "rms": rms, "outlier_fraction": outliers}
        for b in range(4)
    ]
    return {"log_stellar_mass": bins, "log_lbol_agn": bins, "log_sfr": bins, "elapsed_s": 60.0, "sha": "abc"}


def test_chimera_report_gates_every_quantity_and_shows_the_change_from_main():
    cv = _load("chimera_validation")
    limits = {str(b): {"max_outlier_fraction": 0.2, "max_rms": 0.5} for b in range(4)}
    thresholds = {"photometry": {name: limits for name in cv.QUANTITIES}}

    text, failures = cv.render_report({"photometry": _metrics(0.4, 0.1)}, thresholds, sha="abc",
                                      baseline={"photometry": _metrics(0.3, 0.1)}, baseline_sha="main")
    assert failures == []
    assert "✅ passed" in text and "0.40 (+0.10) · 10%" in text

    text, failures = cv.render_report({"photometry": _metrics(0.6, 0.1)}, thresholds, sha="abc")
    assert len(failures) == 3 * 4  # every quantity is gated in every bin
    assert "❌ failed" in text

    # A run whose fit job crashed is a failure, not a silent omission.
    text, failures = cv.render_report({"photometry": _metrics(0.4, 0.1)}, thresholds, sha="abc", missing=["joint"])
    assert failures == ["Photometry + spectra: no results (the fit did not finish; see the workflow run)"]
    assert "❌ failed" in text and "photometry + spectra: ❌ no results" in text


def test_perf_report_fails_only_slowdowns_beyond_the_noise():
    bench = _load("jaxsedfit_pr_benchmark")

    def records(ms, compile_s=2.0):
        return [{"sha": "abcdef0", "ms_per_eval": m, "compile_s": compile_s, "log_density": -1.0} for m in ms]

    base = records([1.00, 1.01, 0.99, 1.00, 1.02])
    _, failures = bench.render_report(base, records([1.30, 1.31, 1.29, 1.30, 1.32]))
    assert any("value + gradient" in f for f in failures)
    # 20% slower, but within the run-to-run scatter.
    _, failures = bench.render_report(records([1.0, 0.6, 1.4, 1.0, 0.7]), records([1.2, 0.7, 1.7, 1.2, 0.8]))
    assert failures == []
    _, failures = bench.render_report(base, records([1.0, 1.0, 1.0, 1.0, 1.0], compile_s=3.0))
    assert any("compile" in f for f in failures)
