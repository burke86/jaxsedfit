import csv
from pathlib import Path

import jax
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
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
