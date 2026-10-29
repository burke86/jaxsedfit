# jaxsedfit benchmarks

This directory contains reusable performance benchmarks for `jaxsedfit`.

## PR benchmark

`jaxsedfit_pr_benchmark.py` times the jit-compiled value and gradient of the photometric log density of the fixed-redshift Fairall 9 tutorial fit (14 bands), the function every optimizer and sampler step evaluates. `measure` times the `jaxsedfit` importable in its process and writes one JSON record (compile time, median time per evaluation over several trials, and the log density at a fixed reference point); `report` combines base and PR records into a short comment:

| | base | PR | change |
|---|---:|---:|---:|
| value + gradient | 0.350 ms | 0.352 ms | +0.6% ± 1.1% |
| compile | 1.99 s | 2.01 s | +1% |
| log density at a fixed point | -124.464 | -124.464 | unchanged |

`report` exits non-zero, failing the PR check, if value + gradient is more than 10% slower and beyond 3σ of the run-to-run scatter, or compile is more than 25% slower. A changed log density flags a model change; it is not gated.

`.github/workflows/pr-benchmarks.yml` alternates five base and five PR processes on one runner (base, PR, base, …), each with its side's source tree first on `PYTHONPATH`, so drifts of the shared machine cancel. `.github/workflows/publish-pr-benchmarks.yml` posts or updates the PR comment, also when the benchmark failed. Locally:

```bash
for i in 1 2 3; do
  PYTHONPATH=../jaxsedfit-main/src python benchmarks/jaxsedfit_pr_benchmark.py measure --sha main --output /tmp/bench/base-$i.json
  PYTHONPATH=src python benchmarks/jaxsedfit_pr_benchmark.py measure --sha pr --output /tmp/bench/pr-$i.json
done
python benchmarks/jaxsedfit_pr_benchmark.py report --base "/tmp/bench/base-*.json" --pr "/tmp/bench/pr-*.json" --output /tmp/bench/output
```

## Chimera validation

`chimera_validation.py` checks the recovery of stellar mass, AGN bolometric luminosity and SFR on the luminosity-stratified Chimera sample (COSMOS galaxies combined with SDSS quasars; up to 400 objects in each bin of true injected AGN luminosity, `L < 42`, `42–43`, `43–44`, `44–45`). Each object is fitted with the staged MAP used by `JAXSEDFit.fit_map`, but all objects run through one compiled program with `jaxsedfit.batched`, so the full sample takes about 20 minutes on a laptop instead of hours of per-object NUTS. Error bars come from a Laplace approximation at the MAP.

The truths are the COSMOS2015 LePhare stellar mass and SFR (`MASS_MED`, `SFR_MED`) of the host and the injected AGN luminosity, the SDSS DR7 `logLbol` of the quasar plus log10 of the chimera QSO weight. The fitted AGN luminosity is `log_agn_bol_luminosity_fit` (dereddened 5100 Å luminosity × 9.26), converted from W to erg/s. The SDSS luminosities are not corrected for reddening, and the Chimera quasar never passed through the host's dust, so the report also scores `log_agn_bol_luminosity_attenuated_fit` (the same, attenuated by the fitted host and nuclear screens) against the same truth. It has no limits yet. SFR metrics use star-forming hosts only (true log sSFR > −11); COSMOS2015 SFRs of quiescent hosts are effectively upper limits, so those hosts are plotted but not scored.

`run` writes `results.json`, `metrics.json` and one residual plot per quantity (`chimera_dmass_vs_mass`, `chimera_dlbol_vs_lbol`, `chimera_dlbol_attenuated_vs_lbol`, `chimera_dsfr_vs_sfr`, as PNG and PDF). `report` turns the photometry-only run, and optionally the joint run, into one compact Markdown report: per run, one table of RMS and outlier fraction (|Δ| > 0.5 dex) by quantity and luminosity bin, with the change from main's metrics when `--baseline` is given, and the full per-bin statistics and plots behind a fold. It exits non-zero if any bin of any quantity exceeds its limits in `chimera_validation_thresholds.json` (per run, quantity and bin; set from a reference run as RMS × 1.15 and outlier fraction + 0.05). Update the limits in the same PR as an intended change to the recovery.

```bash
python benchmarks/chimera_validation.py run \
  --output-dir /tmp/chimera-validation \
  --dsps-ssp-fn ../jaxqsofit/tempdata.h5   # add --per-bin 50 for a quick look
python benchmarks/chimera_validation.py report --photometry /tmp/chimera-validation --output /tmp/chimera-validation/report.md
```

`data/chimera_validation_sample.csv` holds the sample's photometry and truth values. Two cuts keep the truth values meaningful:

- **Host photo-z.** The truth masses are COSMOS2015 LePhare masses computed at the photo-z, while the Chimeras use the spec-z. For hosts with a catastrophic photo-z the "true" mass can be off by more than a dex, and these hosts dominate below log M★ ≈ 8. The sample keeps hosts with |PHOTOZ − z_spec|/(1+z_spec) < 0.15.
- **Quasar host light.** The SDSS quasar photometry contains the quasar's own host galaxy, which the chimera adds to the COSMOS galaxy. For such chimeras neither the galaxy truth nor the AGN truth is clean. The sample keeps chimeras with `QSO_host_contamination_dex < 0.05`, a per-chimera estimate from the Shen et al. (2011) host fraction, the QSO weight and the galaxy's own 5100 Å luminosity.

The current file was built from Chimeras made with the per-chimera contamination estimate. That estimate is proposed upstream in [JohannesBuchner/chimeras#4](https://github.com/JohannesBuchner/chimeras/pull/4) and was built from the `burke86/chimeras` branch `agent/qso-host-contamination`; the photo-z cut is applied here, in `make-sample`. Rebuild the sample only when the sample definition changes:

```bash
python benchmarks/chimera_validation.py make-sample \
  --data-dir <chimeras build directory with chimeras-grahsp.fits and chimeras-fullinfo.fits> \
  --cosmos2015-catalog <path to COSMOS2015_Laigle+.fits>
```

`.github/workflows/chimera-validation.yml` runs the photometry-only and joint fits in parallel on pull requests that touch `src/`, on pushes to main and on manual dispatch, then one report job writes a single PR comment and gates on the limits. Runs on main store their `metrics.json` and plots under `main/` on the `chimera-validation-plots` branch; pull requests show their change from those. For pull requests from branches of this repository the plots go under `pr-<number>/<sha>/` on the same branch so the comment can embed them.

### Joint photometry + spectra

`run --spectra` fits the objects that have a spectrum in `data/chimera_validation_spectra.fits` with their photometry and spectrum together (quasar emission lines and Fe II on; the spectrum scale is fixed because the composite spectra are scaled to the total photometry). It makes the same plots as the photometry-only run. `--compare-photometry` also fits the same objects with photometry only; the report shows that table under the joint run's fold. The joint results have their own limits in the thresholds file.

```bash
python benchmarks/chimera_validation.py run \
  --output-dir /tmp/chimera-joint-validation --dsps-ssp-fn tempdata.h5 --spectra --compare-photometry --chunk 8
```

### Chimera spectra

`data/chimera_validation_spectra.fits` (14 MB) holds composite spectra for the 241 objects in `chimera_validation_sample.csv` whose host has a flux-calibrated spectrum from DESI DR1 or LEGA-C. It is meant for spectral and joint photometry + spectra tests on the Chimeras. Rows are keyed by `id`, like the photometry sample. Each row holds the observed-frame wavelength [Å] and the total, error, host and weighted-quasar flux densities [mJy] as float32 arrays, at R = 1500 (header keyword `SPEC_RES`). About 30–40% of the pixels in LEGA-C hosts are masked (NaN flux, infinite error). The 241 are simply the sample objects that have a spectrum, so they are not balanced across luminosity bins and some share a host.

Each spectrum is the host spectrum plus the quasar's SDSS DR7 spectrum times the chimera QSO weight. Each component is dereddened and scaled to its own photometry, the quasar is moved to the host redshift, and both are convolved to R = 1500. The spectra are built by the `make spectra` target of the Chimeras (branch `agent/chimera-spectra` of `burke86/chimeras`; about 300 MB for all 2,474 Chimeras with spectra, not committed). Rebuild the fixture from such a build with:

```bash
python benchmarks/make_chimera_spectra_sample.py --chimeras-dir <chimeras build directory with chimeras-spectra.fits>
```
