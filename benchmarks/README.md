# jaxsedfit benchmarks

This directory contains reusable performance benchmarks for `jaxsedfit`.

## PR benchmark

`jaxsedfit_pr_benchmark.py` runs a fixed-redshift Fairall 9 photometric likelihood benchmark and writes JSON plus Markdown output suitable for a pull-request comment. Timing measurements run multiple trials by default and report mean plus standard error.

Local example:

```bash
conda run -n sed python benchmarks/jaxsedfit_pr_benchmark.py run \
  --output-dir /tmp/jaxsedfit-bench \
  --label local \
  --sha local \
  --dsps-ssp-fn ../jaxqsofit/tempdata.h5 \
  --trials 3
```

Compare two benchmark JSON files:

```bash
conda run -n sed python benchmarks/jaxsedfit_pr_benchmark.py compare \
  --baseline-json /tmp/jaxsedfit-bench-base/benchmark.json \
  --candidate-json /tmp/jaxsedfit-bench-head/benchmark.json \
  --output-dir /tmp/jaxsedfit-bench-compare
```

The GitHub workflows in `.github/workflows/pr-benchmarks.yml` and `.github/workflows/publish-pr-benchmarks.yml` run this benchmark for base and PR commits, upload artifacts, and create or update a PR comment containing the pre/post results.

## Chimera validation

`chimera_validation.py` checks stellar-mass recovery on the luminosity-stratified Chimera sample (COSMOS galaxies combined with SDSS quasars; up to 400 objects in each bin of true injected AGN luminosity, `L < 42`, `42–43`, `43–44`, `44–45`). Each object is fitted with the staged MAP used by `JAXSEDFit.fit_map`, but all objects run through one compiled program with `jaxsedfit.batched`, so the full sample takes about 20 minutes on a laptop instead of hours of per-object NUTS. Error bars come from a Laplace approximation at the MAP.

The run writes `results.json`, `metrics.json`, the residual plot `chimera_dmass_vs_mass.png` and a Markdown summary (`output`). `check` fails if, in any luminosity bin, the fraction of objects with |Δlog M★| > 0.5 dex or the RMS of Δlog M★ exceeds the limits in `chimera_validation_thresholds.json`. Update those limits in the same PR as an intended change to mass recovery.

```bash
python benchmarks/chimera_validation.py run \
  --output-dir /tmp/chimera-validation \
  --dsps-ssp-fn ../jaxqsofit/tempdata.h5   # add --per-bin 50 for a quick look
python benchmarks/chimera_validation.py check --output-dir /tmp/chimera-validation
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

### Chimera spectra

`data/chimera_validation_spectra.fits` (14 MB) holds composite spectra for the 241 objects in `chimera_validation_sample.csv` whose host has a flux-calibrated spectrum from DESI DR1 or LEGA-C. It is meant for spectral and joint photometry + spectra tests on the Chimeras. Rows are keyed by `id`, like the photometry sample. Each row holds the observed-frame wavelength [Å] and the total, error, host and weighted-quasar flux densities [mJy] as float32 arrays, at R = 1500 (header keyword `SPEC_RES`). About 30–40% of the pixels in LEGA-C hosts are masked (NaN flux, infinite error). The 241 are simply the sample objects that have a spectrum, so they are not balanced across luminosity bins and some share a host.

Each spectrum is the host spectrum plus the quasar's SDSS DR7 spectrum times the chimera QSO weight. Each component is dereddened and scaled to its own photometry, the quasar is moved to the host redshift, and both are convolved to R = 1500. The spectra are built by the `make spectra` target of the Chimeras (branch `agent/chimera-spectra` of `burke86/chimeras`; about 300 MB for all 2,474 Chimeras with spectra, not committed). Rebuild the fixture from such a build with:

```bash
python benchmarks/make_chimera_spectra_sample.py --chimeras-dir <chimeras build directory with chimeras-spectra.fits>
```

`.github/workflows/chimera-validation.yml` runs this on pull requests that touch `src/` (and on manual dispatch). `.github/workflows/publish-chimera-validation.yml` then pushes the plot to the `chimera-validation-plots` branch and creates or updates a PR comment with the metrics table and the embedded plot.
