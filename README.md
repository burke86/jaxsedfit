<p align="center">
  <img src="docs/_static/jaxsedfit_logo.png" alt="JAXSEDFit logo" width="600">
</p>

# JAXSEDFit

`JAXSEDFit` is a Bayesian SED fitting code for AGN and galaxies, built with JAX and NumPyro. It is an experimental JAX-based implementation of `CIGALE`, `GRAHSP` and `PyQSOFit`: it ports `GRAHSP`/`pcigale` model components and the `PyQSOFit` quasar spectral model into JAX/NumPyro and combines them with JAX-native galaxy models based on `Diffstar` + `DSPS`. It fits broadband photometry, spectra, or both together: in a joint fit, one physical model of the host galaxy and the AGN describes the photometry and the spectrum at once.

**Documentation: [https://jaxsedfit.readthedocs.io/](https://jaxsedfit.readthedocs.io/)**

`jaxsedfit` includes:

- a JAX/NumPyro fitting engine with MAP, Laplace, variational, NUTS and nested-sampling inference, and batched fitting of whole catalogs
- `Diffstar` + `DSPS` host-galaxy modeling
- JAX ports of selected `GRAHSP` AGN, attenuation, redshifting and dust-emission components
- a differentiable quasar spectral model (tied emission lines, Fe II, Balmer continuum, custom components) shared by joint and spectrum-only fits
- an aperture-aware likelihood for photometry and spectra measured in different apertures
- `pcigale`-style SED plots

The [`jaxqsofit`](https://github.com/burke86/jaxqsofit) package provides a spectrum-focused interface to the same spectral model; `jaxsedfit` does not depend on it. Reusable spectral integrations should import the public `jaxsedfit.spectroscopy` API rather than the private `spectral_*` modules.

## Install

`jaxsedfit` requires Python 3.10 or newer. Clone this repository, then:

```bash
python -m pip install .
curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5
```

The second command downloads a continuum-only DSPS SSP template file, which you pass as `cfg.galaxy.dsps_ssp_fn`. Milky Way dereddening also needs the SFD dust maps; see the [installation page](https://jaxsedfit.readthedocs.io/en/latest/installation.html).

## Example

```python
from jaxsedfit import FitConfig, GalaxyConfig, JAXSEDFit, Observation, PhotometryData

cfg = FitConfig(
    observation=Observation(object_id="demo", redshift=0.1),
    photometry=PhotometryData(
        filter_names=["u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss"],
        fluxes=[0.22, 0.48, 0.73, 0.86, 0.91],  # mJy
        errors=[0.03, 0.04, 0.05, 0.06, 0.07],
    ),
    galaxy=GalaxyConfig(dsps_ssp_fn="tempdata.h5"),
)

fitter = JAXSEDFit(cfg)
result = fitter.fit()
fitter.plot_sed(output_path="sed_fit.png")
```

The [quickstart](https://jaxsedfit.readthedocs.io/en/latest/quickstart.html) continues from here, and the [tutorials](https://jaxsedfit.readthedocs.io/en/latest/tutorials.html) work through real examples, including joint fits of photometry and spectra. The notebooks are in [notebooks/](notebooks/).

## Validation

Every pull request that changes the model runs the Chimera validation (`.github/workflows/chimera-validation.yml`): stellar-mass, AGN-luminosity and SFR recovery for Chimeras, real COSMOS galaxies with an added SDSS quasar of known luminosity, with photometry alone and with photometry and spectra. The validation is a regression and calibration tool for this experimental port, not a demonstration of full `GRAHSP`/`CIGALE` parity. The older single-run benchmark is still available as `jaxsedfit-benchmark --output-dir benchmark_outputs --dsps-ssp-fn tempdata.h5`.

## License and provenance

`jaxsedfit` is an experimental port of parts of `CIGALE`, `GRAHSP` and `PyQSOFit`. Some model logic and several bundled resource files are derived from or closely based on `GRAHSP` / `pcigale`, which is distributed under the `CeCILL v2` license. See:

- [LICENSES/CeCILL-v2.txt](LICENSES/CeCILL-v2.txt)
- [LICENSES/THIRD_PARTY_NOTICES.md](LICENSES/THIRD_PARTY_NOTICES.md)

Bundled third-party resources under [src/jaxsedfit/resources](src/jaxsedfit/resources) include per-directory provenance notes. If you use `jaxsedfit` in scientific work, cite `CIGALE` (Boquien et al. 2019), `GRAHSP` (Buchner et al. 2024) and `PyQSOFit` (Guo, Shen & Wang 2018; Shen et al. 2019); see the [documentation](https://jaxsedfit.readthedocs.io/) for the references.
