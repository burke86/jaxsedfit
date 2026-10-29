Quickstart
==========

``jaxsedfit`` is configured through a single :class:`jaxsedfit.FitConfig`
object. It groups the observation metadata, the photometry and optional spectra,
the filters, the galaxy and AGN model options, and the likelihood, inference and
output settings. Build the config, pass it to :class:`jaxsedfit.JAXSEDFit`, and
call :meth:`jaxsedfit.JAXSEDFit.fit`.

Fit one SED
-----------

Photometric ``fluxes`` and ``errors`` are flux densities in mJy. The
``filter_names`` must match packaged filter names (see :doc:`filters`) or be
supplied explicitly through :class:`jaxsedfit.FilterSet`.

.. code-block:: python

   from jaxsedfit import (
       FitConfig,
       GalaxyConfig,
       InferenceConfig,
       JAXSEDFit,
       Observation,
       OutputConfig,
       PhotometryData,
   )

   cfg = FitConfig(
       observation=Observation(object_id="demo", redshift=0.1),
       photometry=PhotometryData(
           filter_names=["u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss"],
           fluxes=[0.22, 0.48, 0.73, 0.86, 0.91],
           errors=[0.03, 0.04, 0.05, 0.06, 0.07],
       ),
       galaxy=GalaxyConfig(dsps_ssp_fn="tempdata.h5"),
       inference=InferenceConfig(
           method="map+nuts",
           map_steps=600,
           num_warmup=100,
           num_samples=100,
       ),
       output=OutputConfig(
           output_dir="fit_outputs",
           plot_fig=False,
           save_fig=True,
           save_result=True,
       ),
   )

   fitter = JAXSEDFit(cfg)
   result = fitter.fit()
   result.summary

``dsps_ssp_fn`` must point to a valid DSPS SSP HDF5 file (see
:doc:`installation`); a missing SSP file is the most common setup failure. For a
quick smoke test, set ``cfg.inference.method = "map"`` to run the optimization
without posterior sampling.

Look at the result
------------------

.. code-block:: python

   import numpy as np

   lo, mid, hi = np.percentile(result.samples["log_stellar_mass"], [16, 50, 84])
   fitter.plot_sed(output_path="sed_fit.png")
   result.plot_corner()

``result.samples`` holds the posterior draws of every sampled parameter;
derived quantities, such as the SFR or the AGN luminosity, come from
``result.predict()``. :doc:`results` covers predictions, plots and saving, and
:doc:`outputs` lists every output with its units.

Next steps
----------

* :doc:`tutorials`: worked examples, including real photometry from VizieR and
  BandWagon, photometric redshifts, and joint fits of photometry and spectra.
* :doc:`photometry`: photometry inputs, filters, and photometry measured in
  different apertures.
* :doc:`host` and :doc:`agn`: the galaxy and AGN model options.
* :doc:`spectroscopy`: adding spectra to a fit, or fitting spectra alone.
* :doc:`likelihood`: the error model and priors.
* :doc:`inference` and :doc:`batch`: choosing a fitting method, and fitting
  catalogs.
