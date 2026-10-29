Working with results
====================

:meth:`jaxsedfit.JAXSEDFit.fit` returns a :class:`jaxsedfit.FitResult`. Every
inference method returns the same kind of result (see :doc:`inference`).

Samples and predictions
-----------------------

.. code-block:: python

   import numpy as np

   result = fitter.fit()
   samples = result.samples        # sampled parameters, first axis = draw
   summary = result.summary
   prediction = result.predict()   # derived quantities and model components

   lo, mid, hi = np.percentile(samples["log_stellar_mass"], [16, 50, 84])
   log_sfr = prediction["log_sfr_fit"]

``result.samples`` holds the sampled parameters. Derived quantities, such as the
SFR, the AGN luminosity and the model photometry, come from ``result.predict()``:
model evaluations at the posterior draws. ``kind="plot"`` (the default) includes
the full component SEDs; ``kind="photometry"`` returns a lighter set.

Check the units in :doc:`outputs` before converting outputs. In particular,
luminosities are in W, so ``log_agn_bol_luminosity_fit + 7`` is the
base-10 logarithm in erg s\ :sup:`-1`, and most ``log_*`` parameters other than
the stellar mass, SFR and luminosities use the natural logarithm.

To request additional NumPyro sample or deterministic sites:

.. code-block:: python

   prediction = result.predict(
       kind="plot",
       required_return_sites=("ebv_gal", "ebv_agn"),
   )

Use ``extra_return_sites`` instead for a site that depends on the configuration
and may be absent. Predictions are cached per product kind, draw limit and set of
requested sites; pass ``cache=False`` to force a new evaluation. Assigning new
posterior samples to the fitter clears the cache.

Plots
-----

.. code-block:: python

   fitter.plot_sed(output_path="sed_fit.png")   # data, model and components
   result.plot_corner()
   result.plot_trace()
   fitter.plot_spectrum()                        # joint fits; see spectroscopy

``plot_sed`` overlays the observed and model photometry on the total model and its
components: the host, host dust, AGN disk, torus, emission lines, Fe II and Balmer
continuum. ``jaxsedfit.plotting.plot_fit_sed(fitter, output_path=...)`` does the
same as a function. The component spectra are computed the first time a plot or
prediction needs them.

Fit quality
-----------

Every fit reports ``sed_chi2``, ``sed_n_eff`` and ``sed_reduced_chi2`` for the
photometry, the ``spectroscopy_*`` equivalents for spectra, and ``joint_*`` sums.
They are standardized-residual diagnostics, not the likelihood; :doc:`outputs`
defines them.

Saving and loading
------------------

With ``cfg.output.save_result = True``, or by calling :meth:`jaxsedfit.FitResult.save`,
a fit is written to an HDF5 bundle named ``<object_id>_samples.h5``. It holds the
fit configuration, the posterior samples and summary metadata, which is enough to
resume or re-analyze the fit. Predictions are not stored; they are recomputed when
first requested after loading.

.. code-block:: python

   import jaxsedfit

   result.save("fit_outputs")
   loaded = jaxsedfit.load_result("fit_outputs/demo_samples.h5")
   loaded.predict()["log_sfr_fit"]

``jaxsedfit.load(...)`` returns the fitter instead of the result. Keep the
configuration with the samples when you use a fit scientifically: it records which
components, priors and fixed values gave meaning to the output names.
