Fitting Interface
=================

.. currentmodule:: jaxsedfit

.. autoclass:: JAXSEDFit
   :members: fit, fit_map, fit_laplace, fit_svi, fit_nuts, fit_ns, predict, save, load, load_result, plot_sed, plot_corner, plot_trace, plot_spectrum
   :show-inheritance:

.. autoclass:: FitResult
   :members: predict, save, plot_corner, plot_trace

.. autoclass:: PredictionResult
   :members: median, keys, items, get

.. autofunction:: load_from_samples

Batched fitting
---------------

See :doc:`../batch`.

.. autofunction:: fit_batch

.. autoclass:: BatchFitResult
   :members: object_ids, summary_table

.. autoclass:: jaxsedfit.batched.BatchedInputs
   :members: stack, inputs, model

.. autofunction:: jaxsedfit.batched.make_batched_fitter

Gaussian approximations
-----------------------

The functions behind ``"laplace"`` and ``"svi"``, shared by single-object and
batched fits.

.. automodule:: jaxsedfit.approximate
   :members: laplace_draws, gaussian_vi, latent_sites, site_transforms
