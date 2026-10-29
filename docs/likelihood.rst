Likelihood and priors
=====================

Photometric likelihood
----------------------

.. code-block:: python

   cfg.likelihood.likelihood_family = "student_t"
   cfg.likelihood.student_t_df = 5.0
   cfg.likelihood.systematics_width = 0.10
   cfg.likelihood.fit_systematics_width = True

``likelihood_family``
   The default is the heavier-tailed ``"student_t"`` likelihood with
   ``student_t_df=5``. ``"gaussian"`` uses a normal likelihood.

``systematics_width``
   A fractional model-error floor, added in quadrature as ``systematics_width``
   times the model flux in each band (as in GRAHSP), so that broadband photometry
   does not constrain the model more tightly than the templates justify. Fix it for
   simpler sampling, or fit it with ``fit_systematics_width=True``.

``variability_uncertainty``
   Adds an AGN-variability term to the photometric uncertainty when AGN emission is
   present.

Upper limits (``is_upper_limit``) use a one-sided likelihood. With
``use_host_capture_model``, the likelihood compares the data with the
aperture-captured model flux described in :doc:`photometry`.

Spectral likelihood
-------------------

Spectra use a Student-t likelihood with ``spectrum_student_t_df`` degrees of
freedom (5 by default). The quoted spectral errors are not assumed correct: as in
``jaxqsofit``, three terms are added in quadrature, a fitted fraction of the model
(``frac_jitter``, prior HalfNormal(0.02)), a fixed fraction of the Fe II model
(``frac_fe_jitter``, 0.20) and a constant (``add_jitter``, 0 by default).
``spectroscopy_chi2`` and the related outputs use the same variance (see
:doc:`outputs`).

Priors
------

Priors are NumPyro distributions set through :class:`jaxsedfit.PriorConfig`. Only
the fields you set change; everything else keeps its default.

.. code-block:: python

   import numpy as np
   from numpyro import distributions as dist

   cfg.prior_config.stellar_mass = dist.Normal(10.5, 0.5)
   cfg.prior_config.host.log_ebv_gal = dist.TruncatedNormal(
       loc=np.log(0.08), scale=0.5, low=np.log(1.0e-4), high=np.log(2.0)
   )
   cfg.prior_config.agn.log_fcov = dist.TruncatedNormal(
       loc=np.log(0.3), scale=0.5, low=np.log(0.01), high=np.log(0.95)
   )
   cfg.prior_config.likelihood.log_systematics_width = dist.TruncatedNormal(
       loc=np.log(0.10), scale=0.15, low=np.log(0.05), high=np.log(0.20)
   )

``log_*`` parameters other than ``log_stellar_mass`` use the natural logarithm (see
:doc:`outputs`).

Fitting the redshift
--------------------

To fit the redshift as a photometric redshift, switch the observation mode and
optionally supply a tabulated prior:

.. code-block:: python

   cfg.observation.redshift_mode = "fit"
   cfg.prior_config.redshift.z_grid = [0.08, 0.09, 0.10, 0.11, 0.12]
   cfg.prior_config.redshift.pdf = [0.1, 0.4, 1.0, 0.4, 0.1]

The Fairall 9 photo-z tutorial (:doc:`tutorials`) shows a complete example.
