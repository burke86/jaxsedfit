Likelihood and priors
=====================

Calibration and error-model terms
---------------------------------

Besides the physical model, a fit includes terms for flux calibration, apertures and
model error. This table lists them all; the sections below and the linked pages
describe each one.

.. list-table::
   :header-rows: 1
   :widths: 24 30 22 24

   * - Term
     - What it does
     - Default
     - Where
   * - ``systematics_width``
     - Fractional model error on every photometric point
     - fitted, about 10%
     - below
   * - ``agn_systematics_width``
     - Fractional error on the AGN part of each photometric point
     - fitted, about 20%
     - below
   * - AGN variability
     - Variance from non-simultaneous photometry of a variable AGN
     - on
     - below
   * - Spectral jitter
     - Extra spectral error: a fraction of the model and of Fe II, plus a constant
     - fitted fraction
     - below
   * - Host capture
     - Fraction of the extended host seen by each aperture, PSF or fiber
     - off
     - :doc:`photometry`
   * - Spectrum scale
     - Gray scale factor per spectrum
     - on with photometry
     - :doc:`spectroscopy`
   * - Continuum tilt
     - Power-law correction to the spectral continuum's slope
     - off
     - :doc:`spectroscopy`
   * - Instrumental resolution
     - Line-spread function of each spectrum
     - from ``resolving_power``
     - :doc:`spectroscopy`
   * - Milky Way extinction
     - Dereddening with the SFD maps (``Observation.apply_mw_deredden``)
     - off
     - :doc:`installation`

Photometric likelihood
----------------------

.. code-block:: python

   cfg.likelihood.likelihood_family = "student_t"   # or "gaussian"
   cfg.likelihood.student_t_df = 5.0

The default Student-t likelihood (5 degrees of freedom) is more tolerant of outlying
catalog points than a Gaussian. Upper limits (``is_upper_limit``) use a one-sided
likelihood. With the host-capture model, the data are compared with the
aperture-captured model flux (see :doc:`photometry`).

For band :math:`b` with model flux :math:`F_b` and AGN flux :math:`F_b^{\rm AGN}`,
the variance adds these terms to the catalog error :math:`\sigma_b`:

.. math::

   \sigma_{{\rm eff},b}^2 = \sigma_b^2
   + (w_{\rm sys} F_b)^2
   + (w_{\rm AGN} F_b^{\rm AGN})^2
   + {\rm NEV}(L_{\rm bol}) \, (F_b^{\rm AGN})^2 .

``systematics_width`` (:math:`w_{\rm sys}`)
   Model error, proportional to the model flux as in GRAHSP, so that broadband
   photometry does not constrain the model more tightly than the templates justify.
   With ``fit_systematics_width=True`` (default) it is fitted with a prior near 10%,
   between 7% and 15%; otherwise it is fixed at ``systematics_width`` (0.10).

``agn_systematics_width`` (:math:`w_{\rm AGN}`)
   Extra error on the AGN contribution, for mismatch of the AGN templates. With
   ``fit_agn_systematics_width=True`` (default) it is fitted with a log-normal prior
   around ``agn_systematics_width_prior_scale`` (0.20).

``variability_uncertainty``
   For AGN variability between photometric epochs: the normalized excess variance
   of Simm et al. (2016), :math:`{\rm NEV} = 10^{-1.43 - 0.74\,(\log L_{\rm bol} -
   45)}` with :math:`L_{\rm bol}` in erg s\ :sup:`-1`, smoothly capped at
   ``agn_nev`` (0.1). On by default.

Spectral likelihood
-------------------

Spectra use a Student-t likelihood with ``spectrum_student_t_df`` degrees of
freedom (5 by default). The quoted spectral errors are not assumed correct: as in
``jaxqsofit``, three terms are added to them in quadrature,

* ``frac_jitter`` times the model, fitted (prior HalfNormal(0.02));
* ``frac_fe_jitter`` times the Fe II model, fixed at 0.20, for Fe II template
  mismatch;
* ``add_jitter``, a constant, 0 by default.

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
