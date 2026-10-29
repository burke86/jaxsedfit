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

Default priors
~~~~~~~~~~~~~~

These are the priors used when ``cfg.prior_config`` leaves a field unset. "ln" and
"log-uniform" refer to the natural logarithm; stellar mass is in log10. A parameter
marked "fixed" is not fitted unless you give it a prior.

.. list-table:: Host
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Default prior
   * - ``log_stellar_mass``
     - Student-t, 5 degrees of freedom, centered at 10 with scale 2 (log10 M\ :sub:`☉`)
   * - ``log_sfh_age_gyr``
     - Log-uniform from 10\ :sup:`-0.8` Gyr (about 160 Myr) to 10 Gyr or the age of
       the Universe at the redshift, whichever is smaller
   * - ``log_sfh_tau_gyr``
     - Log-uniform from 0.1 to 10 Gyr
   * - burst (``host_sfh_model="delayed_burst"``)
     - Mass fraction log-uniform from 10\ :sup:`-4` to 0.2; age log-uniform from
       10 Myr to the smaller of the SFH age and 500 Myr; e-folding time log-uniform
       from 10 to 200 Myr
   * - ``log_sfh_ssfr`` (``sfh_prior="ssfr_uniform"``)
     - Uniform in log10 sSFR over ``galaxy.ssfr_prior_range`` (−14 to −8); see
       :doc:`host`
   * - ``gal_lgmet``, ``gal_lgmet_scatter``
     - Fixed at ``galaxy.stellar_metallicity`` (0.019) and
       ``galaxy.stellar_metallicity_scatter`` (0)
   * - ``ebv_gal``
     - Log-uniform from 0.01 to 10 mag
   * - ``dust_delta`` (``attenuation_curve="noll2009"``)
     - Normal(−0.2, 0.3) truncated to [−0.8, 0.4]
   * - ``dust_alpha`` (Dale et al. 2014 dust)
     - Uniform from 0.75 to 2.75
   * - ``dust_umin`` (Draine & Li 2007 dust)
     - Uniform from 0.1 to 25
   * - ``gal_v_kms``, ``gal_sigma_kms`` (``fit_host_kinematics``, spectra only)
     - Normal(0, 150) and HalfNormal(150), in km/s
   * - nebular ``logU``, ``zgas``, ``ne``
     - Fixed (see :doc:`host`)

.. list-table:: AGN continuum and torus
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Default prior
   * - ``log_agn_amp``
     - Normal with width 3 (ln), centered 4 below a rough estimate of
       λL\ :sub:`λ`\ (5100 Å) from the band nearest rest 5100 Å
   * - ``pl_slope``
     - Normal(−1.85, 0.6) truncated to [−2.7, −1.0]
   * - ``uv_slope``
     - Fixed at 0, as in GRAHSP
   * - ``pl_bend_loc``
     - Normal(1000, 500) Å truncated to [500, 1500] Å
   * - ``pl_bend_width``
     - Normal(5.05, 4.95) truncated to [0.1, 10]
   * - ``pl_cutoff``
     - Fixed at 10\ :sup:`5` Å
   * - ``ebv_agn``
     - Log-uniform from 0.01 to 10 mag (in addition to ``ebv_gal``; see
       :doc:`agn`)
   * - ``fcov``
     - Uniform from 0.05 to 0.95
   * - ``si``
     - Normal(0, 1)
   * - ``cool_lam``, ``hot_lam``
     - Uniform from 10 to 30 µm and from 1 to 5.5 µm
   * - ``cool_width``, ``hot_width``
     - Uniform from 0.2 to 0.65 dex
   * - ``hot_fcov``
     - Log-uniform from 0.04 to 10

.. list-table:: AGN emission features
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Default prior
   * - ``broad_lines_strength``
     - Log-uniform from 0.3 to 20 (broadband line template)
   * - ``narrow_lines_strength``
     - Log-normal around 1, width 0.5 (ln)
   * - ``broad_line_width_kms``
     - Log-normal around 3000 km/s, width 0.4 (ln), truncated to 1000–15,000 km/s
   * - ``narrow_line_width_kms``
     - Log-normal around 500 km/s, width 0.3 (ln), truncated to 100–1500 km/s
   * - ``feii_norm``
     - Log-uniform from 0.63 to 31.6, relative to the broad lines
   * - ``feii_fwhm``, ``feii_shift`` (``fit_feii_broadening``)
     - Log-normal around 3000 km/s with width 0.3 (ln), and Normal(0, 0.01) in v/c;
       otherwise fixed at 3000 km/s and 0
   * - ``balmer_norm``, ``balmer_tau``, ``balmer_vel``
     - Log-normal around 10\ :sup:`-3`, 1 and 3000 km/s, with widths 1, 0.5 and
       0.4 (ln)
   * - spectral Fe II (joint fits): ``spectral_feii_norm``, ``_fwhm``, ``_shift``
     - Log-normal around 10\ :sup:`-3` with width 2 (ln); log-normal around
       3000 km/s with width 0.4 (ln); Normal(0, 0.01) in v/c
   * - spectral Balmer continuum: ``spectral_balmer_norm``, ``_tau``, ``_vel``
     - Log-normal around 10\ :sup:`-3`, 1 and 3000 km/s, with widths 2, 0.5 and
       0.4 (ln)
   * - spectral emission lines
     - From the line table (see :doc:`agn`)

.. list-table:: Calibration and error model
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Default prior
   * - ``systematics_width``
     - ln-normal around 0.10 with width 0.05, truncated to 0.07–0.15
   * - ``agn_systematics_width``
     - Log-normal around 0.20, width 0.5 (ln)
   * - ``log_host_capture_scale_arcsec``
     - Normal(ln 3, 1), with the host-capture model on
   * - ``missing_psf_host_capture_fraction``
     - Uniform from 0 to 1
   * - ``log_spectrum_scale``
     - Normal with width ``spectrum_scale_prior_sigma_dex`` (0.1 dex)
   * - ``spectral_continuum_tilt``
     - Normal(0, 0.1), with ``agn.fit_spectrum_tilt``
   * - ``frac_jitter``
     - HalfNormal(0.02)
   * - ``frac_fe_jitter``, ``add_jitter``
     - Fixed at 0.20 and 0
   * - ``redshift`` (``redshift_mode="fit"``)
     - Normal around the input redshift with width ``redshift_err`` (at least
       0.001), truncated at zero, or the tabulated prior below

Fitting the redshift
--------------------

To fit the redshift as a photometric redshift, switch the observation mode and
optionally supply a tabulated prior:

.. code-block:: python

   cfg.observation.redshift_mode = "fit"
   cfg.prior_config.redshift.z_grid = [0.08, 0.09, 0.10, 0.11, 0.12]
   cfg.prior_config.redshift.pdf = [0.1, 0.4, 1.0, 0.4, 0.1]

The Fairall 9 photo-z tutorial (:doc:`tutorials`) shows a complete example.
