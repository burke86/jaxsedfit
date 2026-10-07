Posterior output reference
==========================

This page describes the posterior parameters and derived outputs of
``JAXSEDFit`` photometric and joint photometric/spectral fits. Outputs depend
on the selected host, AGN, nebular, and spectral components and their priors.
A disabled component can produce zeros or fixed defaults; such an output is
not a measurement of that component.

Access and array conventions
----------------------------

.. code-block:: python

   import numpy as np

   result = fitter.fit()
   samples = result.samples
   prediction = result.predict(kind="plot")
   mass_draws = 10**samples["log_stellar_mass"]
   mass_p16, mass_median, mass_p84 = np.percentile(mass_draws, [16, 50, 84])

``result.samples`` holds inference samples. Derived quantities are available
through ``result.predict()``; ``kind="plot"`` includes full component SEDs,
while ``kind="photometry"`` uses the lighter prediction payload. Predictions
are model evaluations at posterior draws, rather than new noisy observations.
The saved HDF5 ``samples`` group retains the latent sites needed to resume a
fit; regenerate component predictions after loading. MAP (``optax``) fits
provide a point estimate, not a sampled posterior.

Scalar outputs normally have shape ``(draw,)``. Photometric arrays have shape
``(draw, measurement)`` in input photometry order, including repeated filters.
Spectral arrays have shape ``(draw, pixel)`` with input spectra concatenated;
``spec_spectrum_index`` identifies the input spectrum for each pixel.
Per-spectrum quantities have shape ``(draw, spectrum)``; line arrays use a
component or tie-group axis. Full SEDs use the associated wavelength axis.
Fixed grids and masks may be repeated across draws without having uncertainty.

Units and logarithms
--------------------

* Wavelengths are in angstroms (Å), except ``hot_lam`` and ``cool_lam`` in µm.
* Observed flux densities, including filter-integrated flux densities, are
  :math:`f_\nu` in mJy (1 mJy = :math:`10^{-29}` W m\ :sup:`-2` Hz\ :sup:`-1`).
  These are not wavelength-integrated line fluxes.
* Rest SEDs are :math:`L_\lambda` in W Å\ :sup:`-1`; luminosities are in W.
  Stellar masses are in solar masses, SFRs in solar masses per year, and ages
  in Gyr. Velocities are in km s\ :sup:`-1`.
* ``log_stellar_mass`` and the derived ``log_*_luminosity_fit`` and
  ``log_sfr_fit`` use **base 10**. ``gal_lgmet`` is also a base-10 coordinate.
* Other SED ``log_*`` parameters use the **natural logarithm** of the numerical
  value in the listed unit: recover the value with ``np.exp``. This includes
  ``log_agn_amp``, SFH times/fractions, positive component parameters, aperture
  size, and spectral calibration. Do not apply ``10**`` to these sites.
* ``_fit`` means the value used by the model, often a derived or fixed value;
  it does not imply that the quantity was independently sampled. Log outputs
  for zero luminosity/SFR are numerically floored and should not be interpreted
  as detections.

Host galaxy and star formation
------------------------------
.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``log_stellar_mass``
     - log10(M / Msun)
     - Present surviving stellar mass used to normalize the host population, rather than total mass ever formed.
   * - ``formed_stellar_mass``
     - Msun
     - Total mass formed before stellar mass loss.
   * - ``surviving_mass_fraction``
     - fraction
     - Surviving mass divided by formed mass.
   * - ``log_sfh_age_gyr``, ``sfh_age_gyr_fit``
     - ln(Gyr), Gyr
     - Duration of the delayed SFH since onset; not the mass-weighted stellar age.
   * - ``log_sfh_tau_gyr``, ``sfh_tau_gyr_fit``
     - ln(Gyr), Gyr
     - Main delayed SFH timescale; SFR is proportional to t exp(-t/tau).
   * - ``log_sfh_tau_over_age``
     - ln(ratio)
     - Natural logarithm of main SFH timescale divided by age.
   * - ``log_sfh_burst_fraction``, ``sfh_burst_fraction_fit``
     - ln(fraction), fraction
     - Burst fraction of formed stellar mass in the delayed-burst model.
   * - ``log_sfh_burst_age_gyr``, ``sfh_burst_age_gyr_fit``
     - ln(Gyr), Gyr
     - Time since the burst began.
   * - ``log_sfh_burst_tau_gyr``, ``sfh_burst_tau_gyr_fit``
     - ln(Gyr), Gyr
     - Exponential burst timescale.
   * - ``log_sfr_fit``
     - log10(Msun / yr)
     - Current SFR at the observation time.
   * - ``gal_lgmet``, ``gal_lgmet_fit``
     - dex
     - Mean stellar metallicity in the configured SSP coordinate: log10(Z) or log10(Z/Zsun). Check galaxy.ssp_metallicity_coordinate and ssp_solar_metallicity.
   * - ``gal_lgmet_scatter``, ``gal_lgmet_scatter_fit``
     - dex
     - Width of the stellar metallicity distribution; log_gal_lgmet_scatter is its natural logarithm.
   * - ``gal_v_kms``
     - km / s
     - Host line-of-sight velocity offset.
   * - ``gal_sigma_kms``
     - km / s
     - Host intrinsic velocity dispersion (Gaussian sigma, not FWHM).
   * - ``mass_metallicity_relation_logprior``
     - log probability
     - Optional soft mass-metallicity prior contribution; zero when disabled.
   * - ``u_lgmcrit``, ``u_lgy_at_mcrit``, ``u_indx_lo``, ``u_indx_hi``
     - dimensionless
     - Unbounded Diffstar main-sequence coordinates for critical halo mass, efficiency, low/high mass slopes. These are transformed model coordinates, not direct physical measurements.
   * - ``u_lg_qt``, ``u_qlglgdt``, ``u_lg_drop``, ``u_lg_rejuv``
     - dimensionless
     - Diffstar quenching/rejuvenation coordinates, when exposed by the installed Diffstar parameter tuple. The active keys follow DEFAULT_DIFFSTAR_U_PARAMS._fields.
   * - ``host_age_weights``
     - fraction per age bin
     - Formed-mass weights over SSP ages.
   * - ``host_lgmet_weights``
     - fraction per metallicity bin
     - Weights over SSP metallicities.
   * - ``host_ssp_weights``
     - fraction per metallicity × age bin
     - Combined SSP weights used to synthesize the host.
   * - ``gal_sfr_table``
     - Msun / yr
     - SFH evaluated on the host cosmic-time grid (host_basis.gal_t_table, in Gyr for a fixed-redshift fit).
   * - ``gal_smh_table``
     - Msun
     - Cumulative formed stellar mass on that same time grid.

AGN continuum, torus, and native SED features
---------------------------------------------

For the positive parameters below, a ``log_<name>`` site can replace or
accompany the linear site, depending on the prior configuration. It uses ln
of the numerical value in the listed unit. Native SED template strengths
and joint spectral amplitudes have different normalizations.

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``log_agn_amp``, ``log_agn_amp_fit``
     - ln(W)
     - AGN normalization A; the disk is constructed using A / 5100 as its 5100 Å L_lambda normalization.
   * - ``log_disk_luminosity_fit``
     - log10(W / Å)
     - log10(A / 5100). Despite its name this is L_lambda at the disk normalization wavelength, not lambda L_lambda or an integrated disk luminosity.
   * - ``log_agn_bol_luminosity_fit``
     - log10(W)
     - A times the fixed AGN_BOLOMETRIC_CORRECTION_5100; a bolometric proxy, not an integral of the fitted SED.
   * - ``fracAGN_5100_fit``
     - fraction
     - Intrinsic AGN fraction from A / 5100 and host L_lambda(5100 Å), clipped to [0.0001, 0.999].
   * - ``pl_slope``, ``uv_slope``
     - dimensionless
     - Long- and short-wavelength power-law slope parameters of the bent disk continuum in L_lambda.
   * - ``pl_bend_loc``, ``pl_cutoff``
     - Å
     - Disk bend wavelength and fixed long-wavelength cutoff scale.
   * - ``pl_bend_width``
     - dimensionless
     - Smoothness of the disk power-law bend.
   * - ``fcov``
     - dimensionless
     - Torus normalization factor: L_lambda at the torus normalization wavelength is set from 2.5 A fcov / wavelength. Not an integrated torus/disk luminosity ratio.
   * - ``si``
     - dimensionless
     - Signed silicate-feature control, mapped through tanh to a bounded fractional modulation.
   * - ``cool_lam``, ``hot_lam``
     - µm
     - Centers of cool/hot torus bumps in logarithmic wavelength.
   * - ``cool_width``, ``hot_width``
     - dex
     - Bump width w in exp(-((log10(lambda/µm)-log10(center))/w)^2); w is sqrt(2) times the Gaussian sigma in dex.
   * - ``hot_fcov``
     - dimensionless
     - Relative hot-bump weight before common torus normalization.
   * - ``broad_lines_strength``, ``narrow_lines_strength``
     - dimensionless
     - Native broad/narrow emission-line template strength multipliers.
   * - ``broad_line_width_kms``, ``narrow_line_width_kms``
     - km / s
     - FWHM of native SED line templates.
   * - ``feii_norm``
     - dimensionless
     - Native Fe II strength relative to the broad-line normalization.
   * - ``feii_fwhm``
     - km / s
     - Native Fe II broadening FWHM.
   * - ``feii_shift``
     - dimensionless
     - Native Fe II shift coordinate; applied as velocity c times shift in ln wavelength.
   * - ``balmer_norm``
     - dimensionless
     - Native Balmer-continuum strength relative to the disk L_lambda normalization.
   * - ``balmer_tau``
     - dimensionless
     - Balmer optical depth at the continuum edge.
   * - ``balmer_vel``
     - km / s
     - Balmer-continuum broadening velocity scale.
   * - ``agn_variability_nev``
     - dimensionless
     - Luminosity-dependent normalized excess variance used by the AGN variability likelihood.

Attenuation, host dust, and nebular emission
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``ebv_gal``, ``ebv_agn``
     - mag
     - Foreground host E(B-V) and additional nuclear E(B-V). The disk and native AGN lines use their sum; the torus uses the foreground screen.
   * - ``dust_alpha``, ``dust_alpha_fit``
     - dimensionless
     - Dale host-dust heating-distribution index when using the Dale model.
   * - ``dust_umin``, ``dust_umin_fit``
     - dimensionless
     - DL07 minimum radiation-field intensity relative to the reference interstellar field.
   * - ``log_dust_luminosity_fit``
     - log10(W)
     - Host dust luminosity from absorbed stellar/nebular/AGN light and nebular dust absorption; sets energy-balance emission.
   * - ``nebular_logU``, ``nebular_logU_fit``
     - log10(U)
     - Ionization parameter.
   * - ``nebular_zgas``, ``nebular_zgas_fit``
     - mass fraction
     - Absolute gas metallicity Z, not Z/Zsun.
   * - ``nebular_ne``, ``nebular_ne_fit``
     - cm^-3
     - Electron density of the nebular template.
   * - ``nebular_f_esc``, ``nebular_f_esc_fit``
     - fraction
     - Fraction of ionizing photons that escape.
   * - ``nebular_f_dust_fraction``, ``nebular_f_dust_fraction_fit``
     - fraction
     - Dust-absorbed fraction of the non-escaping ionizing photons.
   * - ``nebular_f_dust_fit``
     - fraction
     - Absolute dust-absorbed fraction: (1-f_esc) times f_dust_fraction.
   * - ``nebular_lines_width``, ``nebular_lines_width_fit``
     - km / s
     - Nebular line FWHM.
   * - ``log_nebular_line_scale``, ``nebular_line_scale_fit``
     - ln(scale), scale
     - Extra nebular line multiplier; unity unless a log-scale prior is supplied.
   * - ``nebular_corr_fit``
     - dimensionless
     - CIGALE correction for the escaping and dust-absorbed ionizing photons.
   * - ``nebular_n_ly_young_fit``, ``nebular_n_ly_old_fit``
     - photons / s
     - Ionizing photon production from populations below/above the configured young-age cutoff.

Aperture, calibration, and likelihood outputs
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``redshift``, ``redshift_fit``
     - dimensionless
     - Sampled redshift and the redshift used in model predictions; fixed redshift appears only as the derived value.
   * - ``log_host_capture_scale_arcsec``, ``log_host_capture_scale_arcsec_fit``
     - ln(arcsec)
     - Characteristic size in the host capture model.
   * - ``host_capture_slope_fit``
     - dimensionless
     - Capture-model slope; currently fixed at 2.
   * - ``missing_psf_host_capture_fraction``
     - fraction
     - Independent capture fractions for partial photometric measurements missing a valid spatial scale, in missing-measurement order.
   * - ``host_capture_fraction_fluxes``
     - fraction per measurement
     - Host/extended-source fraction captured by each photometric aperture.
   * - ``spectrum_host_capture_fraction``
     - fraction per spectrum
     - Host/extended-source fraction captured by each spectrum.
   * - ``log_spectrum_scale``, ``log_spectrum_scale_fit``, ``spectrum_scale_fit``
     - ln(scale), ln(scale), scale
     - Per-spectrum multiplicative calibration; spectrum_scale_fit = exp(log_spectrum_scale_fit).
   * - ``spectral_feature_amplitude_scale``
     - dimensionless
     - Reference calibration converting sampled observed feature amplitudes to the shared source coordinate.
   * - ``systematics_width``, ``agn_systematics_width``
     - fraction
     - Extra photometric uncertainty scales; log_systematics_width and log_agn_systematics_width use ln. The AGN term scales with the variable AGN contribution.
   * - ``spectroscopy_likelihood_weight``
     - dimensionless
     - Resolution-dependent spectroscopy log-likelihood weight.
   * - ``spectroscopy_loglike``
     - log probability
     - Spectroscopic log-likelihood contribution.
   * - ``sed_chi2``, ``spectroscopy_chi2``, ``joint_chi2``
     - dimensionless
     - Sum of squared standardized residuals for the photometry, spectra, and their sum.
   * - ``sed_n_eff``, ``spectroscopy_n_eff``, ``joint_n_eff``
     - effective count
     - Effective data counts used for each diagnostic.
   * - ``sed_reduced_chi2``, ``spectroscopy_reduced_chi2``, ``joint_reduced_chi2``
     - dimensionless
     - Corresponding chi2 divided by n_eff. These do not subtract the number of fitted parameters; they are diagnostics, not formal chi-square significance tests.
   * - ``transmitted_fraction_fluxes``
     - fraction per measurement
     - Band-projected attenuated/intrinsic direct-light ratio. Lightweight paths may return unity when this ratio is not needed.

Component arrays and wavelength grids
-------------------------------------

All ``*_fluxes`` below are filter-projected **flux densities in mJy**.
All ``*_obs_sed`` are observed-frame **f_nu in mJy**; all ``*_rest_sed``
are rest-frame **L_lambda in W / Å**. Component outputs generally include
attenuation; observed SEDs also include redshift/distance and IGM transmission.
The aperture/captured outputs explicitly identify spatial losses.

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``total_rest_sed``, ``total_obs_sed``
     - by suffix above
     - All source components.
   * - ``agn_rest_sed``, ``agn_fluxes``, ``agn_obs_sed``
     - by suffix above
     - Combined AGN.
   * - ``host_rest_sed``, ``host_fluxes``, ``host_obs_sed``
     - by suffix above
     - Attenuated stellar host including nebular absorption.
   * - ``host_total_fluxes``, ``host_total_rest_sed``, ``host_total_obs_sed``
     - by suffix above
     - Stellar host plus nebular emission.
   * - ``host_absorbed_rest_sed``
     - by suffix above
     - Absorbed direct light used in energy balance.
   * - ``dust_rest_sed``, ``dust_fluxes``, ``dust_obs_sed``
     - by suffix above
     - Host dust emission.
   * - ``nebular_rest_sed``, ``nebular_fluxes``, ``nebular_obs_sed``
     - by suffix above
     - Nebular lines plus continuum.
   * - ``nebular_lines_rest_sed``, ``nebular_lines_fluxes``, ``nebular_lines_obs_sed``
     - by suffix above
     - Nebular emission lines.
   * - ``nebular_continuum_rest_sed``, ``nebular_continuum_fluxes``, ``nebular_continuum_obs_sed``
     - by suffix above
     - Nebular continuum.
   * - ``nebular_absorption_rest_sed``
     - by suffix above
     - Nebular ionizing-light absorption correction.
   * - ``disk_rest_sed``, ``disk_fluxes``, ``disk_obs_sed``
     - by suffix above
     - AGN disk.
   * - ``torus_rest_sed``, ``torus_fluxes``, ``torus_obs_sed``
     - by suffix above
     - AGN torus.
   * - ``feii_rest_sed``, ``feii_fluxes``, ``feii_obs_sed``
     - by suffix above
     - Native AGN Fe II.
   * - ``line_rest_sed``, ``line_fluxes``, ``line_obs_sed``
     - by suffix above
     - Native AGN emission lines.
   * - ``line_bl_rest_sed``, ``line_bl_fluxes``, ``line_bl_obs_sed``
     - by suffix above
     - Native broad AGN lines.
   * - ``line_nl_rest_sed``, ``line_nl_fluxes``, ``line_nl_obs_sed``
     - by suffix above
     - Native narrow AGN lines.
   * - ``line_liner_rest_sed``, ``line_liner_fluxes``, ``line_liner_obs_sed``
     - by suffix above
     - Native LINER lines.
   * - ``balmer_rest_sed``, ``balmer_fluxes``, ``balmer_obs_sed``
     - by suffix above
     - Native AGN Balmer continuum.

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``pred_fluxes``
     - mJy
     - Total predicted measurement including aperture losses and shared spectral-feature photometry.
   * - ``variable_agn_fluxes``
     - mJy
     - AGN contribution treated as variable in the photometric likelihood.
   * - ``constant_agn_fluxes``
     - mJy
     - AGN contribution treated as constant in the photometric likelihood.
   * - ``host_total_fluxes``
     - mJy
     - Total host stellar plus nebular photometry before capture losses.
   * - ``host_capture_source_fluxes``
     - mJy
     - Host stellar/nebular plus host-dust photometry before capture.
   * - ``captured_host_dust_fluxes``
     - mJy
     - Host dust after aperture capture.
   * - ``agn_narrow_line_fluxes_total``
     - mJy
     - Total native AGN narrow-line photometry used by the capture model.
   * - ``captured_agn_narrow_line_fluxes``
     - mJy
     - Native AGN narrow lines after capture.
   * - ``extended_capture_source_fluxes``
     - mJy
     - Combined extended-source photometry before capture.
   * - ``captured_extended_source_fluxes``
     - mJy
     - Extended-source photometry after capture.
   * - ``rest_wave``
     - Å
     - Rest SED grid.
   * - ``obs_wave``
     - Å
     - Observed SED grid.
   * - ``spectral_line_photometry``
     - mJy
     - Line contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_feii_photometry``
     - mJy
     - Feii contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_extrapolated_feii_photometry``
     - mJy
     - Extrapolated feii contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_balmer_photometry``
     - mJy
     - Balmer contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_extrapolated_broad_photometry``
     - mJy
     - Extrapolated broad contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_extrapolated_narrow_photometry``
     - mJy
     - Extrapolated narrow contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_line_obs_sed``
     - mJy
     - Line contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_feii_obs_sed``
     - mJy
     - Feii contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_balmer_obs_sed``
     - mJy
     - Balmer contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``spectral_custom_continuum_obs_sed``
     - mJy
     - Custom continuum contribution from shared spectral features; photometry uses measurement order, obs_sed uses obs_wave. Extrapolated outputs describe features projected beyond the fitted spectral coverage.
   * - ``total_agn_lines_local_obs_wave``
     - Å
     - Locally refined observed wavelength grid for the named emission-line contribution.
   * - ``total_agn_lines_local_obs_sed``
     - mJy
     - Observed flux density on the matching ``*_local_obs_wave`` grid; resolves lines missed by the coarse SED grid.
   * - ``agn_lines_local_obs_wave``
     - Å
     - Locally refined observed wavelength grid for the named emission-line contribution.
   * - ``agn_lines_local_obs_sed``
     - mJy
     - Observed flux density on the matching ``*_local_obs_wave`` grid; resolves lines missed by the coarse SED grid.
   * - ``nebular_lines_local_obs_wave``
     - Å
     - Locally refined observed wavelength grid for the named emission-line contribution.
   * - ``nebular_lines_local_obs_sed``
     - mJy
     - Observed flux density on the matching ``*_local_obs_wave`` grid; resolves lines missed by the coarse SED grid.

   * - ``total_local_lines_obs_wave``
     - Å
     - Refined observed wavelength grid around nebular emission lines.
   * - ``total_local_lines_obs_sed``
     - mJy
     - Total source flux density on ``total_local_lines_obs_wave``.

Joint spectral parameters and predictions
-----------------------------------------

The prefix is ``spectral`` for joint-fit features. Simple-line names follow
the configured line names; tied-line arrays follow the velocity, width, or
amplitude groups in the line metadata. Prefer ``result.spectrum`` for named
physical components. Shared line amplitudes describe the source before
per-spectrum calibration and aperture capture.

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``spectral_continuum_tilt``
     - dimensionless
     - Additional continuum power-law tilt.
   * - ``spectral_feii_norm``, ``spectral_balmer_norm``
     - mJy
     - Observed normalization at the configured f_nu pivot, transformed to the shared source coordinate using spectral_feature_amplitude_scale.
   * - ``spectral_feii_fwhm``, ``spectral_balmer_vel``
     - km / s
     - Fe II FWHM and Balmer broadening scale.
   * - ``spectral_feii_shift``
     - dimensionless
     - Fe II ln-wavelength shift coordinate (velocity approximately c times shift).
   * - ``spectral_balmer_tau``
     - dimensionless
     - Balmer edge optical depth.
   * - ``spectral_line_amp_<name>``
     - mJy
     - Simple-line sampled peak amplitude in the reference observed calibration coordinate.
   * - ``spectral_line_fwhm_<name>``, ``spectral_line_velocity_<name>``
     - km / s
     - Simple-line FWHM and velocity offset.
   * - ``spectral_line_dmu_group``, ``spectral_line_dmu_independent_group``
     - ln wavelength offset
     - Tied-line center shifts from rest wavelengths.
   * - ``spectral_line_log_fwhm_group``
     - ln(km / s)
     - Tied-group intrinsic FWHM.
   * - ``spectral_line_sig_group``
     - dimensionless
     - Tied-group Gaussian sigma in ln wavelength.
   * - ``spectral_line_amp_group``
     - mJy
     - Tied-group peak normalization before component ratios and calibration conversion.
   * - ``spectral_line_amp_per_component``
     - mJy
     - Shared peak amplitude for each expanded Gaussian component after tie ratios.
   * - ``spectral_line_mu_per_component``
     - ln(Å)
     - Natural logarithm of each rest-frame line center.
   * - ``spectral_line_sig_per_component``
     - dimensionless
     - Intrinsic Gaussian sigma in ln wavelength.
   * - ``spectral_line_broad_mask_per_component``
     - 0 or 1
     - Fixed broad-component indicator.
   * - ``spectral_line_narrow_fwhm_kms``, ``spectral_line_narrow_amp_scale``
     - km / s, scale
     - Narrow-component width diagnostic and configured amplitude multiplier.
   * - ``pred_spectrum_fluxes``, ``spec_continuum_model_fluxes``, ``spec_host_model_fluxes``, ``spec_disk_model_fluxes``, ``spec_torus_model_fluxes``
     - mJy per pixel
     - Calibrated spectrum prediction and its continuum, host, disk, and torus contributions.
   * - ``spectral_continuum_model``, ``spectral_line_model``, ``spectral_line_model_broad``, ``spectral_line_model_narrow``, ``spectral_feii_model``, ``spectral_balmer_model``, ``spectral_total_model``
     - mJy per pixel
     - Shared feature models on the spectral grid, before final per-spectrum calibration.
   * - ``spectral_line_model_aperture``, ``spectral_line_model_narrow_aperture``
     - mJy per pixel
     - Line contribution after extended narrow-line aperture capture.
   * - ``spec_wave_obs``
     - Å
     - Concatenated observed spectral wavelength grid.
   * - ``spec_spectrum_index``
     - integer
     - Zero-based spectrum identity for each concatenated pixel.

Sampler and custom-component sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exact reparameterizations can expose auxiliary ``*_std`` sites (including
``spectral_line_nlr_center_std``, high-ionization/coronal offsets, broad-center
and relative-offset coordinates), ordered-width coordinates, and local
amplitude coordinates. Physical hierarchy outputs include
``spectral_line_nlr_center``, ``spectral_line_high_ion_center``,
``spectral_line_coronal_center``, ``spectral_line_broad_center_value_<index>``,
and ``spectral_line_broad_relative_offsets_<index>`` (ln-wavelength offsets),
``spectral_line_log_broad_fwhm`` and ``spectral_line_log_narrow_fwhm``
(ln(km/s)), and ``spectral_line_log_fwhm_delta_group`` (ln width ratios). These are dimensionless sampler coordinates with
configured transformations, not extra independent physical observables.
``spectral_line_amp_<complex>`` uses mJy in the sampled reference calibration
coordinate; ``spectral_line_ordered_width_logits_<label>`` contains
dimensionless coordinates enforcing width order. Direct widths use
``spectral_line_<label>_log_fwhm`` and
``spectral_line_<low/high>_ion_log_fwhm`` in ln(km/s).
Interpret their physical ``spectral_line_*`` outputs instead. Group and
hierarchy indices are configuration-dependent.

Custom components emit ``spectral_<component.site_name(parameter)>`` sample
sites and ``spectral_<component.deterministic_site_name>`` model arrays.
Parameter units and definitions are supplied by the component author; its
joint-fit model arrays must use mJy on the requested wavelength grid. A finite
list of custom parameter names is therefore not possible.

Unit-explicit spectral results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``result.spectrum.lines[name]`` exposes the following posterior fields:

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Output
     - Unit / coordinate
     - Definition
   * - ``amplitude_mjy``
     - mJy
     - Shared Gaussian peak f_nu amplitude.
   * - ``center_rest_angstrom``
     - Å
     - Rest-frame center, exp(spectral_line_mu_per_component).
   * - ``sigma_ln_lambda``
     - dimensionless
     - Gaussian sigma in ln wavelength.
   * - ``fwhm_kms``
     - km / s
     - 2.354820045 times c times sigma_ln_lambda.
   * - ``velocity_offset_kms``
     - km / s
     - c times ln(center / nominal rest wavelength); positive is redward.
   * - ``flux_w_m2``
     - W / m^2
     - Integrated observed line flux, including the sampled redshift conversion; before per-spectrum calibration/capture.
   * - ``total_flux_w_m2``
     - W / m^2
     - Sum of component integrated fluxes in result.spectrum.line_groups[name].

The line flux integrates the Gaussian in ln wavelength as f_nu over
observed frequency; it is not peak amplitude times FWHM in angstroms.
``parent_line``, ``component_index``, ``kind``, ``rest_wavelength_angstrom``,
and group ``component_names`` are fixed metadata.

``result.spectrum.observations`` provides one result per input spectrum.
``model_flux_mjy``, ``continuum_flux_mjy``, ``host_flux_mjy``,
``disk_flux_mjy``, ``torus_flux_mjy``, ``line_flux_density_mjy``,
``feii_flux_density_mjy``, ``balmer_flux_density_mjy``, and ``residual_mjy``
are posterior arrays in mJy; residuals are observed minus model.
``observed_flux_mjy``, ``error_mjy``, ``mask``, ``index``, ``instrument``,
and ``wavelength_obs_angstrom`` are input data/metadata, not posteriors.
The top-level spectral result exposes the concatenated model, continuum,
line, Fe II, and Balmer arrays with the same units.

Standalone spectral backend
---------------------------

The lower-level ``quasar_spectral_model`` / ``qso_fsps_joint_model`` backend
also exported by this package uses a separate standalone convention:
rest-frame **f_lambda in units of 10^-17 erg s^-1 cm^-2 Å^-1**.
Its ``model``, ``continuum_model``, ``agn_model``, ``gal_model``,
``gal_model_total``, ``gal_model_intrinsic``, ``gal_model_intrinsic_total``,
``f_pl_model``, ``f_fe_mgii_model``, ``f_fe_balmer_model``, ``f_bc_model``,
``f_poly_model``, ``line_model`` (including broad/narrow and intrinsic
variants), ``line_component_profiles``, and their ``*_psf`` variants use
these units. ``sigma_tot`` is the total spectral uncertainty in the same
units. Do not mix these arrays with joint-fit mJy arrays.

Standalone continuum parameters include ``PL_norm`` (f_lambda normalization),
``PL_slope`` (dimensionless), ``ebv`` and ``reddening_a2500`` (mag),
``host_amp`` (template normalization), ``frac_host`` (fraction), and
``log_frac_host`` (ln fraction). ``Fe_uv_norm`` / ``Fe_op_norm`` and
``Balmer_norm`` normalize their templates (in the standalone flux-density convention);
``Balmer_Tau`` is the dimensionless edge optical depth and ``Balmer_vel``
the broadening scale in km/s. Their ``log_*`` variants use ln; ``Fe_FWHM``, ``Fe_uv_FWHM``,
``Fe_op_FWHM`` use km/s, while ``Fe_shift``, ``Fe_uv_shift``, and
``Fe_op_shift`` are ln-wavelength shift coordinates. Template and polynomial
parameter names depend on the supplied prior mapping. ``fsps_weights`` are
template amplitudes and ``fsps_weights_frac`` their normalized fractions.
``host_aperture_scale`` is a dimensionless host multiplier and
``log_host_aperture_scale`` its natural logarithm;
``gal_sigma_effective_kms`` includes effective host broadening.

Standalone ``line_amp_per_component`` uses the f_lambda unit above;
``line_mu_per_component`` is ln(rest wavelength / Å), and
``line_sig_per_component`` is Gaussian sigma in ln wavelength.
``line_amp_effective_per_component`` and ``line_sig_effective_per_component``
include instrumental broadening. Tied-group and sampler-coordinate names
follow the same ``line_*`` families as above without the ``spectral_`` prefix.
``delta_m_psf`` is a magnitude offset, ``scale_psf`` its flux scale,
``eta_psf`` a host fraction, and their ``*_raw`` sites are latent coordinates.
``sigma_phot_extra`` is extra photometric scatter in mag in the backend's magnitude
likelihood. ``spectral_likelihood_weight`` is dimensionless.
``log_lambda_Llambda_<wavelength>_agn`` is log10(lambda L_lambda / erg s^-1)
at the named rest wavelength. Physical-host fits can also expose
``sfh_age_gyr``, ``sfh_tau_gyr``, ``formed_stellar_mass``,
``surviving_mass_fraction``, and ``mass_metallicity_relation_logprior``
with the host units defined above.

For scientific use, retain the fit configuration with the samples: it records
which components, metallicity convention, template normalizations, line ties,
and fixed values gave meaning to the output names.
