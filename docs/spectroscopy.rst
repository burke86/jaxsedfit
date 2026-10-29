Spectroscopy
============

Spectra are added with :class:`jaxsedfit.SpectroscopyData`; there is no separate
switch. A spectrum can be fitted together with photometry (a joint fit) or on its
own. The :doc:`tutorials` include a joint fit of a Chimera, whose true host and AGN
spectra are known, and of an SDSS quasar with BandWagon photometry.

Adding a spectrum
-----------------

Spectroscopic ``fluxes`` and ``errors`` are observed-frame flux densities in mJy on
the ``wave_obs`` grid (observed-frame vacuum wavelengths in Å):

.. code-block:: python

   from jaxsedfit import SpectroscopyData

   cfg.spectroscopy = SpectroscopyData(
       wave_obs=wave,
       fluxes=flux_mjy,
       errors=err_mjy,
       mask=good,                     # optional: True keeps a pixel
       instrument="sdss",
       aperture_diameter_arcsec=3.0,  # fiber or slit size
       resolving_power=2000.0,
   )

``resolving_power`` sets the instrumental broadening of the model. Several spectra
can be passed as a list; each gets its own calibration and aperture.

Enable the AGN emission lines, Fe II and Balmer continuum that the spectrum
resolves on the AGN configuration (see :doc:`agn`):

.. code-block:: python

   cfg.agn.fit_lines = True
   cfg.agn.fit_feii = True
   cfg.agn.fit_balmer_continuum = True
   cfg.agn.tied_lines = True

For the host, fit the stellar kinematics and use an SFH with a recent burst (see
:doc:`host`):

.. code-block:: python

   cfg.galaxy.fit_host_kinematics = True
   cfg.galaxy.host_sfh_model = "delayed_burst"

A spectrum constrains the host's absorption lines and continuum shape far more
tightly than photometry. Without the velocity dispersion and the burst, the host
model cannot follow a real galaxy spectrum, and the AGN continuum absorbs the
misfit: in tests on Chimeras (real galaxies with an added quasar of known
luminosity), the AGN luminosity came out too high and the stellar mass too low.

Spectrum calibration and apertures
----------------------------------

A fiber or slit spectrum sees only part of an extended host, and its absolute
calibration can differ from the photometry. Two settings handle this:

.. code-block:: python

   cfg.likelihood.fit_spectrum_scale = True
   cfg.likelihood.use_host_capture_model = True

``fit_spectrum_scale`` fits a gray scale factor between the spectrum and the
photometry, with a prior width of ``spectrum_scale_prior_sigma_dex``.
``use_host_capture_model`` gives the spectrum its own captured fraction of the
extended host, from its ``aperture_diameter_arcsec`` or ``psf_fwhm_arcsec``, as
described in :doc:`photometry`. At each pixel the model is

.. math::

   f^{\rm spec} = s_{\rm spec}
   \left[ f^{\rm compact} + \eta_{\rm spec} f^{\rm extended} \right],

where :math:`s_{\rm spec}` is the gray scale and :math:`\eta_{\rm spec}` the
captured host fraction. Turn ``fit_spectrum_scale`` off only when the spectrum is
known to be calibrated to the same total flux as the photometry, as for simulated
data.

Faint AGN
---------

When the AGN contributes only a few percent of the optical light, a joint fit can
attribute part of any remaining host misfit to the AGN continuum, which sets a
floor on the inferred AGN luminosity. In tests with one massive Chimera host
(log M* = 11.25), the fitted AGN bolometric luminosity stayed near
10\ :sup:`44` erg s\ :sup:`-1` even when the true AGN was up to 100 times fainter.
Treat faint-AGN luminosities from joint fits as upper limits unless the broad
lines or the continuum shape clearly require the AGN.

Spectrum-only fits
------------------

Photometry is optional, so a spectrum can be fitted on its own:

.. code-block:: python

   from jaxsedfit import FitConfig, SpectroscopyData

   cfg = FitConfig(
       observation=observation,
       spectroscopy=SpectroscopyData(
           wave_obs=wave, fluxes=flux, errors=error, resolving_power=2000
       ),
   )

The ``jaxqsofit`` package provides a spectrum-focused interface to the same
spectral model.

Spectral components
-------------------

:class:`jaxsedfit.SpectrumConfig` switches the spectral components with the same
config object as standalone ``jaxqsofit``:

.. code-block:: python

   from jaxsedfit import SpectrumConfig

   cfg.spectrum = SpectrumConfig(
       power_law_enabled=True,
       host_enabled=True,
       lines_enabled=True,
       feii_enabled=True,
       balmer_continuum_enabled=True,
       line_definitions=line_table,
       components=[component],
   )

``line_definitions`` and ``components`` take the line table and custom components
described in :doc:`agn`. Photometry, preprocessing and inference settings stay in
their own sections of :class:`jaxsedfit.FitConfig`.

Reading spectral results
------------------------

``result.spectrum`` is the supported interface for spectral results. It has named,
unit-explicit fields, so user code does not depend on internal NumPyro site names.
Every numerical field keeps the posterior-draw axis.

The fitted model and its components have shape ``(draw, pixel)``:

.. code-block:: python

   spectrum = result.spectrum
   wave = spectrum.wavelength_obs_angstrom
   model_draws = spectrum.model_flux_mjy
   continuum_draws = spectrum.continuum_flux_mjy
   line_draws = spectrum.line_flux_density_mjy
   feii_draws = spectrum.feii_flux_density_mjy
   balmer_draws = spectrum.balmer_flux_density_mjy

Each Gaussian component of a line has its own name:

.. code-block:: python

   hb1 = result.spectrum.lines["Hb_br_1"]
   hb1.amplitude_mjy, hb1.fwhm_kms, hb1.flux_w_m2

The available quantities are ``amplitude_mjy``, ``center_rest_angstrom``,
``sigma_ln_lambda``, ``fwhm_kms``, ``velocity_offset_kms`` and ``flux_w_m2``. The
amplitude and integrated flux describe the source model before per-spectrum
calibration or aperture scaling. The metadata fields are ``parent_line``,
``component_index``, ``kind`` and ``rest_wavelength_angstrom``. Single-component
lines have no ``_1`` suffix (for example ``OIII_5007``). To sum all components of a
line, use ``line_groups``:

.. code-block:: python

   group = result.spectrum.line_groups["Hb_br"]
   group.component_names              # ("Hb_br_1", "Hb_br_2", ...)
   hb_total_flux_draws = group.total_flux_w_m2

With several input spectra, ``observations`` separates the model by spectrum and
includes the data, errors, mask and residuals:

.. code-block:: python

   sdss = result.spectrum.observations[0]
   sdss.instrument
   sdss.observed_flux_mjy
   sdss.model_flux_mjy
   sdss.residual_mjy

The same interface is available on a prediction, which can limit the number of
evaluated draws:

.. code-block:: python

   prediction = result.predict(n_draws=200)
   prediction.spectrum.line_groups["Hb_br"]

It is also rebuilt after loading a saved fit with ``jaxsedfit.load_result(...)``.
``fitter.plot_spectrum()`` plots the spectrum, the model components and the
residuals. :doc:`outputs` lists every spectral output with its units.
