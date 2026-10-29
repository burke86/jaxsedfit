Photometry and apertures
========================

Photometry inputs
-----------------

:class:`jaxsedfit.PhotometryData` carries the fluxes and errors, in mJy, and
optional per-point metadata:

.. code-block:: python

   from jaxsedfit import PhotometryData

   cfg.photometry = PhotometryData(
       filter_names=["g_sdss", "r_sdss", "i_sdss", "W1", "W2"],
       fluxes=[0.48, 0.73, 0.86, 1.9, 1.6],
       errors=[0.04, 0.05, 0.06, 0.08, 0.08],
       is_upper_limit=[False, False, False, False, False],
       psf_fwhm_arcsec=[1.4, 1.3, 1.3, 6.1, 6.8],
       aperture_diameter_arcsec=[None, None, None, None, None],
       photometry_method=["psf", "psf", "psf", "profile", "profile"],
   )

``is_upper_limit``
   Marks one-sided photometric constraints.

``psf_fwhm_arcsec``
   The effective PSF size of each point. Used by the aperture model below.

``aperture_diameter_arcsec``
   The aperture diameter, when a catalog flux comes from a known circular
   aperture. Takes precedence over ``psf_fwhm_arcsec``.

``photometry_method``
   Provenance: ``psf`` for point-source/PSF photometry, ``profile`` for
   profile-fit photometry such as AllWISE ``W?mag``, ``aperture`` for an explicit
   fixed aperture, ``auto`` for Kron/AUTO-like photometry,
   ``model``/``cmodel``/``petrosian`` for extended-source model measurements, and
   ``catalog`` when the aperture semantics are unknown. With the host-capture
   model on, ``profile``, ``auto``, ``model``, ``cmodel`` and ``petrosian``
   measurements are treated as total light (see below); otherwise it is recorded
   for plots and checks.

Filters
-------

``filter_names`` refers to the packaged transmission curves; :doc:`filters`
lists them with their aliases (for example ``u_sdss``, ``J_2mass`` or ``W1``).
Packaged curves are converted from photon response to the internal energy-response
convention before synthetic photometry. Custom curves can be supplied directly:

.. code-block:: python

   from jaxsedfit import FilterCurve, FilterSet

   cfg.filters = FilterSet(
       curves=[
           FilterCurve(
               name="my_filter",
               wave=[4000.0, 5000.0, 6000.0],
               transmission=[0.0, 1.0, 0.0],
           )
       ]
   )

Photometry from different apertures
-----------------------------------

SED photometry often mixes measurements that see different amounts of an extended
host: SDSS PSF fluxes, 2MASS point-source photometry, fixed apertures, catalog
model or ``AUTO`` fluxes, and sometimes a fiber or slit spectrum. ``jaxsedfit``
handles this with an empirical host-capture model. It is off by default; enable it
when the measurements do not all see the same aperture:

.. code-block:: python

   cfg.likelihood.use_host_capture_model = True

The model builds the intrinsic source first, then gives each measurement its own
captured fraction :math:`\eta` of the **extended** components: stellar light, host
dust, nebular emission and narrow-line emission. The **compact** components, the AGN
continuum and broad-line emission, are always fully captured. The model flux in
band :math:`b` is

.. math::

   F_b^{\rm model} = F_b^{\rm compact} + \eta_b F_b^{\rm extended}.

The captured fraction depends on an effective radius :math:`r_b` of the
measurement and on one fitted host size :math:`r_{\rm host}`, the radius at which
half the extended light is captured:

.. math::

   \eta_b = \frac{r_b^2}{r_b^2 + r_{\rm host}^2}.

This is a logistic function of :math:`\ln r_b` with a fixed slope of 2. The prior on
:math:`r_{\rm host}` is :math:`\ln (r_{\rm host}/{\rm arcsec}) \sim
\mathcal{N}(\ln 3, 1)`; it is the output ``log_host_capture_scale_arcsec``. Each
measurement's radius comes from its metadata and ``photometry_method``:

* with ``aperture_diameter_arcsec``, :math:`r_b` is half the diameter;
* otherwise, with ``psf_fwhm_arcsec``, :math:`r_b` is the Gaussian PSF's effective
  radius, :math:`\sqrt{2}\,{\rm FWHM}/2.355 \approx 0.6\,{\rm FWHM}`;
* bands whose ``photometry_method`` is a total-light measurement (``profile``,
  ``auto``, ``model``, ``cmodel`` or ``petrosian``) capture all the extended light,
  :math:`\eta_b = 1`, whatever their sizes;
* any other band without a size gets its own free capture fraction, with a uniform
  prior between 0 and 1 (output ``missing_psf_host_capture_fraction``).

For example, for SDSS PSF photometry, 2MASS point-source photometry and AllWISE
profile-fit photometry:

.. code-block:: python

   cfg.photometry = PhotometryData(
       filter_names=[
           "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss",
           "J_2mass", "H_2mass", "Ks_2mass", "W1", "W2",
       ],
       fluxes=[...],
       errors=[...],
       photometry_method=[
           "psf", "psf", "psf", "psf", "psf",
           "psf", "psf", "psf", "profile", "profile",
       ],
       psf_fwhm_arcsec=[
           1.4, 1.4, 1.4, 1.4, 1.4,
           2.5, 2.5, 2.5, 6.08, 6.84,
       ],
   )

Here the SDSS and 2MASS points capture the host according to their PSF sizes, and
the WISE points, as profile-fit photometry, count as total light. Small-PSF optical
points can therefore see less host light than the infrared points while the model
keeps one intrinsic source.

A spectrum gets its own capture fraction in the same way, from the
``aperture_diameter_arcsec`` or ``psf_fwhm_arcsec`` of its
:class:`jaxsedfit.SpectroscopyData` (for example a 3 arcsec SDSS fiber); a spectrum
with neither captures all the light. :doc:`spectroscopy` describes the other
spectrum calibration terms, and :doc:`likelihood` lists all calibration and
error-model terms.
