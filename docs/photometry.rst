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
   ``catalog`` when the aperture semantics are unknown. It is recorded for plots
   and checks; it does not change the model by itself.

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
host: SDSS PSF fluxes, 2MASS point-source photometry, AllWISE profile-fit
photometry, fixed apertures, catalog ``AUTO``/total-like fluxes, and sometimes a
spectrum from a fiber or slit. ``jaxsedfit`` handles this with an empirical
extended-light capture model. It is off by default; enable it when the
measurements do not all see the same aperture:

.. code-block:: python

   cfg.likelihood.use_host_capture_model = True

The model is driven by the angular sizes, not by ``photometry_method``. For
example, for SDSS PSF photometry, 2MASS point-source photometry and AllWISE
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

The model builds the intrinsic source first, then gives each measurement its own
captured fraction of the **extended** components: stellar host light, host dust,
nebular emission and narrow-line emission. The **compact** components, the AGN
continuum and broad-line emission, are always fully captured. Small-PSF optical
points can therefore see less host light than large-beam infrared points while
sharing one physical model.

For band :math:`b`, the effective angular scale is

.. math::

   \theta_b =
   \begin{cases}
   d_{{\rm ap},b}, & d_{{\rm ap},b}\ {\rm supplied}, \\
   {\rm FWHM}_{{\rm PSF},b}, & {\rm otherwise\ if\ supplied}, \\
   \infty, & {\rm otherwise},
   \end{cases}

and the captured fraction of the extended light is a logistic function of
:math:`\ln\theta_b`,

.. math::

   \eta_b =
   \left[
   1 + \exp\left(
   -\alpha_{\rm cap}
   \left[\ln \theta_b - \ln \theta_{\rm cap}\right]
   \right)
   \right]^{-1},

where the turnover scale :math:`\theta_{\rm cap}` and the slope
:math:`\alpha_{\rm cap}` are fitted, with priors ``log_host_capture_scale_arcsec``
(centered near 3 arcsec) and ``host_capture_slope`` (centered near 2). A band
with neither an aperture nor a PSF size is treated as total photometry,
:math:`\eta_b = 1`. The model flux in band :math:`b` is

.. math::

   F_b^{\rm model} = F_b^{\rm compact} + \eta_b F_b^{\rm extended},

which enters the photometric likelihood (see :doc:`likelihood`). With the capture
model off, every band uses :math:`\eta_b = 1`.

A spectrum gets its own capture fraction from the ``aperture_diameter_arcsec`` or
``psf_fwhm_arcsec`` of its :class:`jaxsedfit.SpectroscopyData`, for example 3
arcsec for an SDSS fiber; see :doc:`spectroscopy`. The photometry and the spectrum
are then different views of one intrinsic source:

* SDSS PSF photometry measures the compact AGN light plus a PSF-captured fraction
  of the extended host;
* the SDSS spectrum measures the compact AGN light plus a fiber-captured fraction
  of the host;
* 2MASS and AllWISE photometry use their PSF/profile scales rather than being
  treated as total host measurements;
* large-aperture or total photometry approaches the full host when given a large
  aperture, or no size at all.
