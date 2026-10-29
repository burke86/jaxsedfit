Host galaxy
===========

The host-galaxy model is controlled by :class:`jaxsedfit.GalaxyConfig`. It
samples the stellar mass, metallicity, dust and star-formation history (SFH).

Stellar populations
-------------------

.. code-block:: python

   from jaxsedfit import GalaxyConfig

   cfg.galaxy = GalaxyConfig(
       fit_host=True,
       host_sfh_model="delayed",
       dsps_ssp_fn="tempdata.h5",
       ssp_imf="kroupa_2001",
       ssp_metallicity_coordinate="absolute_log10_z",
       ssp_solar_metallicity=0.019,
       rest_wave_min=100.0,
       rest_wave_max=3.0e6,
       n_wave=1024,
   )

``ssp_imf`` and ``ssp_metallicity_coordinate`` declare the provenance of the
loaded SSP library; they do not regenerate it. The IMF declaration also selects
the matching DSPS surviving-mass calibration, so it must describe the SSP file
accurately. The public DSPS FSPS-v3.2 file from :doc:`installation` was generated
with the FSPS default Kroupa (2001) IMF. Supported IMFs are ``chabrier_2003``,
``salpeter_1955``, ``kroupa_2001`` and ``van_dokkum_2008``; supported metallicity
coordinates are ``absolute_log10_z`` and ``log10_z_over_zsun``. Custom IMFs need an
SSP format that supplies its own surviving-mass fractions and are not
approximated.

Other useful settings:

``fit_host``
   Disable only for AGN-only experiments. At least one of ``fit_host`` or
   ``agn.fit_agn`` must be true.

``n_wave``
   The size of the internal rest-frame wavelength grid. Larger values are more
   faithful for detailed spectra but slower.

``ssp_template_fwhm_angstrom``
   The rest-frame FWHM resolution of the SSP templates: 2.5 Å by default, for
   MILES-based FSPS libraries. Set it to match your SSP file, or ``None`` to ignore
   the template resolution.

``use_energy_balance`` and ``dust_alpha``
   Control the host-dust energy-balance component. Disable dust only for debugging
   or deliberately dust-free comparisons.

Star-formation history
----------------------

``host_sfh_model`` selects the SFH:

``"delayed"`` (default)
   A delayed-:math:`\tau` SFH. Low-dimensional and usually the most stable choice
   for broadband SEDs.

``"delayed_burst"``
   Adds a recent burst to the delayed-:math:`\tau` SFH. Recommended when a spectrum
   resolves the host's continuum shape (see :doc:`spectroscopy`).

``"diffstar"``
   The Diffstar parameterization.

By default the delayed SFHs use GRAHSP-like priors on the SFH age and ``tau``:
log-uniform from about 160 Myr to 10 Gyr (or the age of the Universe) and from 0.1 to
10 Gyr. :doc:`likelihood` lists all default priors, including the burst's. Flat
priors on age and ``tau`` are not flat in derived quantities: they imply a fairly
informative prior on the specific star-formation rate. When the photometry does not
constrain the SFH well, you can put the prior on the sSFR (current SFR per formed
stellar mass, in yr\ :sup:`-1`) instead:

.. code-block:: python

   cfg.galaxy.sfh_prior = "ssfr_uniform"          # default: "age_tau"
   cfg.galaxy.ssfr_prior_range = (-14.0, -8.0)    # log10 sSFR / yr^-1

This samples ``log_sfh_ssfr`` uniformly in that range (or from a prior set on
``cfg.prior_config.host.log_sfh_ssfr``), draws the SFH age log-uniformly up to the
largest age that can still reach that sSFR, and solves for ``tau``. With
``"delayed_burst"``, the burst parameters keep their usual priors and ``tau`` is
chosen so that the total (main + burst) sSFR equals the sampled value; where the
burst alone already exceeds it, the total sSFR is higher than sampled. Priors on
``log_sfh_age_gyr`` or ``log_sfh_tau_gyr`` cannot be combined with this mode, and
it does not apply to Diffstar.

Dust attenuation
----------------

The host attenuation curve defaults to the GRAHSP broken power law. For rest-frame
UV photometry you can switch to the Noll et al. (2009) modified Calzetti curve,
which adds the 2175 Å bump and a variable slope:

.. code-block:: python

   cfg.galaxy.attenuation_curve = "noll2009"      # default: "grahsp"

The curve is ``[k_Calzetti + D(E_b)] (lambda / 5500 A)^delta``, renormalized so that
``ebv_gal`` stays an E(B-V). The bump amplitude is tied to the slope by Kriek &
Conroy (2013), ``E_b = 0.85 - 1.9 delta``, so the only new parameter is
``dust_delta`` (default prior: normal with mean -0.2 and width 0.3, truncated to
[-0.8, 0.4]; override it with ``cfg.prior_config.host.dust_delta``). AGN light sees
the host screen through this curve and its own screen, ``ebv_agn``, through the
GRAHSP curve.

Stellar kinematics
------------------

``fit_host_kinematics=True`` fits the host's velocity offset and velocity
dispersion in spectroscopic fits. When a spectrum sets ``resolving_power``, the host
model on that spectrum is convolved with
``sqrt(sigma*^2 + sigma_inst^2 - sigma_template^2)``, so ``gal_sigma_kms`` is the
intrinsic dispersion; without a resolving power it is an effective broadening. It
has no effect on photometry-only fits.

Nebular emission
----------------

Host nebular emission is configured with :class:`jaxsedfit.NebularConfig`:

.. code-block:: python

   from jaxsedfit import NebularConfig

   cfg.nebular = NebularConfig(
       enabled=True,
       emission=True,
       logU=-2.0,
       zgas=0.019,
       ne=100.0,
       f_esc=0.0,
       f_dust=0.0,
       lines_width=300.0,
   )

By default ``logU``, ``zgas`` and ``ne`` are fixed to the GRAHSP-like values
``-2.0``, ``0.019`` and ``100 cm^-3``. Set ``zgas=None`` to tie the gas metallicity
to the host metallicity. Only add priors for ``logU``, ``zgas`` or ``ne`` when the
data can constrain them:

.. code-block:: python

   from numpyro import distributions as dist

   cfg.prior_config.nebular.logU = dist.Normal(-2.0, 0.3)
   cfg.prior_config.nebular.zgas = dist.TruncatedNormal(
       loc=0.02, scale=0.01, low=1.0e-4, high=0.1
   )

The packaged nebular templates are interpolated over metallicity, ionization
parameter and density, so off-grid values stay smooth for gradient-based
inference.

``cfg.likelihood.use_local_line_photometry`` integrates emission lines through each
filter on a local high-resolution grid. Keep it on for science fits; turning it off
removes these line corrections and can make a quick test easier to sample.
