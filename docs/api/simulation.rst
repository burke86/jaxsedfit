Simulation
==========

Batched forward simulation of broadband photometry for mock catalogues, population
studies, prior predictive checks and training data for simulation-based (amortized)
inference. Parameters you pass are held fixed; every other model parameter is drawn from
the priors in the :class:`FitConfig`.

.. code-block:: python

   import numpy as np
   import numpyro.distributions as dist
   import jaxsedfit

   cfg = jaxsedfit.simulation_config(
       [f"sloan.sdss.{b}" for b in "ugriz"],
       redshift_range=(0.05, 2.0),        # redshift sampled; projection cached over the range
       dsps_ssp_fn="tempdata.h5",
   )
   # population prior for the AGN luminosity (the default is seeded from observed fluxes)
   cfg.prior_config.agn.log_amp = dist.Uniform(np.log(1e33), np.log(1e38))

   rng = np.random.default_rng(0)
   res = jaxsedfit.simulate(cfg, {"log_stellar_mass": rng.uniform(8, 11.5, 10_000)}, seed=1)
   res.fluxes                          # (10000, 5) total fluxes in mJy
   res.components["agn_fluxes"]        # per-component band fluxes
   res.params["redshift"]              # every sampled parameter, fixed or drawn
   df = res.to_dataframe()

Notes:

- The configured photometry is not used as data: only filter names and PSF/aperture
  metadata matter. :func:`simulation_config` fills placeholder fluxes.
- The default AGN luminosity prior is seeded from the configured fluxes; set
  ``prior_config.agn.log_amp`` (or pass ``log_agn_amp``) for a population prior.
  :func:`simulate` warns otherwise.
- Priors that depend on other parameters (for example an SFH age bounded by the age of the
  universe at each redshift) can be imposed by drawing those parameters yourself and
  passing them as fixed arrays.
- Batches are padded to ``batch_size`` so one compilation serves the whole run; results do
  not depend on the batch size.

.. currentmodule:: jaxsedfit

.. autofunction:: simulate

.. autofunction:: simulation_config

.. autofunction:: make_simulator

.. autoclass:: SimulationResult
   :members: to_dataframe
   :no-undoc-members:
   :no-special-members:
