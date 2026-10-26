Fitting many objects
====================

:func:`jaxsedfit.fit_batch` fits many objects that share a model configuration
with a single compiled program. Fitting objects one at a time with
:class:`jaxsedfit.JAXSEDFit` compiles and optimizes each separately. ``fit_batch``
compiles once and fits a whole chunk of objects together with ``jax.vmap``. Use it
when you have a catalog: for example, the 1,600-object Chimera validation (MAP plus
Laplace for every object) takes about 10 minutes on an 8-core laptop.

Supported methods are ``"map"``, ``"laplace"`` and ``"svi"`` (see
:doc:`inference`). For NUTS or nested sampling, loop over objects with
:class:`jaxsedfit.JAXSEDFit`.

Example
-------

Build one :class:`jaxsedfit.FitConfig` per object, usually from catalog rows with a
small helper, then pass the list to ``fit_batch``:

.. code-block:: python

   import jaxsedfit
   from jaxsedfit import FitConfig, GalaxyConfig, InferenceConfig, Observation, PhotometryData

   FILTERS = ["u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss"]

   def make_config(row):
       return FitConfig(
           observation=Observation(object_id=str(row["id"]), redshift=row["z"]),
           photometry=PhotometryData(
               filter_names=FILTERS,
               fluxes=[row[f] for f in FILTERS],
               errors=[row[f + "_err"] for f in FILTERS],
           ),
           galaxy=GalaxyConfig(dsps_ssp_fn="tempdata.h5"),
           inference=InferenceConfig(method="laplace", map_steps=1500, num_samples=400),
       )

   configs = [make_config(row) for row in catalog]
   batch = jaxsedfit.fit_batch(configs, chunk_size=32)

   table = batch.summary_table(sites=["log_stellar_mass"])  # 16/50/84th percentiles
   batch.samples[0]["log_stellar_mass"]      # draws for the first object
   batch.map_values[0]["log_stellar_mass"]   # its MAP
   result = batch[0]                         # a FitResult, as from JAXSEDFit.fit()
   result.fitter.plot_sed()

``method`` defaults to the first config's ``inference.method``. The MAP, Laplace and
SVI settings (``map_steps``, ``learning_rate``, ``staged_map``, ``num_samples``,
``svi_*``) come from the configs.

What can be batched
-------------------

Objects are grouped automatically. Two objects share a compiled program when their
configurations agree on everything except per-object data:

* the object id, redshift and sky position (``Observation``);
* the fluxes, errors and upper-limit flags (``PhotometryData``);
* the spectral pixels (wavelengths, fluxes, errors, mask) and resolving power of
  each spectrum (``SpectroscopyData``).

Everything else (filters, model options, priors and inference settings) defines the
group. A batch can therefore mix filter sets or priors, but each distinct
combination compiles its own program, so batching only pays off when many objects
share one. The redshift can be fixed per object or free (``redshift_mode``); objects
with different ``redshift_err`` fall into different groups.

Spectra
-------

Joint (photometry plus spectra) and spectrum-only fits are batched too. Two things
make the spectral model differ between objects, and ``fit_batch`` removes both:

* **Pixel counts.** Every spectrum in a group is padded to the group's longest with
  masked pixels that continue its log-wavelength grid. Padding leaves the model on
  the real pixels unchanged, apart from the Balmer-continuum convolution grid, which
  follows the array length (a change of about 1e-8 in the log density).
* **The tied-line table.** It normally contains the lines a spectrum covers, so it
  changes with redshift. Objects are instead split into redshift windows no wider
  than ``redshift_window`` in ln(1 + z) (default 0.25). Each window shares one table
  spanning its members' coverage (``AGNConfig.line_activation_rest``), and lines
  outside an object's own coverage get zero amplitude, so only their priors
  constrain them. Wider windows mean fewer compiled programs but more masked lines
  per object; ``redshift_window=None`` gives each configuration group one table.

Different resolving powers are per-object data and share one program. Fitted
redshifts cannot be batched for spectroscopic objects.

A fixed table can change a fit slightly compared with :class:`jaxsedfit.JAXSEDFit`:
a tied group that straddles the coverage edge keeps all its members, with the
uncovered ones masked. ``batch.configs`` carries each object's table, so
``JAXSEDFit(batch.configs[i])``, and ``batch[i]``, fit and predict the same model.

Other models
------------

:class:`jaxsedfit.batched.BatchedInputs` stacks per-object keyword inputs of any
NumPyro model with a common structure, and
:func:`jaxsedfit.batched.make_batched_fitter` fits them with the same vmapped
MAP, Laplace or SVI program. jaxqsofit's ``fit_batch`` uses these with
:func:`jaxsedfit.spectral_model.qso_fsps_joint_model`, whose ``pixel_mask`` argument
and ``tied_line_meta["line_mask"]`` entry support padded spectra and shared line
tables.

Results
-------

:class:`jaxsedfit.BatchFitResult` keeps results in input order:

* ``samples[i]``: draws of every parameter for object ``i``. For ``"map"`` there is a
  single draw, the MAP.
* ``map_values[i]``: the MAP of every parameter (all methods start from it).
* ``diagnostics[i]``: ``loss`` (the negative log posterior at the MAP), plus
  ``hessian_min_eigenvalue`` for Laplace or ``elbo_loss`` for SVI.
* ``summary_table(sites, percentiles)``: an :class:`astropy.table.Table` of
  percentiles, one row per object.
* ``batch[i]``: a :class:`jaxsedfit.FitResult` for one object, for predictions, SED
  plots and ``save``. It rebuilds that object's model context, which takes about a
  second, because the batch does not keep contexts in memory.

Each fitted object matches the single-object fit of its returned config with the
same settings: the batched model evaluates the same log density as
:class:`jaxsedfit.JAXSEDFit`, to numerical precision. Random starting points come from the seed and the object's position in
``configs``, so results do not depend on ``chunk_size``.

Performance and memory
----------------------

``chunk_size`` objects are fitted together (default 32). Each object's model context
takes roughly 40 MB with the default configuration. Contexts for the next chunk are
built on a background thread while the current chunk runs. Larger chunks amortize
per-step overhead, but use more memory and give little speedup once the CPU cores
are busy. The first chunk of each group also pays the compilation cost, which is
usually a few tens of seconds.
