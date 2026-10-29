Inference methods
=================

``jaxsedfit`` offers six ways to fit one object. Choose one with
``cfg.inference.method`` and call :meth:`jaxsedfit.JAXSEDFit.fit`, or call the
matching ``fit_*`` method directly. Every method returns a
:class:`jaxsedfit.FitResult`. ``result.samples`` maps each model parameter to an
array whose first axis indexes draws, so summaries, predictions, plots and
``save`` work the same way whichever method produced them.

.. list-table::
   :header-rows: 1
   :widths: 14 22 24 40

   * - ``method``
     - Returns
     - Typical cost (photometry)
     - Use it for
   * - ``"map"``
     - the MAP (one point)
     - seconds
     - quick looks, smoke tests, starting points
   * - ``"laplace"``
     - Gaussian draws at the MAP
     - MAP + one Hessian
     - fast uncertainties for well-constrained, near-Gaussian posteriors; large samples
   * - ``"svi"``
     - Gaussian variational draws
     - MAP + a few thousand ELBO steps
     - fast uncertainties when posteriors are correlated or mildly skewed
   * - ``"nuts"``
     - MCMC draws
     - minutes
     - accurate posteriors
   * - ``"map+nuts"`` (default)
     - MCMC draws, started at the MAP
     - minutes
     - accurate posteriors (recommended default)
   * - ``"ns"``
     - nested-sampling draws and evidence
     - longest
     - multimodal posteriors, model comparison

``"optax"`` and ``"optax+nuts"`` are kept as aliases of ``"map"`` and
``"map+nuts"``; ``result.method`` echoes the name you used. To fit many objects at
once with ``"map"``, ``"laplace"`` or ``"svi"``, see :doc:`batch`.

How to choose
-------------

* Start with ``"map"`` to check that the model can describe the data (look at the
  SED plot and the residuals) before spending time on uncertainties.
* Use ``"map+nuts"`` for results you will publish, and whenever the posterior may
  be multimodal, strongly non-Gaussian, or pushed against a prior bound.
* Use ``"laplace"`` or ``"svi"`` when you need uncertainties for many objects, or a
  fast first estimate. Check them against ``"map+nuts"`` on a few representative
  objects before relying on them for a new kind of data.
* Use ``"ns"`` when the posterior is multimodal or you need the evidence.

MAP (``"map"``)
---------------

The maximum a posteriori point: the most probable parameters given the data and the
priors. It is found with Adam (``map_steps``, ``learning_rate``). When
``staged_map=True`` (the default), a first stage fits the smooth continuum without
emission lines (``staged_steps``, default ``map_steps // 3``) and the second stage
fits the full model from there.

.. code-block:: python

   cfg.inference.method = "map"
   cfg.inference.map_steps = 1500
   result = JAXSEDFit(cfg).fit()
   result.samples["log_stellar_mass"]  # shape (1,): the MAP

A MAP is a single point. It carries no uncertainty, and for skewed posteriors it can
differ from the posterior median by a few tenths of a dex.

Laplace approximation (``"laplace"``)
-------------------------------------

The MAP, followed by a Gaussian centred on it whose covariance is the inverse
Hessian (the curvature) of the negative log posterior. The Gaussian lives in
NumPyro's unconstrained parameter space, so draws of bounded parameters stay inside
their priors. ``num_samples`` sets the number of draws.

.. code-block:: python

   cfg.inference.method = "laplace"
   cfg.inference.num_samples = 400
   fitter = JAXSEDFit(cfg)
   result = fitter.fit()
   fitter.approx_result["hessian_min_eigenvalue"]

The Laplace approximation costs one Hessian after the MAP, which is negligible next
to the optimization. It is accurate when the posterior is close to Gaussian around
its peak, which is typical for well-measured quantities such as the stellar mass of
a galaxy with good photometry. It is unreliable when:

* the posterior has several modes (the Gaussian only describes the one the MAP found);
* a parameter sits against a prior bound (for example a line flux consistent with zero);
* parameters are tied together by a strongly curved degeneracy.

If ``approx_result["hessian_min_eigenvalue"]`` is zero or negative, the Hessian was
not positive definite. The flat or non-convex directions were then given a very
wide Gaussian, so treat that object's uncertainties as unreliable.

Variational inference (``"svi"``)
---------------------------------

The MAP, followed by stochastic variational inference: a Gaussian in unconstrained
space is fitted to the posterior by maximizing the evidence lower bound (ELBO),
starting at the MAP. Unlike the Laplace approximation, which uses the curvature at
one point, SVI adapts the Gaussian to the posterior over the region it covers, so it
copes better with correlated and mildly skewed posteriors. It is still a single
Gaussian, with the same limitations for multimodal posteriors.

.. code-block:: python

   cfg.inference.method = "svi"
   cfg.inference.svi_steps = 2000         # optimization steps
   cfg.inference.svi_learning_rate = None  # None uses learning_rate
   cfg.inference.svi_num_particles = 4     # Monte Carlo draws per step
   cfg.inference.svi_full_rank = True      # full covariance; False for diagonal
   cfg.inference.num_samples = 400
   fitter = JAXSEDFit(cfg)
   result = fitter.fit()
   fitter.approx_result["losses"]  # negative ELBO per step; check that it has flattened

The full-rank Gaussian corresponds to NumPyro's ``AutoMultivariateNormal`` guide and
the diagonal one to ``AutoNormal``. If the loss has not flattened by the last step,
increase ``svi_steps`` or ``svi_learning_rate``.

NUTS (``"nuts"`` and ``"map+nuts"``)
------------------------------------

The No-U-Turn Sampler, a Hamiltonian Monte Carlo method, draws from the full
posterior with no Gaussian assumption. ``"map+nuts"`` runs the MAP first and starts
the chains there, which shortens warmup. With ``"nuts"``, the chains also start from
the MAP while ``use_map_init=True`` (the default). See the NUTS settings in
:class:`jaxsedfit.InferenceConfig` (``num_warmup``, ``num_samples``,
``num_chains``, ``target_accept_prob``, ``dense_mass``, ``max_tree_depth``).

.. code-block:: python

   cfg.inference.method = "map+nuts"
   cfg.inference.num_warmup = 200
   cfg.inference.num_samples = 200
   cfg.inference.target_accept_prob = 0.85
   cfg.inference.dense_mass = "blocks"
   cfg.inference.max_tree_depth = 8

Fewer warmup and posterior draws are fine for notebooks and tests; increase them
for published results.

``dense_mass="blocks"`` (the default) learns the covariance only within groups of
related parameters (host, AGN continuum, torus, spectral lines). By default the
model is also sampled in better-conditioned coordinates that leave the posterior
unchanged: an exact normalization coordinate
(``reparameterize_normalizations=True``) and, in joint fits, standard-normal
coordinates for the Fe II and Balmer-continuum priors
(``reparameterize_spectral_features=True``). Set either to ``False`` for a
controlled comparison.

If NUTS keeps reaching the maximum tree depth:

1. check the MAP with a ``"map"`` fit, and inspect the printed NumPyro diagnostics;
2. fix ``cfg.likelihood.systematics_width`` at a reasonable value, which removes a
   common model-error degeneracy;
3. only then increase ``max_tree_depth``, which can multiply the run time without
   improving the geometry.

Nested sampling (``"ns"``)
--------------------------

Nested sampling with ``jaxns`` explores the whole prior volume, so it handles
multimodal posteriors and returns the Bayesian evidence. It is the slowest option.
Its settings are the ``ns_*`` fields of :class:`jaxsedfit.InferenceConfig`.

.. code-block:: python

   cfg.inference.method = "ns"
   cfg.inference.ns_num_live_points = 200
   cfg.inference.ns_dlogz = 0.1
