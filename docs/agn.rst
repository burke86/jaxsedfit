AGN model
=========

The AGN continuum, torus, emission lines, Fe II and Balmer continuum are
controlled by :class:`jaxsedfit.AGNConfig` and the AGN prior fields.

.. code-block:: python

   from jaxsedfit import AGNConfig

   cfg.agn = AGNConfig(
       fit_agn=True,
       use_powerlaw_disk=True,
       agn_type=1,
       fit_feii_broadening=False,
       fit_balmer_continuum=False,
   )

In a broadband fit, the AGN is a disk power law, a torus, and SED-scale broad and
narrow emission-line corrections. When a spectrum is present, the same
configuration also controls the detailed Fe II, Balmer-continuum and emission-line
model (see :doc:`spectroscopy`); these shared components are evaluated in the
spectrum and projected into the photometry.

AGN type
--------

``agn_type`` selects the empirical emission-line branch used for the broadband
line corrections:

``agn_type=1``
   Broad-line AGN. Includes the broad-line template, the Seyfert-2-like
   narrow-line template, and the SED-scale Fe II and Balmer-continuum components
   when those are enabled.

``agn_type=2``
   Narrow-line AGN (Seyfert 2). No broad lines; the Seyfert-2-like narrow-line
   template. No SED-scale Fe II or Balmer continuum.

``agn_type=3``
   LINER. No broad lines; the LINER narrow-line template. This is a line-ratio
   template choice for low-ionization narrow-line AGN, not a "type 3 quasar".

In joint fits, the lines covered by the spectrum use the detailed spectral line
model; the SED-scale corrections still apply to filters outside the spectral
coverage.

Accretion disk and torus
------------------------

The continuum follows GRAHSP: a bent power-law accretion disk and an empirical
torus whose luminosity is tied to the disk's.

**Disk.** The disk is normalized by its luminosity at rest 5100 Å,
``agn_amp`` :math:`= \lambda L_\lambda(5100\,\text{Å})` in W (sampled as
``log_agn_amp``, a natural logarithm). It is a smoothly bent power law,
:math:`L_\lambda \propto \lambda^{\texttt{pl\_slope}}` redward of the bend at
``pl_bend_loc`` (about 1000 Å) and :math:`\propto \lambda^{\texttt{uv\_slope}}`
blueward of it, with the bend's sharpness set by ``pl_bend_width`` and an exponential
cutoff, :math:`1 - e^{-\lambda_c / \lambda}`, at :math:`\lambda_c = 10^5` Å. Both
dust screens attenuate the disk (see `AGN luminosity and reddening`_).

**Torus.** The torus is two log-normal dust bumps, cool and hot, with a silicate
feature:

.. math::

   L_\lambda^{\rm torus}(\lambda) = \frac{2.5\, f_{\rm cov}\, \lambda L_\lambda(5100\,\text{Å})}{12\,\mu{\rm m}}
   \, \frac{S(\lambda)}{S(12\,\mu{\rm m})}
   \left[ 1 + \tanh({\rm si})\, P_{\rm Si}(\lambda) \right],

.. math::

   S(\lambda) = \exp\!\left[ -\left( \frac{\log_{10}(\lambda / \lambda_{\rm cool})}{w_{\rm cool}} \right)^2 \right]
   + h\, \frac{\lambda_{\rm cool}}{\lambda_{\rm hot}}
   \exp\!\left[ -\left( \frac{\log_{10}(\lambda / \lambda_{\rm hot})}{w_{\rm hot}} \right)^2 \right],

.. math::

   P_{\rm Si}(\lambda) = \exp\!\left[ -\tfrac{1}{2} \left( \frac{\lambda - 9.84\,\mu{\rm m}}{1.03\,\mu{\rm m}} \right)^2 \right]
   - 0.29 \exp\!\left[ -\tfrac{1}{2} \left( \frac{\lambda - 14.22\,\mu{\rm m}}{1.16\,\mu{\rm m}} \right)^2 \right].

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Parameter
     - Meaning (default prior)
   * - ``fcov`` (:math:`f_{\rm cov}`)
     - Torus normalization: :math:`\lambda L_\lambda(12\,\mu{\rm m}) = 2.5\, f_{\rm cov}\,
       \lambda L_\lambda(5100\,\text{Å})` before the silicate term, so the default
       range 0.05–0.95 allows 12 µm-to-5100 Å ratios from 0.13 to 2.4. A proxy, not a
       geometric covering factor or a ratio of integrated luminosities.
   * - ``cool_lam``, ``cool_width``
     - Peak wavelength (10–30 µm) and width in dex (0.2–0.65) of the cool bump
   * - ``hot_lam``, ``hot_width``
     - Peak wavelength (1–5.5 µm) and width in dex (0.2–0.65) of the hot bump
   * - ``hot_fcov`` (:math:`h`)
     - Hot-to-cool amplitude ratio, before the common normalization (log-uniform
       from 0.04 to 10)
   * - ``si``
     - Silicate strength: positive for emission near 9.8 µm, negative for
       absorption; ``tanh`` keeps the torus positive (Normal(0, 1))

The hot bump dominates the torus from about 1 to 5 µm rest frame and the cool bump
in the mid-infrared. The torus is not energy-balanced: its luminosity scales with
the *intrinsic* disk luminosity through ``fcov``, independently of the light the
nuclear screen ``ebv_agn`` absorbs. Its own obscuration is part of the template, so
it sees only the host screen ``ebv_gal``.

Because the disk and torus share ``agn_amp``, the torus is the main infrared
constraint on the AGN luminosity. With photometry that stops short of rest-frame
1 µm, it is barely sampled, and an AGN disk can be hard to tell apart from young
stars.

AGN luminosity and reddening
----------------------------

The disk, Fe II, Balmer continuum and AGN emission lines pass through two dust
screens: the host galaxy's (``ebv_gal``, which also attenuates the stars) and an
additional nuclear one (``ebv_agn``). The torus sees only the host screen. The AGN
luminosities therefore come in two versions:

``dereddened``
   Corrected for both screens: the intrinsic luminosity of the model.
   ``log_agn_bol_luminosity_fit`` is this version.

``attenuated``
   Not corrected: what an observer would measure from the AGN continuum, as in
   quasar catalogs that do not correct for reddening, such as the SDSS
   luminosities of Shen et al. (2011).

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Output
     - Definition
   * - ``log_agn_l5100_dereddened_fit``, ``log_agn_l5100_attenuated_fit``
     - :math:`\lambda L_\lambda` of the AGN disk at rest 5100 Å, in log10(W)
   * - ``log_agn_bol_luminosity_dereddened_fit``, ``log_agn_bol_luminosity_attenuated_fit``
     - :math:`9.26\,\lambda L_{5100}`, in log10(W)
   * - ``agn_transmission_5100_fit``
     - Fraction of the AGN light at 5100 Å that passes both screens

The attenuated value is the dereddened one times ``agn_transmission_5100_fit``.
Luminosities are in W; add 7 to the logarithms for erg s\ :sup:`-1`.

Compare with the version that matches the reference. The Chimera tutorial
(:doc:`tutorials`) is an example: its quasar never passed through the host's dust and its
reference luminosity is not dereddened, so the attenuated luminosity is the right
comparison.

Emission-line table
-------------------

The detailed line model is defined by a table of
:class:`jaxsedfit.LineDefinition` rows, ``cfg.agn.line_table``. Broad lines have
names such as ``Ha_br``, ``Hb_br``, ``MgII_br``, ``CIV_br`` and ``Lya_br``; the
number of Gaussian components of each is set by ``LineDefinition.components``.
Copy the default table, change the rows you want, and assign it before
constructing :class:`jaxsedfit.JAXSEDFit`:

.. code-block:: python

   from dataclasses import replace

   from jaxsedfit import LineDefinition
   from jaxsedfit.spectral_defaults import DEFAULT_LINE_PRIOR_ROWS

   line_table = [LineDefinition.from_mapping(row) for row in DEFAULT_LINE_PRIOR_ROWS]
   line_table = [
       replace(line, components=3)
       if line.name in {"Ha_br", "Hb_br", "MgII_br"}
       else replace(line, components=2)
       if line.name in {"CIV_br", "Lya_br"}
       else line
       for line in line_table
   ]

   cfg.agn.line_table = line_table

The default table includes the strongest narrow lines, such as a single
``OII3728``. ``cfg.agn.include_elg_narrow_lines = True`` adds the narrow lines of
emission-line galaxies: the resolved [O II] 3726, 3729 doublet, [Ne III], narrow
Balmer lines, [O III] 4363 and others. All of these lines belong to the AGN model,
so in a joint fit they also absorb line emission from the host's star formation.

Custom components
-----------------

A custom continuum or line component is defined once with
:class:`jaxsedfit.SpectralComponentSpec` and used by both joint fits and
standalone ``jaxqsofit`` fits:

.. code-block:: python

   import numpyro.distributions as dist
   from jaxsedfit import SpectralComponentSpec

   component = SpectralComponentSpec(
       name="extra_continuum",
       kind="continuum",  # or "broad_line" / "narrow_line"
       parameter_priors={"amplitude": dist.HalfNormal(1.0)},
       evaluate=my_component,
   )
   cfg.agn.components = [component]

The evaluator must be JAX-compatible and return the observed flux density in mJy
on the wavelength grid it receives. The helpers ``make_custom_component`` and
``make_custom_line_component`` build the same kind of component for
``cfg.agn.custom_components``:

.. code-block:: python

   import jax.numpy as jnp
   import numpyro.distributions as dist
   from jaxsedfit import make_custom_component


   def extra_continuum(wave, params, metadata):
       return params["amplitude"] * (wave / metadata["pivot"]) ** params["slope"]


   component = make_custom_component(
       "extra_continuum",
       {
           "amplitude": dist.LogNormal(jnp.log(0.1), 0.5),
           "slope": dist.Normal(0.0, 1.0),
       },
       extra_continuum,
       metadata={"pivot": 5100.0},
       frame="rest",  # or "observed"
   )
   cfg.agn.custom_components = (component,)

The parameters are sampled once, and the same realization is rendered on the
spectral pixels, on each filter's wavelength grid, and on the posterior SED grid.
Use ``make_custom_line_component(..., line_kind="broad" | "narrow")`` for an
additive line profile; narrow custom lines get the same host-aperture capture as
the built-in narrow lines.

**Broad absorption lines.** A custom component with
``metadata["component_type"] == "bal_absorption"`` returns an optical depth and is
applied as partial-covering multiplicative absorption, not additive flux. It
attenuates the AGN disk, additive AGN custom continua, Fe II, the Balmer continuum
and the broad lines consistently in the spectrum, the photometry and the rendered
SED; host, torus and narrow-line emission are not attenuated. In spectrum plots,
BAL curves are negative and show the removed flux.

.. warning::

   Posterior bundles made before the multiplicative joint-BAL correction encoded
   the BAL optical depth as positive additive flux. They still load but are
   scientifically invalid and must be refitted; this includes all
   ``aug31_def_bal`` bundles.
