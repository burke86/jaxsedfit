JAXSEDFit Documentation
=======================

JAXSEDFit is a Bayesian SED fitting package for AGN and galaxies, built with
JAX and NumPyro. It is an **experimental JAX-based port of portions of
CIGALE, GRAHSP and PyQSOFit**. It fits broadband photometry, spectra, or both together:
in a joint fit, one physical model of the host galaxy and the AGN describes the
photometry and the spectrum at once (see :doc:`spectroscopy`).

If you use JAXSEDFit in scientific work, cite these codes and their
associated papers:

* `CIGALE <https://cigale.lam.fr/>`_: Boquien et al. (2019),
  *CIGALE: a python Code Investigating GALaxy Emission*, A&A, 622, A103
  (`paper <https://doi.org/10.1051/0004-6361/201834156>`__).
* `GRAHSP <https://github.com/JohannesBuchner/GRAHSP>`_: Buchner et al. (2024),
  *Genuine Retrieval of the AGN Host Stellar Population (GRAHSP)*,
  A&A, 692, A161
  (`paper <https://doi.org/10.1051/0004-6361/202449372>`__).
* `PyQSOFit <https://github.com/legolason/PyQSOFit>`_: Guo, Shen & Wang (2018),
  *PyQSOFit: Python code to fit the spectrum of quasars*, Astrophysics Source
  Code Library, ascl:1809.008 (`ASCL <https://ascl.net/1809.008>`__), and
  Shen et al. (2019), ApJS, 241, 34
  (`paper <https://doi.org/10.3847/1538-4365/ab074f>`__).

Also follow their citation guidance for the models and templates used
in your analysis. See :doc:`outputs` for posterior definitions and units.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User guide

   photometry
   host
   agn
   spectroscopy
   likelihood
   inference
   batch
   results

.. toctree::
   :maxdepth: 2
   :caption: Reference

   outputs
   filters
   api/index
