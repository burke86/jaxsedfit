JAXSEDFit Documentation
=======================

JAXSEDFit is a Bayesian SED fitting package for AGN and galaxies, built with
JAX and NumPyro. It is an **experimental JAX-based port of portions of
CIGALE and GRAHSP**.

If you use JAXSEDFit in scientific work, cite both codes and their
associated papers:

* `CIGALE <https://cigale.lam.fr/>`_: Boquien et al. (2019),
  *CIGALE: a python Code Investigating GALaxy Emission*, A&A, 622, A103
  (`paper <https://doi.org/10.1051/0004-6361/201834156>`__).
* `GRAHSP <https://github.com/JohannesBuchner/GRAHSP>`_: Buchner et al. (2024),
  *Genuine Retrieval of the AGN Host Stellar Population (GRAHSP)*,
  A&A, 692, A161
  (`paper <https://doi.org/10.1051/0004-6361/202449372>`__).

Also follow their citation guidance for the models and templates used
in your analysis. See :doc:`outputs` for posterior definitions and units.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   tutorials
   outputs
   api/index
