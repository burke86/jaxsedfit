Installation
============

``jaxsedfit`` requires Python 3.10 or newer. Install it from the repository root:

.. code-block:: bash

   python -m pip install .

SSP templates
-------------

The host-galaxy model needs a DSPS SSP template file. A continuum-only FSPS
template is recommended, because ``jaxsedfit`` models nebular emission separately:

.. code-block:: bash

   curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_continuum_fsps_v3.2_lgmet_age.h5

Pass the file's path as ``cfg.galaxy.dsps_ssp_fn``. Other DSPS template files are
available at https://halos.as.arizona.edu/suchethacooray/dsps_ssp/; set
``GalaxyConfig.ssp_imf`` and the metallicity settings to match the file you use
(see :doc:`host`).

Dust maps
---------

Milky Way dereddening (``Observation(apply_mw_deredden=True)``) uses the SFD maps
through ``dustmaps``. Download them once:

.. code-block:: python

   from dustmaps.config import config
   config["data_dir"] = "/path/to/dustmaps"

   import dustmaps.sfd
   dustmaps.sfd.fetch()
