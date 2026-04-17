Galaxy sets
===========

A :class:`~bagpuss.galaxies.GalaxySet` is the data container that carries
the output of Stage 1 (universe simulation) into Stage 3 (catalogue
construction).  It holds per-galaxy redshifts and luminosities as NumPy
arrays, making it straightforward to apply vectorised selection functions or
feed data directly into downstream analysis code.

Creating a galaxy set
---------------------

In normal use a :class:`~bagpuss.galaxies.GalaxySet` is produced by calling
:meth:`Universe.sample() <bagpuss.universe.Universe.sample>` rather than
constructed directly.  See :doc:`universes` for the full workflow.

.. code-block:: python

    from astropy.cosmology import Planck18
    from bagpuss.universe import Universe, PointProcess
    # (supply your own LuminosityModel)

    universe = Universe(
        cosmology=Planck18,
        structure=PointProcess(z_max=1.0),
        luminosity=my_luminosity_model,
    )
    galaxies = universe.sample(10_000)

    print(galaxies.redshifts.shape)    # (10000,)
    print(galaxies.luminosities.shape) # (10000,)
    print(len(galaxies))               # 10000

API reference
-------------

.. autoapi-members::
   :modules: bagpuss.galaxies
