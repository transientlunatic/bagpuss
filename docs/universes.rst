Universe simulations
====================

The first thing Bagpuss needs to do is to simulate a Universe.
This means defining a cosmology, a large-scale structure model, and a
luminosity model, then combining them into a :class:`~bagpuss.universe.Universe`
that can produce galaxy samples on demand.

The design is deliberately modular: the :class:`~bagpuss.universe.Structure`
and :class:`~bagpuss.universe.LuminosityModel` classes are abstract, so new
physical models can be added without touching any other part of the pipeline.

Concepts
--------

Cosmology
^^^^^^^^^

The background cosmology is any `astropy`_ cosmology object — for example
``astropy.cosmology.Planck18``.  It is passed directly to the structure model
so that distance or volume computations are internally consistent.

.. _astropy: https://docs.astropy.org/en/stable/cosmology/index.html

Structure models
^^^^^^^^^^^^^^^^

A :class:`~bagpuss.universe.Structure` model draws the *spatial* distribution
of galaxies, specifically their redshifts.  The only requirement is that a
subclass implements :meth:`~bagpuss.universe.Structure.sample_redshifts`.

The built-in structure model is :class:`~bagpuss.universe.PointProcess`, which
assumes galaxies are uniformly distributed in comoving volume — i.e. a Poisson
point process.  This is the simplest physically motivated prior and a common
choice for mock catalogues.

Luminosity models
^^^^^^^^^^^^^^^^^

A :class:`~bagpuss.universe.LuminosityModel` draws luminosities independently
of galaxy positions.  Subclasses implement
:meth:`~bagpuss.universe.LuminosityModel.sample`.

The built-in luminosity model is
:class:`~bagpuss.luminosity.SchechterLuminosityModel`, which samples from the
Schechter (1976) luminosity function.  See :doc:`luminosity` for full details
and parameter descriptions.

Sampling galaxies
-----------------

Once a :class:`~bagpuss.universe.Universe` is constructed, calling
:meth:`~bagpuss.universe.Universe.sample` returns a
:class:`~bagpuss.galaxies.GalaxySet` containing redshifts and luminosities for
the requested number of galaxies.

.. code-block:: python

    import numpy as np
    from astropy.cosmology import Planck18

    from bagpuss.luminosity import SchechterLuminosityModel
    from bagpuss.universe import Universe, PointProcess

    universe = Universe(
        cosmology=Planck18,
        structure=PointProcess(z_max=1.0),
        luminosity=SchechterLuminosityModel(
            phi_star=1.61e-2,
            m_star=-19.66,
            alpha=-1.16,
            m_min=-25.0,
            m_max=-14.0,
        ),
    )

    galaxies = universe.sample(10_000, rng=np.random.default_rng(0))
    print(f"Sampled {len(galaxies)} galaxies up to z = {galaxies.redshifts.max():.2f}")

API reference
-------------

.. autoapi-members::
   :modules: bagpuss.universe
