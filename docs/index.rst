Bagpuss
=======

Bagpuss is a Python package for generating simulated galaxy catalogues and
gravitational-wave catalogues for testing cosmological inference pipelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   universes
   luminosity
   galaxies
   catalogue
   population
   injection
   mdc
   plotting
   autoapi/index

Pipeline stages
---------------

Bagpuss runs a seven-stage simulation pipeline:

1. **Universe simulation** — draw a cosmology, galaxy formation model, and luminosity model
2. **Galaxy set** — sample galaxies from the simulated distribution
3. **Galaxy catalogue** — apply a selection function to produce a fake observed catalogue
4. **BBH population** — simulate binary black holes across the galaxy set
5. **GW injection** — inject signals into noise and threshold by detector network
6. **Parameter estimation** — produce posteriors and skymaps for each observed event
7. **Cosmological inference** — feed data products into a downstream inference pipeline
