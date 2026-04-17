Visualising a sampled universe
==============================

This tutorial walks through the two ways to generate a heatmap of a simulated
galaxy catalogue: via the Python API and via the command-line tool.

Both approaches produce a 2-D density plot with redshift on the x-axis and
:math:`\log_{10}(L/L_\odot)` on the y-axis, coloured by the number of galaxies
per bin.

Python API
----------

The :func:`~bagpuss.plotting.plot_universe_heatmap` function takes a
:class:`~bagpuss.galaxies.GalaxySet` and returns a
:class:`matplotlib.figure.Figure`.

The snippet below builds a standard Planck 2018 universe with a
:class:`~bagpuss.universe.PointProcess` structure model and a
:class:`~bagpuss.luminosity.SchechterLuminosityModel` luminosity function,
samples 10 000 galaxies, and plots the result:

.. plot::

   import numpy as np
   from astropy.cosmology import Planck18

   from bagpuss.luminosity import SchechterLuminosityModel
   from bagpuss.plotting import plot_universe_heatmap
   from bagpuss.universe import PointProcess, Universe

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

   galaxies = universe.sample(10_000, rng=np.random.default_rng(42))
   plot_universe_heatmap(galaxies)

The function accepts several keyword arguments for customisation:

.. code-block:: python

   plot_universe_heatmap(
       galaxies,
       bins=80,
       title="My universe",
       colormap="plasma",
       colorbar_label="Galaxy count",
   )

To save the figure, use the returned :class:`~matplotlib.figure.Figure` object
directly:

.. code-block:: python

   fig = plot_universe_heatmap(galaxies)
   fig.savefig("heatmap.pdf", dpi=150, bbox_inches="tight")

Command-line interface
----------------------

The same plot can be produced entirely from the terminal.  Install bagpuss and
run:

.. code-block:: console

   $ bagpuss plot heatmap --n-galaxies 10000 --seed 42 --output heatmap.png

This writes ``heatmap.png`` to the current directory.  Use ``--help`` to see all
available options:

.. code-block:: console

   $ bagpuss plot heatmap --help

For detailed parameter control, write a YAML config file and pass it with
``--config``.  Any option that appears in both the file and on the command line
is resolved in favour of the command line.

Example config file (``universe.yaml``):

.. code-block:: yaml

   model:
     cosmology: Planck18
     n_galaxies: 10000
     seed: 42

   structure:
     point_process:
       z_max: 1.0

   luminosity:
     schechter:
       phi_star: 1.61e-2
       m_star: -19.66
       alpha: -1.16
       m_min: -25.0
       m_max: -14.0

   plot:
     bins: 60
     colormap: inferno
     output: heatmap.png

Run it with:

.. code-block:: console

   $ bagpuss plot heatmap --config universe.yaml
