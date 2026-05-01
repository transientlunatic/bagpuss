Galaxy catalogues and selection functions
=========================================

A raw :class:`~bagpuss.galaxies.GalaxySet` contains every galaxy in the
simulated volume.  Real surveys can only observe galaxies that are bright
enough for their instruments to detect.  Stage 3 of the Bagpuss pipeline
models this by applying a **selection function** to the full galaxy set,
producing an observed :class:`~bagpuss.catalogue.GalaxyCatalogue`.

Concepts
--------

Selection function
^^^^^^^^^^^^^^^^^^

A :class:`~bagpuss.catalogue.SelectionFunction` takes a
:class:`~bagpuss.galaxies.GalaxySet` and a background cosmology, and returns
the subset of galaxies that would be detected by the survey.  The cosmology
is supplied at call-time rather than at construction, so the same selection
function object can be applied to galaxy sets drawn from different
cosmological models.

Magnitude-limited surveys
^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common observational selection criterion is a **limiting apparent
magnitude** :math:`m_{\rm lim}`.  A galaxy is included in the catalogue if
its apparent magnitude satisfies :math:`m \leq m_{\rm lim}`.

The apparent magnitude follows from the galaxy's intrinsic luminosity and
its luminosity distance:

.. math::

   m = M + \mu, \qquad
   M = M_\odot - 2.5\log_{10}\!\left(\frac{L}{L_\odot}\right), \qquad
   \mu = 5\log_{10}\!\left(\frac{d_L}{10\,\mathrm{pc}}\right)

Because more-distant galaxies have larger distance moduli :math:`\mu`, a
magnitude-limited survey is biased toward intrinsically brighter galaxies at
high redshift — an observational effect that any downstream cosmological
inference must account for.

Worked example
--------------

The code below samples 5 000 galaxies from a Planck 2018 universe out to
:math:`z = 0.5`, applies a limiting magnitude of :math:`m_{\rm lim} = 19.5`,
and visualises the selection cut.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy.cosmology import Planck18

   from bagpuss.catalogue import MagnitudeLimitedSurvey
   from bagpuss.luminosity import SchechterLuminosityModel
   from bagpuss.universe import PointProcess, Universe

   rng = np.random.default_rng(42)
   universe = Universe(
       cosmology=Planck18,
       structure=PointProcess(z_max=0.5),
       luminosity=SchechterLuminosityModel(
           phi_star=1.61e-2,
           m_star=-19.66,
           alpha=-1.16,
           m_min=-25.0,
           m_max=-14.0,
       ),
   )
   galaxies = universe.sample(5_000, rng=rng)

   m_lim = 19.5
   survey = MagnitudeLimitedSurvey(m_lim=m_lim)
   catalogue = survey.apply(galaxies, Planck18)

   # Compute apparent magnitudes for the full set to show rejected galaxies
   d_L_pc = Planck18.luminosity_distance(galaxies.redshifts).to("pc").value
   mu = 5.0 * np.log10(d_L_pc / 10.0)
   M_abs = 4.83 - 2.5 * np.log10(galaxies.luminosities)
   all_m = M_abs + mu

   fig, axes = plt.subplots(1, 2, figsize=(10, 4))

   ax = axes[0]
   rejected = all_m > m_lim
   ax.scatter(
       galaxies.redshifts[rejected], all_m[rejected],
       s=2, color="lightgrey", label="rejected", rasterized=True,
   )
   ax.scatter(
       catalogue.redshifts, catalogue.apparent_magnitudes,
       s=2, color="steelblue", label="selected", rasterized=True,
   )
   ax.axhline(m_lim, color="crimson", linestyle="--", linewidth=1,
              label=f"$m_{{\\rm lim}} = {m_lim}$")
   ax.set_xlabel("Redshift $z$")
   ax.set_ylabel("Apparent magnitude $m$")
   ax.invert_yaxis()
   ax.legend(markerscale=4)

   ax = axes[1]
   bins = np.linspace(0, 0.5, 30)
   ax.hist(galaxies.redshifts, bins=bins, color="lightgrey", label="all galaxies")
   ax.hist(catalogue.redshifts, bins=bins, color="steelblue", alpha=0.8, label="catalogue")
   ax.set_xlabel("Redshift $z$")
   ax.set_ylabel("Count")
   ax.legend()

   n_sel, n_tot = len(catalogue), len(galaxies)
   fig.suptitle(
       f"Magnitude-limited survey ($m_{{\\rm lim}} = {m_lim}$,"
       f" {n_sel} / {n_tot} galaxies selected)"
   )
   fig.tight_layout()

The left panel shows every galaxy's apparent magnitude against its redshift;
the dashed red line marks :math:`m_{\rm lim}`.  The right panel compares the
full redshift distribution with that of the selected catalogue, illustrating
how the cut preferentially removes faint, distant galaxies.

The resulting :class:`~bagpuss.catalogue.GalaxyCatalogue` carries three
arrays that can be passed directly into downstream pipeline stages:

.. code-block:: python

   catalogue.redshifts            # redshifts of selected galaxies
   catalogue.luminosities         # luminosities in solar luminosities
   catalogue.apparent_magnitudes  # apparent magnitudes in the survey band

Implementing a custom selection function
----------------------------------------

Any new survey model can be added by subclassing
:class:`~bagpuss.catalogue.SelectionFunction` and implementing
:meth:`~bagpuss.catalogue.SelectionFunction.apply`.  The only contract is
that the method returns a :class:`~bagpuss.catalogue.GalaxyCatalogue`
containing the selected subset.

.. code-block:: python

   import numpy as np
   from astropy.cosmology import FLRW

   from bagpuss.catalogue import GalaxyCatalogue, SelectionFunction
   from bagpuss.galaxies import GalaxySet

   class RedshiftLimitedSurvey(SelectionFunction):
       """Select all galaxies below a maximum redshift."""

       def __init__(self, z_max: float) -> None:
           self.z_max = z_max

       def apply(self, galaxies: GalaxySet, cosmology: FLRW) -> GalaxyCatalogue:
           mask = galaxies.redshifts <= self.z_max
           return GalaxyCatalogue(
               redshifts=galaxies.redshifts[mask],
               luminosities=galaxies.luminosities[mask],
               apparent_magnitudes=np.full(mask.sum(), np.nan),
           )

API reference
-------------

See :doc:`autoapi/bagpuss/catalogue/index` for the full API reference.
