GW injection sets
=================

Stage 5 of the Bagpuss pipeline produces a simulated gravitational-wave
injection set by assigning host galaxies from a
:class:`~bagpuss.catalogue.GalaxyCatalogue` to BBH events drawn from a
:class:`~bagpuss.population.PopulationModel`, drawing the remaining extrinsic
parameters, and — optionally — filtering by a detectability model.

The result is an :class:`~bagpuss.injection.InjectionSet` that can be written
to HDF5 for direct consumption by bilby, pycbc, or any other downstream
parameter-estimation tool.

Concepts
--------

Host galaxy assignment
^^^^^^^^^^^^^^^^^^^^^^

Each BBH event is assigned a host galaxy drawn uniformly at random from the
catalogue.  The event inherits the host's sky position (RA, Dec) and
redshift; the luminosity distance is then computed from the redshift via the
background cosmology.

This flat weighting is appropriate when the expected merger rate is
proportional to the number of catalogued galaxies.  For stellar-mass- or
star-formation-rate-weighted host assignment, subclass or wrap
:func:`~bagpuss.injection.create_injection_set`.

Extrinsic parameters
^^^^^^^^^^^^^^^^^^^^

The following parameters are drawn independently of the host galaxy:

* **Inclination** :math:`\theta_{JN}` — uniform in :math:`\cos\theta_{JN}`,
  equivalent to a uniform distribution over the sphere.
* **Polarisation** :math:`\psi` — uniform in :math:`[0, \pi)`.
* **Geocentric merger time** — uniform within the specified observation window
  (defaults to O3: 2019-04-01 to 2020-03-27).

Detectability models
^^^^^^^^^^^^^^^^^^^^^

A :class:`~bagpuss.injection.Detectable` object is a callable that maps a
parameter dictionary to per-event detection probabilities in :math:`[0, 1]`.
Events are retained stochastically with probability :math:`p_{\rm det}`.

The built-in :class:`~bagpuss.injection.DistanceThreshold` applies a hard
cut at a maximum luminosity distance :math:`d_{\max}`.  For more realistic
models (SNR-based, network-dependent), implement the
:class:`~bagpuss.injection.Detectable` interface.

Worked example
--------------

The following example builds a complete end-to-end pipeline from a universe
model through to an injection set, then visualises the injections.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy.cosmology import Planck18

   from bagpuss.catalogue import MagnitudeLimitedSurvey
   from bagpuss.injection import DistanceThreshold, create_injection_set
   from bagpuss.luminosity import SchechterLuminosityModel
   from bagpuss.population import (
       IsotropicSpinDistribution,
       PopulationModel,
       PowerLawPlusPeakMassDistribution,
   )
   from bagpuss.universe import PointProcess, Universe

   rng = np.random.default_rng(42)

   # Stage 1–3: galaxy catalogue
   universe = Universe(
       cosmology=Planck18,
       structure=PointProcess(z_max=0.5),
       luminosity=SchechterLuminosityModel(
           phi_star=1.61e-2, m_star=-19.66, alpha=-1.16,
           m_min=-25.0, m_max=-14.0,
       ),
   )
   galaxies = universe.sample(5_000, rng=rng)
   catalogue = MagnitudeLimitedSurvey(m_lim=19.5).apply(galaxies, Planck18)

   # Stage 4: BBH population
   population = PopulationModel(
       mass=PowerLawPlusPeakMassDistribution(
           alpha=3.5, beta_q=1.4, m_min=5.0, m_max=87.0,
           lambda_peak=0.03, mu_m=34.0, sigma_m=3.6, delta_m=4.8,
       ),
       spin=IsotropicSpinDistribution(),
   )

   # Stage 5: injection set with distance threshold
   injections = create_injection_set(
       catalogue=catalogue,
       population=population,
       cosmology=Planck18,
       n_draw=500,
       detectable=DistanceThreshold(d_max=1_500.0),
       rng=rng,
   )

   fig, axes = plt.subplots(1, 3, figsize=(12, 4))

   ax = axes[0]
   ax.hist(injections.redshift, bins=25, color="steelblue")
   ax.set_xlabel("Redshift $z$")
   ax.set_ylabel("Count")
   ax.set_title("Host galaxy redshifts")

   ax = axes[1]
   ax.hist(injections.m1_source, bins=25, color="coral", alpha=0.8, label=r"$m_1$")
   ax.hist(injections.m2_source, bins=25, color="steelblue", alpha=0.8, label=r"$m_2$")
   ax.set_xlabel(r"Source-frame mass $[M_\odot]$")
   ax.set_ylabel("Count")
   ax.set_title("Component masses")
   ax.legend()

   ax = axes[2]
   ax.scatter(
       np.degrees(injections.ra),
       np.degrees(injections.dec),
       s=4, alpha=0.5, c=injections.luminosity_distance, cmap="viridis",
   )
   ax.set_xlabel("RA [deg]")
   ax.set_ylabel("Dec [deg]")
   ax.set_title("Sky positions (coloured by $d_L$)")

   fig.tight_layout()

The resulting :class:`~bagpuss.injection.InjectionSet` can be saved to HDF5:

.. code-block:: python

   injections.to_hdf5("injections.h5")

   # Load it back
   from bagpuss.injection import InjectionSet
   loaded = InjectionSet.from_hdf5("injections.h5")

The HDF5 file contains one dataset per parameter under the ``/injections``
group, matching the column names expected by bilby's injection infrastructure.

Implementing a custom detectability model
------------------------------------------

.. code-block:: python

   import numpy as np
   from bagpuss.injection import Detectable

   class SNRThreshold(Detectable):
       """Simplified SNR-based detectability."""

       def __init__(self, snr_threshold: float = 8.0) -> None:
           self.snr_threshold = snr_threshold

       def __call__(self, params: dict) -> np.ndarray:
           # Chirp mass proxy (ignores spin corrections)
           m1, m2 = params["m1_source"], params["m2_source"]
           mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
           d_L = params["luminosity_distance"]
           # Rough SNR ∝ Mc^(5/6) / d_L; normalised to SNR=8 at Mc=28, d=500
           snr = 8.0 * (mc / 28.0) ** (5.0 / 6.0) * (500.0 / d_L)
           return (snr >= self.snr_threshold).astype(float)

API reference
-------------

See :doc:`autoapi/bagpuss/injection/index` for the full API reference.
