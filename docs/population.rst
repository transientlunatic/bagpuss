BBH population models
=====================

Stage 4 of the Bagpuss pipeline draws binary black hole (BBH) intrinsic
parameters — masses and spins — from a population model.  The design mirrors
the :class:`~bagpuss.universe.Universe` pattern: two abstract components
(:class:`~bagpuss.population.MassDistribution` and
:class:`~bagpuss.population.SpinDistribution`) are composed into a
:class:`~bagpuss.population.PopulationModel` that produces a
:class:`~bagpuss.population.BBHSet` on demand.

The hyperparameter names in this module match those used by
`gwpopulation <https://gwpopulation.readthedocs.io>`_, so gwpopulation model
objects can be used as drop-in replacements for the concrete classes
provided here.

Concepts
--------

Mass distribution
^^^^^^^^^^^^^^^^^

A :class:`~bagpuss.population.MassDistribution` draws source-frame primary
and secondary masses :math:`m_1 \geq m_2 \geq m_{\min}`.  The built-in
model is :class:`~bagpuss.population.PowerLawPlusPeakMassDistribution`
(Talbot & Thrane 2018), which combines a truncated power law with an
optional Gaussian peak and a smooth low-mass taper:

.. math::

   p(m_1) \propto \bigl[(1 - \lambda_{\rm peak})\,m_1^{-\alpha}
       + \lambda_{\rm peak}\,\mathcal{N}(m_1;\,\mu_m, \sigma_m)\bigr]
       \cdot S(m_1;\,m_{\min}, \delta_m)

   p(q \mid m_1) \propto q^{\beta_q} \cdot S(m_1 q;\,m_{\min}, \delta_m)

where :math:`q = m_2/m_1` is the mass ratio and :math:`S` is a smooth
tapering window that turns off below :math:`m_{\min} + \delta_m`.

Spin distribution
^^^^^^^^^^^^^^^^^

A :class:`~bagpuss.population.SpinDistribution` draws six spin parameters
per event: magnitudes :math:`a_1, a_2`; tilt cosines
:math:`\cos\theta_1, \cos\theta_2`; and azimuthal angles
:math:`\phi_{12}, \phi_{JL}`.

Two built-in models are provided:

* :class:`~bagpuss.population.IsotropicSpinDistribution` — uniform magnitudes
  and isotropic tilts.  The simplest reference model.
* :class:`~bagpuss.population.DefaultSpinDistribution` — magnitudes from a
  Beta distribution; tilts from a mixture of an isotropic component and a
  preferentially aligned component (Abbott et al. 2021 / gwpopulation
  ``Default``).

Worked example
--------------

The snippet below builds an LVK-like population model using the O3 best-fit
hyperparameters, draws 2 000 events, and plots the resulting mass distribution.

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt

   from bagpuss.population import (
       IsotropicSpinDistribution,
       PopulationModel,
       PowerLawPlusPeakMassDistribution,
   )

   rng = np.random.default_rng(42)

   population = PopulationModel(
       mass=PowerLawPlusPeakMassDistribution(
           alpha=3.5,
           beta_q=1.4,
           m_min=5.0,
           m_max=87.0,
           lambda_peak=0.03,
           mu_m=34.0,
           sigma_m=3.6,
           delta_m=4.8,
       ),
       spin=IsotropicSpinDistribution(),
   )

   bbh = population.sample(2_000, rng=rng)

   fig, axes = plt.subplots(1, 2, figsize=(10, 4))

   ax = axes[0]
   ax.hist2d(bbh.m2_source, bbh.m1_source, bins=40, cmap="Blues")
   ax.set_xlabel(r"$m_2\;[M_\odot]$")
   ax.set_ylabel(r"$m_1\;[M_\odot]$")
   ax.set_title("Component masses")
   ax.plot([0, 90], [0, 90], "k--", lw=0.8, alpha=0.5)

   ax = axes[1]
   ax.hist(bbh.a1, bins=30, alpha=0.7, label=r"$a_1$")
   ax.hist(bbh.a2, bins=30, alpha=0.7, label=r"$a_2$")
   ax.set_xlabel("Spin magnitude")
   ax.set_ylabel("Count")
   ax.set_title("Spin magnitudes (isotropic)")
   ax.legend()

   fig.tight_layout()

Using the LVK default spin model
---------------------------------

To use physically motivated spin tilts (Beta magnitudes, preferentially
aligned tilts), swap :class:`~bagpuss.population.IsotropicSpinDistribution`
for :class:`~bagpuss.population.DefaultSpinDistribution`:

.. code-block:: python

   from bagpuss.population import DefaultSpinDistribution, PopulationModel

   population = PopulationModel(
       mass=...,
       spin=DefaultSpinDistribution(
           alpha_chi=2.0,
           beta_chi=4.0,
           xi_spin=0.6,
           sigma_spin=1.5,
       ),
   )

Implementing a custom model
-----------------------------

Any new mass or spin model can be added by subclassing the relevant ABC.

.. code-block:: python

   import numpy as np
   from bagpuss.population import MassDistribution

   class FlatMassDistribution(MassDistribution):
       """Equal-mass binaries drawn uniformly in [m_min, m_max]."""

       def __init__(self, m_min: float, m_max: float) -> None:
           self.m_min = m_min
           self.m_max = m_max

       def sample(
           self, n: int, rng: np.random.Generator | None = None
       ) -> tuple[np.ndarray, np.ndarray]:
           if rng is None:
               rng = np.random.default_rng()
           m = rng.uniform(self.m_min, self.m_max, size=n)
           return m, m.copy()

API reference
-------------

See :doc:`autoapi/bagpuss/population/index` for the full API reference.
