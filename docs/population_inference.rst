Population inference
====================

:mod:`bagpuss.population_inference` checks that a catalogue can be used for
what it is for: it fits the mass and redshift distributions of the detectable
events with `gwpopulation <https://gwpopulation.readthedocs.io>`_, corrects
for the selection effect with the :doc:`sensitivity injections <sensitivity>`,
and reports where the true hyperparameters fall in the posterior.

.. code-block:: console

   pip install 'bagpuss[population]'
   bagpuss mdc fit-population path/to/release --out-dir fit

The fit is in :math:`(m_1, q, z)`: spins, orientation, sky position and merger
time are marginalised. They are independent of the masses and redshift both in
the injection distribution and in the true population, so this is exact:
:func:`~bagpuss.population_inference.selection_data` divides them out of the
stored draw density
(:meth:`~bagpuss.sensitivity.InjectionDistribution.nuisance_log_prob`) and
changes the measure from :math:`dm_1\,dm_2` to :math:`dm_1\,dq`. The unit
tests check that ``gwpopulation``'s mass and redshift models are bagpuss's, and
that ``gwpopulation.vt.ResamplingVT`` on this data reproduces
:meth:`~bagpuss.sensitivity.SensitivityInjectionSet.detection_probability`.

The events are used at their true parameters (one sample each), which isolates
the population model and the selection function from parameter estimation.
By default the fitted hyperparameters are ``alpha``, ``beta``, ``lam``,
``mpp``, ``sigpp`` and the redshift index ``lamb``; the others are held at
their true values (``--free`` changes this).

The command exits non-zero if any true value lies outside the central credible
interval (``--credible-level``, default 0.99; with several hyperparameters a
90% interval would be missed by chance most of the time), and writes
``summary.json`` with each parameter's quantile and :math:`z`-score.

SimplePE posteriors
-------------------

``--simplepe-dir`` (the asimov project's ``working`` directory) replaces the
true parameters with the ``SimplePE`` posteriors of the detectable events.
:func:`~bagpuss.population_inference.simplepe_samples` converts the
detector-frame chirp mass, symmetric mass ratio and distance to
:math:`(m_1, q, z)` with the release's cosmology and divides out the prior
``SimplePE`` assumed with its defaults (uniform detector-frame component masses
and a uniform-volume distance prior, :math:`\pi \propto d_L^2`), which in the
source-frame variables is :math:`d_L^2\,(1+z)^2\,m_1\,\mathrm{d}d_L/\mathrm{d}z`.
The importance weights are used to resample each event to the same number of
samples. The unit tests check the conversion by sampling the prior and
confirming that weighting by its inverse is flat.

Events with no posterior (no skymap, or the analysis failed) are excluded; they
are listed in ``summary.json``. ``--drop-unsupported`` also drops events whose
posterior has no sample inside the true population, which make the likelihood
exactly zero; these are parameter-estimation failures, and are listed too.
