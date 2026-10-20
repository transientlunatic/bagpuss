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
