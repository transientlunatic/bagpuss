Sensitivity injections
======================

A hierarchical analysis (a dark-siren or population inference) needs the
probability that an event drawn from a population :math:`\Lambda` is detected,

.. math::

   P(D|\Lambda) = \int P(D|\theta)\, p(\theta|\Lambda)\, d\theta ,

at every :math:`\Lambda` the sampler visits. This is **not** what
:mod:`bagpuss.injection` produces. That is one realisation of one population
(the "truth" of a mock catalogue), with the undetected events discarded;
:math:`P(D|\Lambda)` is needed under hypotheses other than the truth.

:mod:`bagpuss.sensitivity` instead draws injections from a fixed, deliberately
broad :class:`~bagpuss.sensitivity.InjectionDistribution` that covers the
support of every population of interest, keeps *every* injection (found or
missed, or at least how many there were), and estimates the integral by
importance sampling (Essick et al. 2025, `arXiv:2508.10638
<https://arxiv.org/abs/2508.10638>`_; Farr 2019):

.. math::

   \hat P(D|\Lambda) = \frac{1}{N_{\rm total}}
       \sum_{i\in{\rm found}}
       \frac{p(\theta_i|\Lambda)}{p(\theta_i|\Lambda_{\rm inj})} .

Three details matter:

* :math:`N_{\rm total}` counts **every** injection drawn, including any
  discarded before detection because they are hopeless
  (:meth:`~bagpuss.sensitivity.SensitivityInjectionSet.select` keeps it);
* each injection stores the density it was drawn from, evaluated by the code
  that drew it; and
* the Monte Carlo uncertainty is reported: the estimate's ``n_eff`` should
  exceed about four times the number of events in the analysis
  (:meth:`~bagpuss.sensitivity.DetectionEstimate.is_reliable`).

.. code-block:: python

   import numpy as np
   from astropy.cosmology import Planck18
   from bagpuss.sensitivity import InjectionDistribution, draw_injections

   dist = InjectionDistribution(Planck18, t_start=1.238e9, t_end=1.333e9, z_max=3.0)
   injections = draw_injections(dist, 1_000_000, np.random.default_rng(0))
   snr = my_network_snr(injections)  # your detector model
   estimate = injections.detection_probability(log_p_target, snr, snr_threshold=8.0)
   print(estimate.value, estimate.std, estimate.n_eff)

``log_p_target`` takes a dict of parameter arrays and returns the log density of
the target population in the measure :data:`bagpuss.sensitivity.DENSITY_MEASURE`.

The same detection criterion as the event set
---------------------------------------------

The detector network, duty cycles and SNR threshold must be those of the event
set, or the inference is biased. In the MDC pipeline
(:func:`bagpuss.mdc.pipeline.generate_sensitivity_shard`) the duty-cycle
schedule is drawn from the same seed as in
:func:`~bagpuss.mdc.pipeline.detect_injection_shard`, so the same detectors are
observing at the same times, and the SNR comes from the same backend.

Cosmology
---------

Injections are drawn in source-frame redshift, and the luminosity distance is
computed from the fiducial cosmology. For an :math:`H_0` inference the estimate
must be reweighted by the Jacobian of (redshift, luminosity distance); the
redshift is stored so that this stays explicit.
