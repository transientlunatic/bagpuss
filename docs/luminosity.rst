Luminosity models
=================

A luminosity model determines how galaxy luminosities are distributed in the
simulated Universe.  All luminosity models are subclasses of
:class:`~bagpuss.universe.LuminosityModel` and implement a single
:meth:`~bagpuss.universe.LuminosityModel.sample` method.

The Schechter luminosity function
----------------------------------

The most widely used empirical luminosity function in observational cosmology
is the Schechter (1976) function.  In magnitude form it reads:

.. math::

    \phi(M) \, \mathrm{d}M = 0.4 \ln(10) \; \phi^{*}
        \left[ 10^{0.4 (M^{*} - M)} \right]^{\alpha + 1}
        \exp\!\left[ -10^{0.4 (M^{*} - M)} \right] \mathrm{d}M

The three parameters are:

* :math:`\phi^{*}` — the normalisation (number density per magnitude interval),
* :math:`M^{*}` — the characteristic "knee" magnitude, and
* :math:`\alpha` — the faint-end power-law slope (typically negative).

The function rises steeply towards fainter (more positive) magnitudes and drops
exponentially bright-ward of :math:`M^{*}`.

:class:`~bagpuss.luminosity.SchechterLuminosityModel` wraps
`astropy.modeling.models.Schechter1D`_ and samples from the distribution using
an inverse-CDF method.  Sampled magnitudes are converted to solar luminosities
via:

.. math::

    \frac{L}{L_\odot} = 10^{0.4 \, (M_\odot - M)}

.. _astropy.modeling.models.Schechter1D: https://docs.astropy.org/en/stable/api/astropy.modeling.models.Schechter1D.html

Example usage
-------------

.. code-block:: python

    import numpy as np
    from astropy.cosmology import Planck18

    from bagpuss.luminosity import SchechterLuminosityModel
    from bagpuss.universe import Universe, PointProcess

    # B-band Schechter parameters (Norberg et al. 2002, approximate)
    lum_model = SchechterLuminosityModel(
        phi_star=1.61e-2,  # h^3 Mpc^-3
        m_star=-19.66,     # absolute B-band magnitude
        alpha=-1.16,
        m_min=-25.0,       # bright limit
        m_max=-14.0,       # faint limit
    )

    universe = Universe(
        cosmology=Planck18,
        structure=PointProcess(z_max=0.5),
        luminosity=lum_model,
    )

    galaxies = universe.sample(10_000, rng=np.random.default_rng(42))
    print(f"Luminosity range: {galaxies.luminosities.min():.2e} – "
          f"{galaxies.luminosities.max():.2e} L_sun")

References
----------

.. [Schechter1976] Schechter, P. 1976, ApJ, 203, 297.
   https://ui.adsabs.harvard.edu/abs/1976ApJ...203..297S/abstract

API reference
-------------

.. autoapi-members::
   :modules: bagpuss.luminosity
