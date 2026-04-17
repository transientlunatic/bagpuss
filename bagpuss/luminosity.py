"""Luminosity distribution models for the bagpuss pipeline.

This module provides concrete implementations of
:class:`~bagpuss.universe.LuminosityModel` that can be slotted directly into a
:class:`~bagpuss.universe.Universe`.

The primary model provided here is :class:`SchechterLuminosityModel`, which
samples galaxy luminosities from the Schechter (1976) luminosity function
using the `~astropy.modeling.models.Schechter1D` model from astropy.
"""

from __future__ import annotations

import numpy as np
from astropy.modeling.models import Schechter1D

from bagpuss.universe import LuminosityModel

__all__: list[str] = ["SchechterLuminosityModel"]

#: Absolute magnitude of the Sun (bolometric), used as the default reference
#: band when converting magnitudes to solar luminosities.
_M_SUN_BOLOMETRIC: float = 4.83

_GRID_SIZE: int = 10_000


class SchechterLuminosityModel(LuminosityModel):
    r"""A luminosity model based on the Schechter (1976) luminosity function.

    Galaxy luminosities are sampled from the Schechter luminosity function,
    parameterised in terms of absolute magnitudes.  Sampling is performed via
    an inverse-CDF method on a fine magnitude grid.  Magnitudes are then
    converted to solar luminosities for use in the rest of the pipeline.

    Parameters
    ----------
    phi_star : float
        Normalisation factor of the Schechter function in units of number
        density (e.g. Mpc\ :sup:`−3` mag\ :sup:`−1`).  Does not affect the
        *shape* of the luminosity distribution and therefore does not influence
        which luminosities are sampled; it is stored for future use in
        number-count predictions.
    m_star : float
        Characteristic absolute magnitude at the "knee" of the luminosity
        function, where the power-law turns over into an exponential cutoff.
    alpha : float
        Faint-end power-law slope.  Typically negative (e.g. ``-1.2`` to
        ``-2.0``).
    m_min : float
        Bright-end absolute magnitude limit of the sampling range.  Because
        brighter sources have more negative magnitudes, ``m_min < m_max``.
    m_max : float
        Faint-end absolute magnitude limit of the sampling range.
    m_sun : float, optional
        Absolute magnitude of the Sun in the relevant photometric band, used to
        convert sampled magnitudes to solar luminosities via

        .. math::

            \frac{L}{L_\odot} = 10^{0.4 \, (M_\odot - M)} .

        Defaults to ``4.83`` (bolometric).

    Raises
    ------
    ValueError
        If ``m_min >= m_max``.

    Notes
    -----
    The Schechter (1976) luminosity function in magnitude form is:

    .. math::

        \phi(M) \, \mathrm{d}M = 0.4 \ln(10) \; \phi^{*}
            \left[ 10^{0.4 (M^{*} - M)} \right]^{\alpha + 1}
            \exp\!\left[ -10^{0.4 (M^{*} - M)} \right] \mathrm{d}M

    Sampling proceeds by:

    1. Evaluating :math:`\phi(M)` on a grid of ``10 000`` points in
       ``[m_min, m_max]``.
    2. Computing the cumulative distribution function (CDF) via the
       trapezoidal rule.
    3. Drawing :math:`u \sim \mathrm{Uniform}(0, 1)` and inverting the CDF
       via linear interpolation.

    References
    ----------
    .. [1] Schechter, P. 1976, ApJ, 203, 297.
           https://ui.adsabs.harvard.edu/abs/1976ApJ...203..297S/abstract
    """

    def __init__(
        self,
        phi_star: float,
        m_star: float,
        alpha: float,
        m_min: float,
        m_max: float,
        m_sun: float = _M_SUN_BOLOMETRIC,
    ) -> None:
        if m_min >= m_max:
            raise ValueError(
                f"m_min must be less than m_max, got m_min={m_min!r}, m_max={m_max!r}"
            )
        self.phi_star = phi_star
        self.m_star = m_star
        self.alpha = alpha
        self.m_min = m_min
        self.m_max = m_max
        self.m_sun = m_sun
        self._model = Schechter1D(phi_star=phi_star, m_star=m_star, alpha=alpha)

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw luminosities from the Schechter luminosity function.

        Parameters
        ----------
        n : int
            Number of luminosities to sample.
        rng : numpy.random.Generator or None, optional
            Random number generator.  If *None*, ``numpy.random.default_rng()``
            is used.

        Returns
        -------
        numpy.ndarray
            Luminosities in solar luminosities, shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()

        m_grid = np.linspace(self.m_min, self.m_max, _GRID_SIZE)
        pdf = self._model(m_grid)

        # Build the CDF via the trapezoidal rule and normalise to [0, 1]
        delta = np.diff(m_grid)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * delta)])
        cdf /= cdf[-1]

        u = rng.uniform(0.0, 1.0, size=n)
        magnitudes = np.interp(u, cdf, m_grid)

        # Convert absolute magnitudes to solar luminosities
        return 10.0 ** (0.4 * (self.m_sun - magnitudes))
