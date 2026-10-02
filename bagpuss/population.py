"""BBH population models — Stage 4 of the bagpuss pipeline.

Samples binary black hole (BBH) intrinsic parameters from a population
model.  The two primary extension points are :class:`MassDistribution` and
:class:`SpinDistribution`, which are composed into a :class:`PopulationModel`
following the same pattern as :class:`~bagpuss.universe.Universe`.

The hyperparameter conventions in this module match those of `gwpopulation
<https://gwpopulation.readthedocs.io>`_, so that gwpopulation model objects
can be used as drop-in replacements for the concrete classes provided here.

References
----------
.. [Talbot2018] Talbot & Thrane 2018, ApJ 856 173.
   https://doi.org/10.3847/1538-4357/aab34c
.. [LVK2021] Abbott et al. 2021, ApJ 913 L7.
   https://doi.org/10.3847/2041-8213/abe949
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from astropy.cosmology import FLRW
from scipy.stats import beta as beta_dist
from scipy.stats import truncnorm

from bagpuss.universe import Structure

__all__: list[str] = [
    "BBHSet",
    "MassDistribution",
    "SpinDistribution",
    "PopulationModel",
    "PowerLawPlusPeakMassDistribution",
    "IsotropicSpinDistribution",
    "DefaultSpinDistribution",
    "MergerRate",
    "ConstantMergerRate",
    "expected_n_mergers",
    "sample_merger_redshifts",
    "SECONDS_PER_YEAR",
]

_GRID_SIZE: int = 1_000

#: Grid resolution used to numerically integrate/sample the rate-weighted
#: redshift distribution (see :func:`expected_n_mergers` and
#: :func:`sample_merger_redshifts`).
_RATE_GRID_SIZE: int = 10_000

#: Julian year in seconds (365.25 days), used to convert an observation
#: window in GPS seconds into years for the rate integral.
SECONDS_PER_YEAR: float = 365.25 * 86400.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class BBHSet:
    r"""A set of BBH intrinsic parameters sampled from a population model.

    All angular quantities are in radians; masses are in solar masses.

    Parameters
    ----------
    m1_source : numpy.ndarray
        Source-frame primary mass in :math:`M_\odot`, shape ``(n,)``.
        Always satisfies ``m1_source >= m2_source``.
    m2_source : numpy.ndarray
        Source-frame secondary mass in :math:`M_\\odot`, shape ``(n,)``.
    a1 : numpy.ndarray
        Primary spin magnitude, shape ``(n,)``.  Values in ``[0, 1]``.
    a2 : numpy.ndarray
        Secondary spin magnitude, shape ``(n,)``.  Values in ``[0, 1]``.
    cos_tilt1 : numpy.ndarray
        Cosine of the primary spin tilt angle relative to the orbital angular
        momentum, shape ``(n,)``.  Values in ``[-1, 1]``.
    cos_tilt2 : numpy.ndarray
        Cosine of the secondary spin tilt angle, shape ``(n,)``.
    phi12 : numpy.ndarray
        Azimuthal angle between the in-plane spin components, shape ``(n,)``.
        Values in ``[0, 2π)``.
    phi_jl : numpy.ndarray
        Azimuthal angle between the total angular momentum and the orbital
        angular momentum, shape ``(n,)``.  Values in ``[0, 2π)``.
    """

    m1_source: np.ndarray
    m2_source: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    cos_tilt1: np.ndarray
    cos_tilt2: np.ndarray
    phi12: np.ndarray
    phi_jl: np.ndarray

    def __len__(self) -> int:
        """Return the number of BBH events."""
        return int(self.m1_source.shape[0])


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class MassDistribution(ABC):
    """Abstract base class for BBH mass distributions.

    Subclasses must implement :meth:`sample`, which returns arrays of
    source-frame primary and secondary masses.
    """

    @abstractmethod
    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Draw source-frame masses for a set of BBH events.

        Parameters
        ----------
        n : int
            Number of events to sample.
        rng : numpy.random.Generator or None, optional
            Random number generator.

        Returns
        -------
        m1_source : numpy.ndarray
            Primary masses in :math:`M_\odot`, shape ``(n,)``.
        m2_source : numpy.ndarray
            Secondary masses in :math:`M_\odot`, shape ``(n,)``.
        """


class SpinDistribution(ABC):
    """Abstract base class for BBH spin distributions.

    Subclasses must implement :meth:`sample`, which returns six spin
    parameter arrays.
    """

    @abstractmethod
    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Draw spin parameters for a set of BBH events.

        Parameters
        ----------
        n : int
            Number of events to sample.
        rng : numpy.random.Generator or None, optional
            Random number generator.

        Returns
        -------
        a1, a2 : numpy.ndarray
            Spin magnitudes, shape ``(n,)``.
        cos_tilt1, cos_tilt2 : numpy.ndarray
            Cosines of spin tilt angles, shape ``(n,)``.
        phi12, phi_jl : numpy.ndarray
            Azimuthal spin angles in radians, shape ``(n,)``.
        """


# ---------------------------------------------------------------------------
# PopulationModel
# ---------------------------------------------------------------------------


class PopulationModel:
    """A BBH population model composed of a mass and spin distribution.

    Wires together a :class:`MassDistribution` and a :class:`SpinDistribution`
    to produce complete sets of BBH intrinsic parameters via :meth:`sample`.

    Parameters
    ----------
    mass : MassDistribution
        Distribution over primary and secondary masses.
    spin : SpinDistribution
        Distribution over spin magnitudes, tilts, and azimuthal angles.

    Examples
    --------
    >>> import numpy as np
    >>> pop = PopulationModel(
    ...     mass=PowerLawPlusPeakMassDistribution(
    ...         alpha=3.5, beta_q=1.4, m_min=5.0, m_max=87.0,
    ...         lambda_peak=0.03, mu_m=34.0, sigma_m=3.6, delta_m=4.8,
    ...     ),
    ...     spin=IsotropicSpinDistribution(),
    ... )
    >>> bbh = pop.sample(1000, rng=np.random.default_rng(42))
    >>> len(bbh)
    1000
    """

    def __init__(self, mass: MassDistribution, spin: SpinDistribution) -> None:
        self.mass = mass
        self.spin = spin

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> BBHSet:
        """Sample intrinsic BBH parameters from the population.

        Parameters
        ----------
        n : int
            Number of BBH events to sample.
        rng : numpy.random.Generator or None, optional
            Random number generator passed to both sub-models.

        Returns
        -------
        BBHSet
            A set of ``n`` BBH events with fully sampled intrinsic parameters.
        """
        m1, m2 = self.mass.sample(n, rng)
        a1, a2, cos_tilt1, cos_tilt2, phi12, phi_jl = self.spin.sample(n, rng)
        return BBHSet(
            m1_source=m1,
            m2_source=m2,
            a1=a1,
            a2=a2,
            cos_tilt1=cos_tilt1,
            cos_tilt2=cos_tilt2,
            phi12=phi12,
            phi_jl=phi_jl,
        )


# ---------------------------------------------------------------------------
# Concrete mass distributions
# ---------------------------------------------------------------------------


def _smoothing(m: np.ndarray, m_min: float, delta_m: float) -> np.ndarray:
    r"""Tapering window that smoothly turns on at ``m_min``.

    Implements the window :math:`S(m; m_{\min}, \delta_m)` from
    Talbot & Thrane (2018):

    .. math::

        S(m) = \begin{cases}
            0 & m \leq m_{\min} \\
            \left[\exp\!\left(\frac{\delta_m}{m - m_{\min}}
              + \frac{\delta_m}{m - m_{\min} - \delta_m}\right) + 1\right]^{-1}
              & m_{\min} < m < m_{\min} + \delta_m \\
            1 & m \geq m_{\min} + \delta_m
        \end{cases}

    When ``delta_m = 0`` the function reduces to a step at ``m_min``.
    """
    if delta_m == 0.0:
        return np.where(m >= m_min, 1.0, 0.0)

    out = np.ones_like(m, dtype=float)
    lo = m <= m_min
    hi = m >= m_min + delta_m
    transition = ~lo & ~hi
    if np.any(transition):
        mt = m[transition]
        f = delta_m / (mt - m_min) + delta_m / (mt - m_min - delta_m)
        with np.errstate(over="ignore"):
            out[transition] = 1.0 / (np.exp(f) + 1.0)
    out[lo] = 0.0
    out[hi] = 1.0
    return out


class PowerLawPlusPeakMassDistribution(MassDistribution):
    r"""Power-law-plus-Gaussian-peak mass distribution (Talbot & Thrane 2018).

    The primary mass :math:`m_1` is drawn from a mixture of a truncated power
    law and a Gaussian peak, both smoothed near :math:`m_{\min}`:

    .. math::

        p(m_1) \propto \bigl[(1 - \lambda_{\rm peak})\,m_1^{-\alpha}
            + \lambda_{\rm peak}\,\mathcal{N}(m_1;\,\mu_m, \sigma_m)\bigr]
            \cdot S(m_1;\,m_{\min}, \delta_m)

    The mass ratio :math:`q = m_2 / m_1` is drawn from a power law
    conditioned on :math:`m_1`:

    .. math::

        p(q \mid m_1) \propto q^{\beta_q}
            \cdot S(m_1 q;\,m_{\min}, \delta_m)

    Both marginals are sampled via grid-based inverse-CDF.

    The hyperparameter names and conventions match those used by
    `gwpopulation <https://gwpopulation.readthedocs.io>`_'s
    ``SinglePeakSmoothedMassDistribution``.

    Parameters
    ----------
    alpha : float
        Power-law index for the primary mass distribution.  Positive values
        give a falling spectrum (more low-mass events).
    beta_q : float
        Power-law index for the mass-ratio distribution.
    m_min : float
        Minimum component mass in :math:`M_\\odot`.
    m_max : float
        Maximum primary mass in :math:`M_\\odot`.
    lambda_peak : float
        Fraction of events in the Gaussian peak.  Must be in ``[0, 1]``.
    mu_m : float
        Mean of the Gaussian peak in :math:`M_\\odot`.
    sigma_m : float
        Standard deviation of the Gaussian peak in :math:`M_\\odot`.
    delta_m : float
        Width of the low-mass smoothing window in :math:`M_\\odot`.  Set to
        ``0`` to disable smoothing and use a hard cut at ``m_min``.
    """

    def __init__(
        self,
        alpha: float,
        beta_q: float,
        m_min: float,
        m_max: float,
        lambda_peak: float,
        mu_m: float,
        sigma_m: float,
        delta_m: float,
    ) -> None:
        if m_min >= m_max:
            raise ValueError(
                f"m_min must be less than m_max, got m_min={m_min}, m_max={m_max}"
            )
        if not (0.0 <= lambda_peak <= 1.0):
            raise ValueError(f"lambda_peak must be in [0, 1], got {lambda_peak}")
        self.alpha = alpha
        self.beta_q = beta_q
        self.m_min = m_min
        self.m_max = m_max
        self.lambda_peak = lambda_peak
        self.mu_m = mu_m
        self.sigma_m = sigma_m
        self.delta_m = delta_m

        self._m1_grid = np.linspace(m_min, m_max, _GRID_SIZE)
        self._m1_cdf = self._build_m1_cdf()

    def _m1_pdf(self, m1: np.ndarray) -> np.ndarray:
        """Evaluate the (unnormalised) primary-mass PDF."""
        power_law = m1 ** (-self.alpha)
        gaussian = np.exp(-0.5 * ((m1 - self.mu_m) / self.sigma_m) ** 2)
        pdf: np.ndarray = (
            (1.0 - self.lambda_peak) * power_law + self.lambda_peak * gaussian
        ) * _smoothing(m1, self.m_min, self.delta_m)
        return pdf

    def _build_m1_cdf(self) -> np.ndarray:
        """Build and normalise the CDF of p(m1) on the precomputed grid."""
        pdf = self._m1_pdf(self._m1_grid)
        dm = self._m1_grid[1] - self._m1_grid[0]
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dm)])
        cdf /= cdf[-1]
        return cdf

    def _sample_m1(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=n)
        return np.interp(u, self._m1_cdf, self._m1_grid)

    def _sample_q(self, m1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample mass ratio q = m2/m1 for each m1, vectorised over events."""
        n = len(m1)
        q_unit = np.linspace(0.0, 1.0, _GRID_SIZE)  # (G,)
        q_lo = self.m_min / m1  # (n,)

        # q values remapped to [q_lo, 1] for each event: shape (n, G)
        q = q_lo[:, None] + (1.0 - q_lo[:, None]) * q_unit[None, :]
        m2 = m1[:, None] * q  # (n, G)

        pdf = q**self.beta_q * _smoothing(m2, self.m_min, self.delta_m)

        dq = q[:, 1:] - q[:, :-1]  # (n, G-1)
        cdf = np.concatenate(
            [
                np.zeros((n, 1)),
                np.cumsum(0.5 * (pdf[:, :-1] + pdf[:, 1:]) * dq, axis=1),
            ],
            axis=1,
        )
        norm = cdf[:, -1:]
        cdf /= np.where(norm > 0, norm, 1.0)

        u = rng.uniform(0.0, 1.0, size=n)
        q_sampled = np.empty(n)
        for i in range(n):
            q_sampled[i] = float(np.interp(u[i], cdf[i], q[i]))
        return q_sampled

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Draw source-frame masses from the power-law-plus-peak distribution.

        Parameters
        ----------
        n : int
            Number of events.
        rng : numpy.random.Generator or None, optional
            Random number generator.

        Returns
        -------
        m1_source, m2_source : numpy.ndarray
            Primary and secondary masses in :math:`M_\odot`, shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        m1 = self._sample_m1(n, rng)
        q = self._sample_q(m1, rng)
        m2 = m1 * q
        return m1, m2


# ---------------------------------------------------------------------------
# Concrete spin distributions
# ---------------------------------------------------------------------------


class IsotropicSpinDistribution(SpinDistribution):
    """Isotropic spin distribution with uniform magnitudes.

    Spin magnitudes are drawn uniformly in ``[0, a_max]``, tilt cosines
    uniformly in ``[-1, 1]`` (isotropic orientations), and azimuthal angles
    uniformly in ``[0, 2π)``.  This is the simplest physically motivated
    prior and a common reference model.

    Parameters
    ----------
    a_max : float, optional
        Maximum spin magnitude.  Defaults to ``1.0``.

    Raises
    ------
    ValueError
        If ``a_max <= 0``.
    """

    def __init__(self, a_max: float = 1.0) -> None:
        if a_max <= 0.0:
            raise ValueError(f"a_max must be positive, got {a_max!r}")
        self.a_max = a_max

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Draw spin parameters from the isotropic distribution.

        Parameters
        ----------
        n : int
            Number of events.
        rng : numpy.random.Generator or None, optional
            Random number generator.

        Returns
        -------
        a1, a2 : numpy.ndarray
            Spin magnitudes uniform in ``[0, a_max]``, shape ``(n,)``.
        cos_tilt1, cos_tilt2 : numpy.ndarray
            Cosines of spin tilts uniform in ``[-1, 1]``, shape ``(n,)``.
        phi12, phi_jl : numpy.ndarray
            Azimuthal angles uniform in ``[0, 2π)``, shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        a1 = rng.uniform(0.0, self.a_max, size=n)
        a2 = rng.uniform(0.0, self.a_max, size=n)
        cos_tilt1 = rng.uniform(-1.0, 1.0, size=n)
        cos_tilt2 = rng.uniform(-1.0, 1.0, size=n)
        phi12 = rng.uniform(0.0, 2.0 * np.pi, size=n)
        phi_jl = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return a1, a2, cos_tilt1, cos_tilt2, phi12, phi_jl


class DefaultSpinDistribution(SpinDistribution):
    r"""LVK default spin distribution (Abbott et al. 2021).

    Spin magnitudes follow a Beta distribution; spin tilt cosines follow a
    mixture of an isotropic component and a preferentially aligned component:

    .. math::

        p(a) &= \mathrm{Beta}(a;\,\alpha_\chi,\, \beta_\chi) \\[4pt]
        p(\cos\theta) &= \xi_{\rm spin}\,
            \mathcal{N}_T(\cos\theta;\,1,\,\sigma_{\rm spin},\,[-1,1])
            + (1 - \xi_{\rm spin})\,\tfrac{1}{2}

    where :math:`\mathcal{N}_T` is a truncated normal distribution.
    Azimuthal angles are drawn uniformly in :math:`[0, 2\pi)`.

    The hyperparameter names match the gwpopulation ``Default`` spin model.

    Parameters
    ----------
    alpha_chi : float
        First shape parameter of the Beta distribution for spin magnitudes.
        Must be positive.
    beta_chi : float
        Second shape parameter of the Beta distribution.  Must be positive.
    xi_spin : float
        Mixing fraction of the preferentially aligned component.
        Must be in ``[0, 1]``.
    sigma_spin : float
        Standard deviation of the aligned component's truncated normal.

    Raises
    ------
    ValueError
        If ``alpha_chi <= 0``, ``beta_chi <= 0``, or ``xi_spin`` is outside
        ``[0, 1]``.
    """

    def __init__(
        self,
        alpha_chi: float,
        beta_chi: float,
        xi_spin: float,
        sigma_spin: float,
    ) -> None:
        if alpha_chi <= 0.0:
            raise ValueError(f"alpha_chi must be positive, got {alpha_chi!r}")
        if beta_chi <= 0.0:
            raise ValueError(f"beta_chi must be positive, got {beta_chi!r}")
        if not (0.0 <= xi_spin <= 1.0):
            raise ValueError(f"xi_spin must be in [0, 1], got {xi_spin!r}")
        self.alpha_chi = alpha_chi
        self.beta_chi = beta_chi
        self.xi_spin = xi_spin
        self.sigma_spin = sigma_spin

    def _sample_cos_tilt(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample cosine tilt from the aligned + isotropic mixture."""
        use_aligned = rng.uniform(0.0, 1.0, size=n) < self.xi_spin
        cos_tilt = rng.uniform(-1.0, 1.0, size=n)
        n_aligned = int(use_aligned.sum())
        if n_aligned > 0:
            a_tn = (-1.0 - 1.0) / self.sigma_spin
            b_tn = (1.0 - 1.0) / self.sigma_spin
            seed = int(rng.integers(0, 2**31))
            aligned = truncnorm.rvs(
                a_tn,
                b_tn,
                loc=1.0,
                scale=self.sigma_spin,
                size=n_aligned,
                random_state=seed,
            )
            cos_tilt[use_aligned] = aligned
        return cos_tilt

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Draw spin parameters from the LVK default distribution.

        Parameters
        ----------
        n : int
            Number of events.
        rng : numpy.random.Generator or None, optional
            Random number generator.

        Returns
        -------
        a1, a2 : numpy.ndarray
            Spin magnitudes drawn from a Beta distribution, shape ``(n,)``.
        cos_tilt1, cos_tilt2 : numpy.ndarray
            Cosines of spin tilts from the aligned + isotropic mixture,
            shape ``(n,)``.
        phi12, phi_jl : numpy.ndarray
            Azimuthal angles uniform in ``[0, 2π)``, shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        seed1 = int(rng.integers(0, 2**31))
        seed2 = int(rng.integers(0, 2**31))
        a1 = beta_dist.rvs(self.alpha_chi, self.beta_chi, size=n, random_state=seed1)
        a2 = beta_dist.rvs(self.alpha_chi, self.beta_chi, size=n, random_state=seed2)
        cos_tilt1 = self._sample_cos_tilt(n, rng)
        cos_tilt2 = self._sample_cos_tilt(n, rng)
        phi12 = rng.uniform(0.0, 2.0 * np.pi, size=n)
        phi_jl = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return a1, a2, cos_tilt1, cos_tilt2, phi12, phi_jl


# ---------------------------------------------------------------------------
# Merger rate
# ---------------------------------------------------------------------------


class MergerRate(ABC):
    r"""Abstract base class for BBH merger-rate density models.

    A merger-rate density model gives the source-frame comoving merger-rate
    density :math:`R(z)` in :math:`\mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}`
    (mergers per unit comoving volume per unit *source-frame* time). This is
    the physical quantity that determines how many BBH events actually occur
    within a survey volume and observation window, playing the same role
    for the injection pipeline that
    :meth:`~bagpuss.universe.LuminosityModel.number_density` plays for
    :func:`~bagpuss.catalogue.build_catalogue`.

    Subclasses must implement :meth:`rate_density`.
    """

    @abstractmethod
    def rate_density(self, z: np.ndarray) -> np.ndarray:
        r"""Return the source-frame comoving merger-rate density :math:`R(z)`.

        Parameters
        ----------
        z : numpy.ndarray
            Redshifts at which to evaluate the rate density, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            Merger-rate density in :math:`\mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}`
            at each redshift, shape ``(n,)``.
        """


class ConstantMergerRate(MergerRate):
    r"""A merger-rate density that does not evolve with redshift.

    :math:`R(z) = R_0` for all :math:`z`, i.e. mergers are uniformly
    distributed in comoving volume and source-frame time. This mirrors the
    non-evolving galaxy number density assumed by
    :class:`~bagpuss.universe.PointProcess`; a redshift-evolving rate (e.g.
    tracking the Madau-Dickinson star-formation history) can be added later
    as another :class:`MergerRate` subclass without changing any call site.

    Parameters
    ----------
    rate_density_value : float
        The (constant) merger-rate density in
        :math:`\mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}`. Must be positive.

    Raises
    ------
    ValueError
        If ``rate_density_value <= 0``.
    """

    def __init__(self, rate_density_value: float) -> None:
        if rate_density_value <= 0.0:
            raise ValueError(
                f"rate_density_value must be positive, got {rate_density_value!r}"
            )
        self.rate_density_value = rate_density_value

    def rate_density(self, z: np.ndarray) -> np.ndarray:
        """Return the constant rate density, broadcast to the shape of ``z``.

        Parameters
        ----------
        z : numpy.ndarray
            Redshifts at which to evaluate the rate density, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            ``rate_density_value`` repeated for each redshift, shape ``(n,)``.
        """
        z = np.asarray(z, dtype=float)
        return np.full_like(z, self.rate_density_value)


def _rate_weight_grid(
    structure: Structure,
    cosmology: FLRW,
    rate: MergerRate,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the ``(z, weight)`` grid used to integrate/sample merger events.

    ``weight(z) = R(z) / (1 + z) * dV_C/dz`` -- the source-frame rate density
    converted to an observer-frame rate per unit redshift. The :math:`1/(1+z)`
    factor accounts for cosmological time dilation: a fixed interval of
    source-frame time corresponds to a longer interval in the detector frame,
    so the *observed* merger rate from a shell at redshift ``z`` is reduced
    relative to the source-frame rate by exactly this factor.
    """
    z_grid = np.linspace(0.0, structure.z_max, grid_size)
    dvc_dz = structure.differential_comoving_volume(z_grid, cosmology)
    weight = rate.rate_density(z_grid) / (1.0 + z_grid) * dvc_dz
    return z_grid, weight


def expected_n_mergers(
    structure: Structure,
    cosmology: FLRW,
    rate: MergerRate,
    t_start: float,
    t_end: float,
    grid_size: int = _RATE_GRID_SIZE,
) -> float:
    r"""Return the expected number of mergers observed in ``[t_start, t_end]``.

    Computes

    .. math::

        N = T_{\rm obs} \int_0^{z_{\max}} \frac{R(z)}{1+z}
            \frac{dV_C}{dz} \, dz

    where :math:`T_{\rm obs}` is the observation window converted to years
    and the :math:`1/(1+z)` factor accounts for cosmological time dilation
    between source-frame merger rate and detector-frame observed rate (see
    :func:`_rate_weight_grid`). Combined with a Poisson draw, this gives a
    physically motivated event count analogous to how
    :func:`~bagpuss.catalogue.build_catalogue` Poisson-realises the galaxy
    count from ``number_density() * survey_volume()``.

    Parameters
    ----------
    structure : bagpuss.universe.Structure
        Structure model providing ``z_max`` and
        :meth:`~bagpuss.universe.Structure.differential_comoving_volume`.
    cosmology : astropy.cosmology.FLRW
        Background cosmology.
    rate : MergerRate
        Merger-rate density model.
    t_start : float
        Start of the observation window in GPS seconds.
    t_end : float
        End of the observation window in GPS seconds.
    grid_size : int, optional
        Number of points used to numerically integrate over ``[0, z_max]``.

    Returns
    -------
    float
        Expected number of observed mergers (before any detectability cut).
    """
    z_grid, weight = _rate_weight_grid(structure, cosmology, rate, grid_size)
    rate_per_year = float(np.trapezoid(weight, z_grid))
    t_obs_years = (t_end - t_start) / SECONDS_PER_YEAR
    return rate_per_year * t_obs_years


def sample_merger_redshifts(
    structure: Structure,
    cosmology: FLRW,
    rate: MergerRate,
    n: int,
    rng: np.random.Generator | None = None,
    grid_size: int = _RATE_GRID_SIZE,
) -> np.ndarray:
    r"""Draw merger redshifts weighted by rate density and time dilation.

    Samples from the distribution :math:`p(z) \propto R(z)/(1+z) \, dV_C/dz`
    (see :func:`_rate_weight_grid`) via inverse-CDF on a precomputed grid,
    the same technique :meth:`~bagpuss.universe.PointProcess.sample_redshifts`
    uses for the unweighted (galaxy) case.

    Parameters
    ----------
    structure : bagpuss.universe.Structure
        Structure model providing ``z_max`` and
        :meth:`~bagpuss.universe.Structure.differential_comoving_volume`.
    cosmology : astropy.cosmology.FLRW
        Background cosmology.
    rate : MergerRate
        Merger-rate density model.
    n : int
        Number of redshifts to sample.
    rng : numpy.random.Generator or None, optional
        Random number generator. If *None*, ``numpy.random.default_rng()``
        is used.
    grid_size : int, optional
        Number of points used to build the inverse-CDF grid.

    Returns
    -------
    numpy.ndarray
        Merger redshifts, shape ``(n,)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    z_grid, weight = _rate_weight_grid(structure, cosmology, rate, grid_size)
    dz = np.diff(z_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[:-1] + weight[1:]) * dz)])
    cdf /= cdf[-1]

    u = rng.uniform(0.0, 1.0, size=n)
    return np.asarray(np.interp(u, cdf, z_grid))
