r"""Sensitivity injections -- the selection function :math:`P(D|\Lambda)`.

A hierarchical (dark-siren or population) analysis needs the probability that
an event drawn from a population :math:`\Lambda` is detected,

.. math::

    P(D|\Lambda) = \int P(D|\theta)\, p(\theta|\Lambda)\, d\theta .

This is *not* the same thing as the event set built in
:mod:`bagpuss.injection`: that is one realisation of one true population, with
the undetected events thrown away. :math:`P(D|\Lambda)` is needed at every
:math:`\Lambda` the sampler visits, so the injections are instead drawn from a
fixed, deliberately broad distribution :math:`p(\theta|\Lambda_{\rm inj})`
which covers the support of every target population, and the integral is
estimated by importance sampling (Essick et al. 2025, arXiv:2508.10638;
Farr 2019, arXiv:1904.10879):

.. math::

    \hat P(D|\Lambda) = \frac{1}{N_{\rm total}}
        \sum_{i \in {\rm found}}
        \frac{p(\theta_i|\Lambda)}{p(\theta_i|\Lambda_{\rm inj})} .

Three details matter, and each is easy to get wrong:

* :math:`N_{\rm total}` counts **every** injection drawn, including those
  discarded before the (expensive) detection step because they are hopeless;
* the draw density :math:`p(\theta_i|\Lambda_{\rm inj})` is stored with each
  injection, evaluated by the same code that did the drawing; and
* the Monte Carlo uncertainty is reported, through
  :math:`N_{\rm eff}=\mu^2/\sigma^2`, which should exceed about four times
  the number of detected events in the analysis.

All densities are with respect to the stored variables (see
:data:`DENSITY_MEASURE`), so a target population must be expressed in the same
measure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, cast, overload

import h5py
import numpy as np
import zarr
from astropy.cosmology import FLRW

__all__: list[str] = [
    "DENSITY_MEASURE",
    "DetectionEstimate",
    "InjectionDistribution",
    "SensitivityInjectionSet",
    "draw_injections",
]

#: The measure :meth:`InjectionDistribution.log_prob` is a density with respect to.
DENSITY_MEASURE: str = (
    "dm1_source dm2_source da1 da2 dcos_tilt1 dcos_tilt2 dphi12 dphi_jl "
    "dtheta_jn dra ddec dpsi dgeocent_time dredshift"
)

_PARAMETER_FIELDS: tuple[str, ...] = (
    "m1_source",
    "m2_source",
    "a1",
    "a2",
    "cos_tilt1",
    "cos_tilt2",
    "phi12",
    "phi_jl",
    "theta_jn",
    "ra",
    "dec",
    "psi",
    "geocent_time",
    "redshift",
    "luminosity_distance",
)

_REDSHIFT_GRID_SIZE = 8192
_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Truncated power laws
# ---------------------------------------------------------------------------


def _pl_integral(s: float, lo: np.ndarray | float, hi: float) -> np.ndarray:
    r"""Return :math:`\int_{lo}^{hi} x^s dx` (elementwise in *lo*)."""
    lo = np.asarray(lo, dtype=float)
    if s == -1.0:
        return np.asarray(np.log(hi / lo))
    return np.asarray((hi ** (s + 1.0) - lo ** (s + 1.0)) / (s + 1.0))


def _pl_sample(
    s: float, lo: np.ndarray | float, hi: float, u: np.ndarray
) -> np.ndarray:
    """Inverse-CDF sample of a power law ``x**s`` truncated to ``[lo, hi]``."""
    lo = np.asarray(lo, dtype=float)
    if s == -1.0:
        return np.asarray(lo * (hi / lo) ** u)
    p = s + 1.0
    return np.asarray((lo**p + u * (hi**p - lo**p)) ** (1.0 / p))


# ---------------------------------------------------------------------------
# The injection distribution
# ---------------------------------------------------------------------------


class InjectionDistribution:
    r"""A broad, fixed distribution to draw sensitivity injections from.

    It is deliberately *not* the true population of a simulation. It should
    cover the support of every population the selection function will be
    evaluated for (otherwise the estimate is silently truncated), with heavy
    tails so that outliers are sampled.

    The distribution is a product of independent marginals, apart from the
    mass ratio's dependence on the primary mass:

    * primary mass :math:`p(m_1) \propto m_1^{-\alpha}` on
      :math:`[m_{\min}, m_{\max}]`; mass ratio
      :math:`p(q|m_1) \propto q^{\beta_q}` on :math:`[m_{\min}/m_1, 1]`
      (so both components lie in the range);
    * spin magnitudes uniform on :math:`[0, a_{\max}]`, spin tilts and
      azimuthal angles isotropic;
    * redshift :math:`p(z) \propto (1+z)^{\kappa-1}\, dV_c/dz` out to
      ``z_max`` (``kappa = 0`` is a constant source-frame merger rate
      density; larger values weight high redshift more heavily);
    * sky position and inclination isotropic, polarisation angle uniform, and
      merger time uniform on ``[t_start, t_end]``.

    Parameters
    ----------
    cosmology : astropy.cosmology.FLRW
        Fiducial cosmology, used for the volume element and to convert the
        redshift of each injection to a luminosity distance.
    t_start, t_end : float
        Observation window in GPS seconds.
    z_max : float, optional
        Maximum redshift.
    m_min, m_max : float, optional
        Source-frame component-mass range in solar masses.
    alpha : float, optional
        Primary-mass power-law index (``p(m1) ~ m1**-alpha``).
    beta_q : float, optional
        Mass-ratio power-law index.
    a_max : float, optional
        Maximum spin magnitude.
    kappa : float, optional
        Redshift-evolution index.

    Raises
    ------
    ValueError
        If a range is empty or non-positive.
    """

    def __init__(
        self,
        cosmology: FLRW,
        t_start: float,
        t_end: float,
        z_max: float = 3.0,
        m_min: float = 3.0,
        m_max: float = 200.0,
        alpha: float = 2.0,
        beta_q: float = 0.0,
        a_max: float = 1.0,
        kappa: float = 0.0,
    ) -> None:
        if z_max <= 0.0:
            raise ValueError(f"z_max must be positive, got {z_max!r}")
        if not 0.0 < m_min < m_max:
            raise ValueError(
                f"need 0 < m_min < m_max, got m_min={m_min!r}, m_max={m_max!r}"
            )
        if not t_end > t_start:
            raise ValueError(
                f"t_end must exceed t_start, got t_start={t_start!r}, t_end={t_end!r}"
            )
        if a_max <= 0.0:
            raise ValueError(f"a_max must be positive, got {a_max!r}")
        self.cosmology = cosmology
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.z_max = float(z_max)
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.alpha = float(alpha)
        self.beta_q = float(beta_q)
        self.a_max = float(a_max)
        self.kappa = float(kappa)

        self._z_grid = np.linspace(0.0, self.z_max, _REDSHIFT_GRID_SIZE)
        weight = (1.0 + self._z_grid) ** (self.kappa - 1.0) * np.asarray(
            cosmology.differential_comoving_volume(self._z_grid).value
        )
        cdf = np.concatenate(
            [[0.0], np.cumsum(0.5 * (weight[:-1] + weight[1:]) * np.diff(self._z_grid))]
        )
        self._z_pdf = weight / cdf[-1]
        self._z_cdf = cdf / cdf[-1]

    # -- redshift ---------------------------------------------------------

    def redshift_pdf(self, z: np.ndarray) -> np.ndarray:
        """Return the normalised redshift density (zero outside ``(0, z_max]``)."""
        z = np.asarray(z, dtype=float)
        pdf = np.interp(z, self._z_grid, self._z_pdf)
        return np.where((z > 0.0) & (z <= self.z_max), pdf, 0.0)

    # -- sampling ---------------------------------------------------------

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw *n* injections; returns the stored parameters (no densities)."""
        m1 = _pl_sample(-self.alpha, self.m_min, self.m_max, rng.uniform(size=n))
        q = _pl_sample(self.beta_q, self.m_min / m1, 1.0, rng.uniform(size=n))
        redshift = np.interp(rng.uniform(size=n), self._z_cdf, self._z_grid)
        luminosity_distance = np.asarray(
            self.cosmology.luminosity_distance(redshift).to("Mpc").value
        )
        return {
            "m1_source": m1,
            "m2_source": m1 * q,
            "a1": rng.uniform(0.0, self.a_max, size=n),
            "a2": rng.uniform(0.0, self.a_max, size=n),
            "cos_tilt1": rng.uniform(-1.0, 1.0, size=n),
            "cos_tilt2": rng.uniform(-1.0, 1.0, size=n),
            "phi12": rng.uniform(0.0, _TWO_PI, size=n),
            "phi_jl": rng.uniform(0.0, _TWO_PI, size=n),
            "theta_jn": np.arccos(rng.uniform(-1.0, 1.0, size=n)),
            "ra": rng.uniform(0.0, _TWO_PI, size=n),
            "dec": np.arcsin(rng.uniform(-1.0, 1.0, size=n)),
            "psi": rng.uniform(0.0, np.pi, size=n),
            "geocent_time": rng.uniform(self.t_start, self.t_end, size=n),
            "redshift": redshift,
            "luminosity_distance": luminosity_distance,
        }

    # -- density ----------------------------------------------------------

    def nuisance_log_prob(self, params: dict[str, np.ndarray]) -> np.ndarray:
        r"""Return the log density of everything except masses and redshift.

        The spin magnitudes, tilts and azimuths, inclination, sky position,
        polarisation and merger time are independent of the masses and the
        redshift, so :meth:`log_prob` is this plus the mass and redshift terms.
        Subtracting it from a stored ``log_p_draw`` leaves the density of
        ``(m1_source, m2_source, redshift)`` alone, which is what a population
        analysis which does not fit the other parameters needs.

        Parameters
        ----------
        params : dict
            Arrays keyed by field name; needs ``theta_jn`` and ``dec``.

        Returns
        -------
        numpy.ndarray
            The log density (the support is not checked).
        """
        theta_jn = np.asarray(params["theta_jn"], dtype=float)
        dec = np.asarray(params["dec"], dtype=float)
        log_p = -2.0 * np.log(self.a_max)  # two uniform spin magnitudes
        log_p = log_p - 2.0 * np.log(2.0)  # two uniform cos(tilt) on [-1, 1]
        log_p = log_p - 2.0 * np.log(_TWO_PI)  # phi12, phi_jl
        log_p = log_p + np.log(np.abs(np.sin(theta_jn)) / 2.0 + 1e-300)
        log_p = log_p + np.log(np.abs(np.cos(dec)) / (2.0 * _TWO_PI) + 1e-300)
        log_p = log_p - np.log(np.pi)  # psi
        log_p = log_p - np.log(self.t_end - self.t_start)
        return np.asarray(log_p)

    def log_prob(
        self,
        params: dict[str, np.ndarray],
        mass_log_prob: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return the log density of *params* in the measure :data:`DENSITY_MEASURE`.

        Parameters
        ----------
        params : dict
            Arrays keyed by field name (at least those in the measure).
        mass_log_prob : callable or None, optional
            If given, replaces this distribution's mass density: a function of
            ``(m1_source, m2_source)`` returning the log density with respect to
            ``dm1_source dm2_source``. Everything else is unchanged, so this
            turns the distribution into a population whose masses follow another
            model (e.g. ``PowerLawPlusPeakMassDistribution.log_prob``).

        Returns
        -------
        numpy.ndarray
            ``log p``; ``-inf`` outside the support.
        """
        m1 = np.asarray(params["m1_source"], dtype=float)
        m2 = np.asarray(params["m2_source"], dtype=float)
        z = np.asarray(params["redshift"], dtype=float)
        a1 = np.asarray(params["a1"], dtype=float)
        a2 = np.asarray(params["a2"], dtype=float)
        t = np.asarray(params["geocent_time"], dtype=float)

        mass_ok = (
            (m1 >= self.m_min) & (m1 <= self.m_max) & (m2 >= self.m_min) & (m2 <= m1)
        )
        other_ok = (
            (z > 0.0)
            & (z <= self.z_max)
            & (a1 >= 0.0)
            & (a1 <= self.a_max)
            & (a2 >= 0.0)
            & (a2 <= self.a_max)
            & (t >= self.t_start)
            & (t <= self.t_end)
        )
        # A replacement mass density defines its own support.
        inside = other_ok & (mass_ok if mass_log_prob is None else True)
        if mass_log_prob is None:
            # Out-of-support entries are evaluated at a safe interior point so
            # they don't raise warnings; they are overwritten with -inf below.
            m1s = np.where(inside, m1, np.sqrt(self.m_min * self.m_max))
            qs = np.where(inside, m2 / m1s, 1.0)
            log_m1 = -self.alpha * np.log(m1s) - np.log(
                _pl_integral(-self.alpha, self.m_min, self.m_max)
            )
            log_q = self.beta_q * np.log(qs) - np.log(
                _pl_integral(self.beta_q, self.m_min / m1s, 1.0)
            )
            # p(m1, m2) = p(m1) p(q|m1) / m1
            log_p = log_m1 + log_q - np.log(m1s)
        else:
            log_p = np.asarray(mass_log_prob(m1, m2), dtype=float)
        log_p = log_p + self.nuisance_log_prob(params)
        with np.errstate(divide="ignore"):
            log_p = log_p + np.log(self.redshift_pdf(np.where(inside, z, 1.0)))
        return np.where(inside, log_p, -np.inf)


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionEstimate:
    r"""A Monte Carlo estimate of :math:`P(D|\Lambda)` and its uncertainty.

    Attributes
    ----------
    value : float
        The estimate :math:`\hat P(D|\Lambda)`.
    std : float
        Its Monte Carlo standard deviation.
    n_eff : float
        :math:`\mu^2/\sigma^2` (Farr 2019): the effective number of injections.
    n_found : int
        Number of detected injections contributing.
    n_total : int
        Total number of injections drawn, including any cut as hopeless.
    """

    value: float
    std: float
    n_eff: float
    n_found: int
    n_total: int

    def is_reliable(self, n_events: int, factor: float = 4.0) -> bool:
        """Return whether ``n_eff >= factor * n_events``.

        The default ``factor`` of 4 is the criterion of Essick et al. (2025)
        for a catalogue of ``n_events`` detections.
        """
        return self.n_eff >= factor * n_events


# ---------------------------------------------------------------------------
# The injection set
# ---------------------------------------------------------------------------


@dataclass
class SensitivityInjectionSet:
    r"""Injections drawn from an :class:`InjectionDistribution`.

    Unlike :class:`bagpuss.injection.InjectionSet` there are no host galaxies
    (the sensitivity must not depend on the catalogue), and the draw density
    and total number of draws are kept so :math:`P(D|\Lambda)` can be
    estimated for any target population.

    Parameters
    ----------
    m1_source, m2_source, a1, a2, cos_tilt1, cos_tilt2, phi12, phi_jl, theta_jn,
    ra, dec, psi, geocent_time, redshift, luminosity_distance : numpy.ndarray
        The injection parameters, as for :class:`~bagpuss.injection.InjectionSet`;
        shape ``(n,)``.
    log_p_draw : numpy.ndarray
        Log of the density each injection was drawn from, in the measure
        :data:`DENSITY_MEASURE`, shape ``(n,)``.
    n_total : int
        The number of injections drawn in all, which may exceed ``n`` where
        some were cut before detection (see :meth:`select`).
    """

    m1_source: np.ndarray
    m2_source: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    cos_tilt1: np.ndarray
    cos_tilt2: np.ndarray
    phi12: np.ndarray
    phi_jl: np.ndarray
    theta_jn: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    psi: np.ndarray
    geocent_time: np.ndarray
    redshift: np.ndarray
    luminosity_distance: np.ndarray
    log_p_draw: np.ndarray
    n_total: int

    def __len__(self) -> int:
        """Return the number of injections held."""
        return int(self.m1_source.shape[0])

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the parameters as a dict of arrays (without ``log_p_draw``)."""
        return {name: getattr(self, name) for name in _PARAMETER_FIELDS}

    def select(self, keep: np.ndarray) -> SensitivityInjectionSet:
        """Return the injections where *keep* is true, **preserving** ``n_total``.

        Use this to discard hopeless injections before the detection step:
        they are no longer held, but still count among the draws.
        """
        keep = np.asarray(keep, dtype=bool)
        data = {f.name: getattr(self, f.name)[keep] for f in _array_fields()}
        return SensitivityInjectionSet(n_total=self.n_total, **data)

    @classmethod
    def concatenate(
        cls, parts: list[SensitivityInjectionSet]
    ) -> SensitivityInjectionSet:
        """Join sets (e.g. shards), summing their ``n_total``."""
        if not parts:
            raise ValueError("cannot concatenate an empty list of injection sets")
        data = {
            f.name: np.concatenate([getattr(p, f.name) for p in parts])
            for f in _array_fields()
        }
        return cls(n_total=sum(p.n_total for p in parts), **data)

    def detection_probability(
        self,
        log_p_target: Callable[[dict[str, np.ndarray]], np.ndarray],
        snr: np.ndarray,
        snr_threshold: float,
    ) -> DetectionEstimate:
        r"""Estimate :math:`P(D|\Lambda)` for a target population.

        Parameters
        ----------
        log_p_target : callable
            Maps a dict of parameter arrays to the log density of the target
            population :math:`p(\theta|\Lambda)`, in the measure
            :data:`DENSITY_MEASURE`. It is evaluated on the detected
            injections only.
        snr : numpy.ndarray
            Network SNR of each held injection, shape ``(len(self),)``.
        snr_threshold : float
            An injection is detected if ``snr >= snr_threshold``.

        Returns
        -------
        DetectionEstimate

        Raises
        ------
        ValueError
            If *snr* does not have one entry per injection.
        """
        snr = np.asarray(snr, dtype=float)
        if snr.shape != (len(self),):
            raise ValueError(f"snr must have shape ({len(self)},), got {snr.shape}")
        found = snr >= snr_threshold
        n_found = int(found.sum())
        if n_found == 0:
            return DetectionEstimate(0.0, 0.0, 0.0, 0, self.n_total)

        params = {name: arr[found] for name, arr in self.as_dict().items()}
        weights = np.exp(log_p_target(params) - self.log_p_draw[found])
        n = self.n_total
        mean = float(weights.sum() / n)
        second = float((weights**2).sum() / n)
        variance = max(second - mean**2, 0.0) / n
        n_eff = mean**2 / variance if variance > 0.0 else float("inf")
        return DetectionEstimate(mean, float(np.sqrt(variance)), n_eff, n_found, n)

    # -- I/O --------------------------------------------------------------

    def to_hdf5(self, path: str | Path, network_snr: np.ndarray | None = None) -> None:
        """Write the set to HDF5, optionally with the network SNRs.

        Each field is a dataset in ``/injections``; ``n_total`` and the
        density measure are attributes of the group.
        """
        with h5py.File(path, "w") as fh:
            grp = fh.create_group("injections")
            for f in _array_fields():
                grp.create_dataset(f.name, data=getattr(self, f.name))
            if network_snr is not None:
                grp.create_dataset("network_snr", data=np.asarray(network_snr))
            grp.attrs["n_total"] = self.n_total
            grp.attrs["density_measure"] = DENSITY_MEASURE

    @overload
    @classmethod
    def from_hdf5(
        cls, path: str | Path, with_snr: Literal[False] = ...
    ) -> SensitivityInjectionSet: ...

    @overload
    @classmethod
    def from_hdf5(
        cls, path: str | Path, with_snr: Literal[True]
    ) -> tuple[SensitivityInjectionSet, np.ndarray]: ...

    @classmethod
    def from_hdf5(
        cls, path: str | Path, with_snr: bool = False
    ) -> SensitivityInjectionSet | tuple[SensitivityInjectionSet, np.ndarray]:
        """Load a set written by :meth:`to_hdf5` (and the SNRs, if asked)."""
        with h5py.File(path, "r") as fh:
            grp = cast(h5py.Group, fh["injections"])
            data = {
                f.name: cast(h5py.Dataset, grp[f.name])[()] for f in _array_fields()
            }
            n_total = int(cast(int, grp.attrs["n_total"]))
            snr = np.asarray(grp["network_snr"]) if with_snr else None
        loaded = cls(n_total=n_total, **data)
        if snr is None:
            return loaded
        return loaded, snr

    def to_zarr(
        self, group: zarr.Group | str | Path, network_snr: np.ndarray | None = None
    ) -> None:
        """Write the set as arrays in a zarr group (one group per shard)."""
        target = (
            group
            if isinstance(group, zarr.Group)
            else zarr.open_group(store=str(group), mode="a")
        )
        for f in _array_fields():
            target.create_array(f.name, data=getattr(self, f.name), overwrite=True)
        if network_snr is not None:
            target.create_array(
                "network_snr", data=np.asarray(network_snr), overwrite=True
            )
        target.attrs["n_total"] = self.n_total

    @classmethod
    def from_zarr(cls, group: zarr.Group | str | Path) -> SensitivityInjectionSet:
        """Load a set written by :meth:`to_zarr`."""
        source = (
            group
            if isinstance(group, zarr.Group)
            else zarr.open_group(store=str(group), mode="r")
        )
        data = {f.name: np.asarray(source[f.name]) for f in _array_fields()}
        return cls(n_total=int(cast(int, source.attrs["n_total"])), **data)


def _array_fields() -> tuple:  # type: ignore[type-arg]
    """Return the dataclass fields which hold per-injection arrays."""
    return tuple(f for f in fields(SensitivityInjectionSet) if f.name != "n_total")


def draw_injections(
    distribution: InjectionDistribution, n: int, rng: np.random.Generator | None = None
) -> SensitivityInjectionSet:
    """Draw *n* injections from *distribution*, recording their draw density.

    Parameters
    ----------
    distribution : InjectionDistribution
        The distribution to draw from.
    n : int
        Number of injections.
    rng : numpy.random.Generator or None, optional
        Random number generator; ``numpy.random.default_rng()`` if *None*.

    Returns
    -------
    SensitivityInjectionSet
        With ``n_total == n``.
    """
    if rng is None:
        rng = np.random.default_rng()
    params = distribution.sample(n, rng)
    return SensitivityInjectionSet(
        log_p_draw=distribution.log_prob(params), n_total=n, **params
    )
