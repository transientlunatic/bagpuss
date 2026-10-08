"""Universe simulation — Stage 1 of the bagpuss pipeline.

Draws a cosmology from a cosmological model, a large-scale structure model,
and a luminosity model to define the simulated Universe from which all
downstream catalogues are generated.

The two primary extension points are :class:`Structure` and
:class:`LuminosityModel`.  Concrete implementations of each can be slotted
into :class:`Universe` to define different physics scenarios without changing
any other part of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from astropy.cosmology import FLRW

from bagpuss.galaxies import GalaxySet

__all__: list[str] = [
    "Universe",
    "Structure",
    "LuminosityModel",
    "PointProcess",
    "SkyPatch",
]


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class Structure(ABC):
    """Abstract base class for galaxy large-scale structure models.

    A structure model is responsible for drawing the *spatial* distribution
    of galaxies — their redshifts and sky positions — given a background
    cosmology.  Subclasses must implement :meth:`sample_redshifts`.
    :meth:`sample_positions` has a default isotropic implementation that
    subclasses may override for anisotropic models.
    """

    @abstractmethod
    def survey_volume(self, cosmology: FLRW) -> float:
        """Return the total comoving survey volume in Mpc³.

        Parameters
        ----------
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute comoving distances.

        Returns
        -------
        float
            Comoving volume of the survey in Mpc³.
        """

    @property
    @abstractmethod
    def z_max(self) -> float:
        """Return the maximum redshift of the simulated volume."""

    @abstractmethod
    def differential_comoving_volume(
        self, z: np.ndarray, cosmology: FLRW
    ) -> np.ndarray:
        r"""Return the comoving volume element :math:`dV_C/dz` in Mpc³.

        Already integrated over the model's sky coverage (the full 4π sr
        for an untiled model, or a tile's solid angle for
        :class:`SkyPatch`), so that integrating the returned values over
        ``z`` from 0 to :attr:`z_max` reproduces :meth:`survey_volume`.
        This is the quantity needed to convert a redshift-dependent rate
        density into an expected event count (see
        :func:`~bagpuss.population.expected_n_mergers`).

        Parameters
        ----------
        z : numpy.ndarray
            Redshifts at which to evaluate the volume element, shape ``(n,)``.
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute the comoving volume element.

        Returns
        -------
        numpy.ndarray
            :math:`dV_C/dz` in Mpc³ at each redshift, shape ``(n,)``.
        """

    @abstractmethod
    def sample_redshifts(
        self,
        n: int,
        cosmology: FLRW,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw redshifts for a set of galaxies.

        Parameters
        ----------
        n : int
            Number of galaxies to sample.
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute comoving volumes or
            distances as required by the model.
        rng : numpy.random.Generator or None, optional
            Random number generator.  If *None*, ``numpy.random.default_rng()``
            is used.

        Returns
        -------
        numpy.ndarray
            Array of redshifts with shape ``(n,)``.
        """

    def sample_positions(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw isotropic sky positions for a set of galaxies.

        The default implementation draws positions uniformly on the sphere,
        which is correct for any spatially homogeneous model.  Subclasses
        may override this for anisotropic large-scale structure models.

        Parameters
        ----------
        n : int
            Number of positions to sample.
        rng : numpy.random.Generator or None, optional
            Random number generator.  If *None*, ``numpy.random.default_rng()``
            is used.

        Returns
        -------
        ra : numpy.ndarray
            Right ascensions in radians, uniformly distributed in
            ``[0, 2π)``, shape ``(n,)``.
        dec : numpy.ndarray
            Declinations in radians, distributed as
            ``arcsin(Uniform(-1, 1))``, shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        ra = rng.uniform(0.0, 2.0 * np.pi, size=n)
        dec = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
        return ra, dec


class LuminosityModel(ABC):
    """Abstract base class for galaxy luminosity distribution models.

    A luminosity model is responsible for drawing galaxy luminosities
    independently of their spatial positions.  Subclasses must implement
    :meth:`sample`.
    """

    @abstractmethod
    def number_density(self) -> float:
        """Return the expected number density of galaxies in Mpc⁻³.

        This is the integral of the luminosity function over the model's
        magnitude range ``[m_min, m_max]``, giving the total number of
        galaxies per unit comoving volume.  Combined with
        :meth:`~bagpuss.universe.Structure.survey_volume`, it determines
        the expected total galaxy count for a Poisson realisation of the
        simulated Universe.

        Returns
        -------
        float
            Expected galaxy number density in Mpc⁻³.
        """

    @abstractmethod
    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw luminosities for a set of galaxies.

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

    @abstractmethod
    def cdf(self, magnitude: np.ndarray) -> np.ndarray:
        """Return the cumulative fraction of the population brighter than ``magnitude``.

        This is the quantity needed by a magnitude-limited
        :class:`~bagpuss.catalogue.SelectionFunction` to compute survey
        completeness: the fraction of galaxies with absolute magnitude at or
        brighter than a given threshold, without having to draw and reject
        individual samples.

        Parameters
        ----------
        magnitude : numpy.ndarray
            Absolute magnitudes at which to evaluate the cumulative
            distribution, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            Fraction of the population with absolute magnitude less than or
            equal to ``magnitude`` (i.e. brighter, since lower magnitudes are
            brighter), shape ``(n,)``.  Values are in ``[0, 1]``.
        """

    def weighted_cdf(self, magnitude: np.ndarray, weight_power: float) -> np.ndarray:
        r"""Return the :math:`L^{p}`-weighted analogue of :meth:`cdf`.

        Fraction of the :math:`L^{p}`-weighted population (``p`` is
        ``weight_power``) with absolute magnitude at or brighter than
        ``magnitude``. Models that do not support ``weight_power != 0`` need
        not override this.
        """
        if weight_power != 0.0:
            raise NotImplementedError(
                f"{type(self).__name__} does not support weighted completeness"
            )
        return self.cdf(magnitude)

    @abstractmethod
    def sample_brighter_than(
        self,
        n: int,
        magnitude_threshold: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw luminosities conditioned on being brighter than a threshold.

        Unlike :meth:`sample`, which draws unconditionally from the full
        magnitude range, this draws each galaxy's luminosity conditioned on
        its absolute magnitude being at or below (i.e. brighter than) a
        per-galaxy threshold. This lets a caller sample directly from the
        *observable* (selection-passing) population -- e.g. one galaxy per
        redshift, each thresholded by that redshift's own distance-modulus
        -shifted magnitude limit -- without ever drawing and discarding the
        unobservable majority.

        Parameters
        ----------
        n : int
            Number of luminosities to sample.
        magnitude_threshold : numpy.ndarray
            Per-galaxy absolute-magnitude threshold, shape ``(n,)``. Only
            galaxies with absolute magnitude at or below (brighter than)
            ``magnitude_threshold[i]`` are drawn for element ``i``.
        rng : numpy.random.Generator or None, optional
            Random number generator.  If *None*, ``numpy.random.default_rng()``
            is used.

        Returns
        -------
        numpy.ndarray
            Luminosities in solar luminosities, shape ``(n,)``.
        """


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


class Universe:
    """A simulated Universe from which galaxy catalogues can be drawn.

    :class:`Universe` wires together a background cosmology, a
    :class:`Structure` model (spatial distribution), and a
    :class:`LuminosityModel` (luminosity distribution) into a single object.
    Galaxies are sampled via :meth:`sample`.

    Parameters
    ----------
    cosmology : astropy.cosmology.FLRW
        Background cosmological model (e.g. ``astropy.cosmology.Planck18``).
    structure : Structure
        Large-scale structure model that determines how galaxy redshifts
        are distributed.
    luminosity : LuminosityModel
        Luminosity distribution model that assigns luminosities to galaxies.

    Examples
    --------
    >>> from astropy.cosmology import Planck18
    >>> universe = Universe(
    ...     cosmology=Planck18,
    ...     structure=PointProcess(z_max=1.0),
    ...     luminosity=my_luminosity_model,
    ... )
    >>> galaxies = universe.sample(1000)
    """

    def __init__(
        self,
        cosmology: FLRW,
        structure: Structure,
        luminosity: LuminosityModel,
    ) -> None:
        self.cosmology = cosmology
        self.structure = structure
        self.luminosity = luminosity

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> GalaxySet:
        """Sample a set of galaxies from the simulated Universe.

        Delegates to :attr:`structure` for redshifts and :attr:`luminosity`
        for luminosities, then bundles the results into a :class:`GalaxySet`.

        Parameters
        ----------
        n : int
            Number of galaxies to sample.
        rng : numpy.random.Generator or None, optional
            Random number generator passed through to both the structure and
            luminosity models.  If *None*, each model calls
            ``numpy.random.default_rng()`` internally.

        Returns
        -------
        bagpuss.galaxies.GalaxySet
            A galaxy set with ``n`` entries.
        """
        redshifts = self.structure.sample_redshifts(n, self.cosmology, rng)
        luminosities = self.luminosity.sample(n, rng)
        ra, dec = self.structure.sample_positions(n, rng)
        return GalaxySet(redshifts=redshifts, luminosities=luminosities, ra=ra, dec=dec)


# ---------------------------------------------------------------------------
# Concrete structure models
# ---------------------------------------------------------------------------


class PointProcess(Structure):
    r"""A Poisson point-process large-scale structure model.

    Assumes that galaxies are *uniformly distributed in comoving volume*
    between redshift 0 and ``z_max``.  Redshifts are sampled via an
    inverse-CDF method on a precomputed comoving-volume grid.

    Parameters
    ----------
    z_max : float
        Maximum redshift of the simulated volume.  Must be positive.

    Raises
    ------
    ValueError
        If ``z_max`` is not positive.

    Notes
    -----
    The comoving volume :math:`V_C(z)` is a monotonically increasing function
    of redshift.  Drawing :math:`u \\sim \\mathrm{Uniform}(0, V_C(z_{\\max}))`
    and inverting gives a sample that is uniform in comoving volume:

    .. math::

        z = V_C^{-1}(u)

    The inversion is performed numerically by linear interpolation on a grid
    of 10 000 points.
    """

    _GRID_SIZE: int = 10_000

    def __init__(self, z_max: float) -> None:
        if z_max <= 0:
            raise ValueError(f"z_max must be positive, got {z_max!r}")
        self._z_max = z_max

    @property
    def z_max(self) -> float:
        """Return the maximum redshift of the simulated volume."""
        return self._z_max

    def survey_volume(self, cosmology: FLRW) -> float:
        """Return the comoving volume out to ``z_max`` in Mpc³.

        Parameters
        ----------
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute the comoving volume.

        Returns
        -------
        float
            Comoving volume in Mpc³.
        """
        return float(
            cosmology.comoving_volume(self.z_max).value  # pyright: ignore[reportAttributeAccessIssue]
        )

    def differential_comoving_volume(
        self, z: np.ndarray, cosmology: FLRW
    ) -> np.ndarray:
        r"""Return :math:`dV_C/dz` in Mpc³, integrated over the full 4π sr sky.

        Parameters
        ----------
        z : numpy.ndarray
            Redshifts at which to evaluate the volume element, shape ``(n,)``.
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute the comoving volume element.

        Returns
        -------
        numpy.ndarray
            :math:`dV_C/dz` in Mpc³ at each redshift, shape ``(n,)``.
        """
        dvc_dz_dOmega = cosmology.differential_comoving_volume(  # pyright: ignore[reportAttributeAccessIssue]
            z
        ).to_value("Mpc3 / sr")
        return np.asarray(4.0 * np.pi * dvc_dz_dOmega)

    def sample_redshifts(
        self,
        n: int,
        cosmology: FLRW,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw redshifts uniformly distributed in comoving volume.

        Parameters
        ----------
        n : int
            Number of redshifts to sample.
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute :math:`V_C(z)`.
        rng : numpy.random.Generator or None, optional
            Random number generator.  If *None*, ``numpy.random.default_rng()``
            is used.

        Returns
        -------
        numpy.ndarray
            Redshifts drawn from a uniform-in-comoving-volume distribution,
            shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()

        z_grid = np.linspace(0.0, self.z_max, self._GRID_SIZE)
        # comoving_volume returns a Quantity; extract the numerical values
        v_grid = cosmology.comoving_volume(z_grid).value  # pyright: ignore[reportAttributeAccessIssue]

        u = rng.uniform(0.0, v_grid[-1], size=n)
        return np.interp(u, v_grid, z_grid)


class SkyPatch(Structure):
    """A sky-tile-restricted view of another :class:`Structure` model.

    Wraps a *base* structure model and restricts sampled sky positions to a
    single tile in ``(ra, sin(dec))`` space, scaling the survey volume by the
    tile's fraction of the full sky.  Because ``sin(dec)`` is uniformly
    distributed for an isotropic model, tiling on evenly-spaced
    ``sin(dec)`` intervals gives exactly equal-area tiles with no polar
    distortion.

    This lets a full-volume simulation be split into independent,
    embarrassingly-parallel shards — e.g. one per HTCondor job — without any
    change to the redshift sampling of the wrapped model: redshift and sky
    position are independent for an isotropic large-scale structure model,
    so ``sample_redshifts`` simply delegates to ``base``.

    Parameters
    ----------
    base : Structure
        The structure model being tiled.
    ra_range : tuple[float, float]
        ``(ra_min, ra_max)`` of the tile in radians, with
        ``0 <= ra_min < ra_max <= 2*pi``.
    sin_dec_range : tuple[float, float]
        ``(sin(dec)_min, sin(dec)_max)`` of the tile, with
        ``-1 <= sin_dec_min < sin_dec_max <= 1``.

    Raises
    ------
    ValueError
        If either range is out of bounds or not increasing.

    Examples
    --------
    >>> from astropy.cosmology import Planck18
    >>> base = PointProcess(z_max=3.0)
    >>> tile = SkyPatch(base, ra_range=(0.0, np.pi), sin_dec_range=(0.0, 1.0))
    >>> tile.survey_volume(Planck18) == base.survey_volume(Planck18) * 0.25
    True
    """

    def __init__(
        self,
        base: Structure,
        ra_range: tuple[float, float],
        sin_dec_range: tuple[float, float],
    ) -> None:
        ra_min, ra_max = ra_range
        u_min, u_max = sin_dec_range
        if not (0.0 <= ra_min < ra_max <= 2.0 * np.pi):
            raise ValueError(
                f"invalid ra_range {ra_range!r}: must satisfy 0 <= min < max <= 2*pi"
            )
        if not (-1.0 <= u_min < u_max <= 1.0):
            raise ValueError(
                f"invalid sin_dec_range {sin_dec_range!r}: "
                "must satisfy -1 <= min < max <= 1"
            )
        self.base = base
        self.ra_range = ra_range
        self.sin_dec_range = sin_dec_range

    @property
    def sky_fraction(self) -> float:
        """Return the tile's fraction of the full 4π sr sky."""
        ra_min, ra_max = self.ra_range
        u_min, u_max = self.sin_dec_range
        return (ra_max - ra_min) * (u_max - u_min) / (4.0 * np.pi)

    @property
    def z_max(self) -> float:
        """Return the base model's maximum redshift."""
        return self.base.z_max

    def differential_comoving_volume(
        self, z: np.ndarray, cosmology: FLRW
    ) -> np.ndarray:
        """Return the base model's volume element scaled by :attr:`sky_fraction`.

        Parameters
        ----------
        z : numpy.ndarray
            Redshifts at which to evaluate the volume element, shape ``(n,)``.
        cosmology : astropy.cosmology.FLRW
            Background cosmology, passed through to ``base``.

        Returns
        -------
        numpy.ndarray
            :math:`dV_C/dz` of the tile in Mpc³, shape ``(n,)``.
        """
        return self.base.differential_comoving_volume(z, cosmology) * self.sky_fraction

    def survey_volume(self, cosmology: FLRW) -> float:
        """Return the base model's survey volume scaled by :attr:`sky_fraction`.

        Parameters
        ----------
        cosmology : astropy.cosmology.FLRW
            Background cosmology, passed through to ``base``.

        Returns
        -------
        float
            Comoving volume of the tile in Mpc³.
        """
        return self.base.survey_volume(cosmology) * self.sky_fraction

    def sample_redshifts(
        self,
        n: int,
        cosmology: FLRW,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Delegate to ``base``; redshift is independent of sky position.

        Parameters
        ----------
        n : int
            Number of redshifts to sample.
        cosmology : astropy.cosmology.FLRW
            Background cosmology, passed through to ``base``.
        rng : numpy.random.Generator or None, optional
            Random number generator.

        Returns
        -------
        numpy.ndarray
            Redshifts with shape ``(n,)``.
        """
        return self.base.sample_redshifts(n, cosmology, rng)

    def sample_positions(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw sky positions uniformly within the tile.

        Parameters
        ----------
        n : int
            Number of positions to sample.
        rng : numpy.random.Generator or None, optional
            Random number generator.  If *None*, ``numpy.random.default_rng()``
            is used.

        Returns
        -------
        ra : numpy.ndarray
            Right ascensions in radians, within ``ra_range``, shape ``(n,)``.
        dec : numpy.ndarray
            Declinations in radians, within ``arcsin(sin_dec_range)``,
            shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        ra_min, ra_max = self.ra_range
        u_min, u_max = self.sin_dec_range
        ra = rng.uniform(ra_min, ra_max, size=n)
        dec = np.arcsin(rng.uniform(u_min, u_max, size=n))
        return ra, dec
