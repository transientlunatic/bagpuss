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
from astropy.cosmology import Cosmology

from bagpuss.galaxies import GalaxySet

__all__: list[str] = [
    "Universe",
    "Structure",
    "LuminosityModel",
    "PointProcess",
]


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class Structure(ABC):
    """Abstract base class for galaxy large-scale structure models.

    A structure model is responsible for drawing the *spatial* distribution
    of galaxies — specifically their redshifts — given a background cosmology.
    Subclasses must implement :meth:`sample_redshifts`.
    """

    @abstractmethod
    def sample_redshifts(
        self,
        n: int,
        cosmology: Cosmology,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw redshifts for a set of galaxies.

        Parameters
        ----------
        n : int
            Number of galaxies to sample.
        cosmology : astropy.cosmology.Cosmology
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


class LuminosityModel(ABC):
    """Abstract base class for galaxy luminosity distribution models.

    A luminosity model is responsible for drawing galaxy luminosities
    independently of their spatial positions.  Subclasses must implement
    :meth:`sample`.
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
    cosmology : astropy.cosmology.Cosmology
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
        cosmology: Cosmology,
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
        return GalaxySet(redshifts=redshifts, luminosities=luminosities)


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
        self.z_max = z_max

    def sample_redshifts(
        self,
        n: int,
        cosmology: Cosmology,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw redshifts uniformly distributed in comoving volume.

        Parameters
        ----------
        n : int
            Number of redshifts to sample.
        cosmology : astropy.cosmology.Cosmology
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
        v_grid = cosmology.comoving_volume(z_grid).value

        u = rng.uniform(0.0, v_grid[-1], size=n)
        return np.interp(u, v_grid, z_grid)
