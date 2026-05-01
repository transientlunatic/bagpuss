"""Galaxy catalogue — Stage 3 of the bagpuss pipeline.

Applies a selection function to the galaxy set to produce a fake observed
galaxy catalogue, mimicking the observational process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from astropy.cosmology import FLRW

from bagpuss.galaxies import GalaxySet

__all__: list[str] = ["GalaxyCatalogue", "SelectionFunction", "MagnitudeLimitedSurvey"]

#: Absolute magnitude of the Sun (bolometric).
_M_SUN_BOLOMETRIC: float = 4.83


@dataclass
class GalaxyCatalogue:
    """An observed galaxy catalogue produced by applying a selection function.

    Parameters
    ----------
    redshifts : numpy.ndarray
        Redshifts of selected galaxies, shape ``(n,)``.
    luminosities : numpy.ndarray
        Luminosities of selected galaxies in solar luminosities, shape ``(n,)``.
    apparent_magnitudes : numpy.ndarray
        Apparent magnitudes of selected galaxies in the survey band, shape ``(n,)``.

    Examples
    --------
    >>> import numpy as np
    >>> cat = GalaxyCatalogue(
    ...     redshifts=np.array([0.1, 0.2]),
    ...     luminosities=np.array([1e10, 2e10]),
    ...     apparent_magnitudes=np.array([18.5, 19.2]),
    ... )
    >>> len(cat)
    2
    """

    redshifts: np.ndarray
    luminosities: np.ndarray
    apparent_magnitudes: np.ndarray

    def __len__(self) -> int:
        """Return the number of galaxies in the catalogue."""
        return int(self.redshifts.shape[0])


class SelectionFunction(ABC):
    """Abstract base class for galaxy survey selection functions.

    A selection function takes a :class:`~bagpuss.galaxies.GalaxySet` and a
    background cosmology, and returns a :class:`GalaxyCatalogue` containing
    only the galaxies that would be observed by the survey.

    Subclasses must implement :meth:`apply`.
    """

    @abstractmethod
    def apply(
        self,
        galaxies: GalaxySet,
        cosmology: FLRW,
    ) -> GalaxyCatalogue:
        """Apply the selection function to a galaxy set.

        Parameters
        ----------
        galaxies : bagpuss.galaxies.GalaxySet
            The full galaxy set from which the observed catalogue is drawn.
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute distances.

        Returns
        -------
        GalaxyCatalogue
            The subset of galaxies that pass the selection.
        """


class MagnitudeLimitedSurvey(SelectionFunction):
    r"""A magnitude-limited galaxy survey selection function.

    Selects galaxies whose apparent magnitude in the survey band is at or
    brighter than the survey's limiting magnitude, i.e. galaxies bright
    enough to be detected.

    The apparent magnitude of each galaxy is computed from its luminosity
    and luminosity distance:

    .. math::

        m = M + \mu, \quad
        M = M_\odot - 2.5 \log_{10}\!\left(\frac{L}{L_\odot}\right), \quad
        \mu = 5 \log_{10}\!\left(\frac{d_L}{10 \,\mathrm{pc}}\right)

    where :math:`d_L` is the luminosity distance in parsecs.  A galaxy is
    selected if :math:`m \leq m_{\mathrm{lim}}`.

    Parameters
    ----------
    m_lim : float
        Limiting apparent magnitude of the survey.  Galaxies with
        :math:`m \leq m_{\mathrm{lim}}` are selected.
    m_sun : float, optional
        Absolute magnitude of the Sun in the survey band, used to convert
        solar luminosities to absolute magnitudes.  Defaults to ``4.83``
        (bolometric).
    """

    def __init__(self, m_lim: float, m_sun: float = _M_SUN_BOLOMETRIC) -> None:
        self.m_lim = m_lim
        self.m_sun = m_sun

    def apply(
        self,
        galaxies: GalaxySet,
        cosmology: FLRW,
    ) -> GalaxyCatalogue:
        """Apply the magnitude limit to select observable galaxies.

        Parameters
        ----------
        galaxies : bagpuss.galaxies.GalaxySet
            The full galaxy set.
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute luminosity distances.

        Returns
        -------
        GalaxyCatalogue
            Galaxies with apparent magnitude at or brighter than ``m_lim``.
        """
        if len(galaxies) == 0:
            empty = np.array([], dtype=float)
            return GalaxyCatalogue(
                redshifts=empty,
                luminosities=empty,
                apparent_magnitudes=empty,
            )

        d_L_pc = cosmology.luminosity_distance(galaxies.redshifts).to("pc").value
        distance_modulus = 5.0 * np.log10(d_L_pc / 10.0)
        abs_magnitudes = self.m_sun - 2.5 * np.log10(galaxies.luminosities)
        apparent_magnitudes = abs_magnitudes + distance_modulus

        mask = apparent_magnitudes <= self.m_lim
        return GalaxyCatalogue(
            redshifts=galaxies.redshifts[mask],
            luminosities=galaxies.luminosities[mask],
            apparent_magnitudes=apparent_magnitudes[mask],
        )
