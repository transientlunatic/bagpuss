"""Galaxy catalogue — Stage 3 of the bagpuss pipeline.

Applies a selection function to the galaxy set to produce a fake observed
galaxy catalogue, mimicking the observational process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from astropy.cosmology import FLRW

from bagpuss.galaxies import GalaxySet
from bagpuss.universe import LuminosityModel, Structure, Universe

__all__: list[str] = [
    "GalaxyCatalogue",
    "SelectionFunction",
    "MagnitudeLimitedSurvey",
    "build_catalogue",
    "expected_n_observed",
    "sample_observed_redshifts",
]

#: Absolute magnitude of the Sun (bolometric).
_M_SUN_BOLOMETRIC: float = 4.83

#: Number of grid points used to numerically integrate/invert the
#: selection-weighted redshift distribution -- matches
#: ``bagpuss.population._RATE_GRID_SIZE``.
_CATALOGUE_GRID_SIZE: int = 10_000

#: Field names written/read by GalaxyCatalogue.to_zarr / from_zarr.
_CATALOGUE_FIELDS: tuple[str, ...] = (
    "redshifts",
    "luminosities",
    "apparent_magnitudes",
    "ra",
    "dec",
)


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
    ra : numpy.ndarray
        Right ascensions in radians, shape ``(n,)``.
    dec : numpy.ndarray
        Declinations in radians, shape ``(n,)``.

    Examples
    --------
    >>> import numpy as np
    >>> cat = GalaxyCatalogue(
    ...     redshifts=np.array([0.1, 0.2]),
    ...     luminosities=np.array([1e10, 2e10]),
    ...     apparent_magnitudes=np.array([18.5, 19.2]),
    ...     ra=np.array([0.5, 2.1]),
    ...     dec=np.array([-0.3, 0.4]),
    ... )
    >>> len(cat)
    2
    """

    redshifts: np.ndarray
    luminosities: np.ndarray
    apparent_magnitudes: np.ndarray
    ra: np.ndarray
    dec: np.ndarray

    def __len__(self) -> int:
        """Return the number of galaxies in the catalogue."""
        return int(self.redshifts.shape[0])

    def to_zarr(self, group: zarr.Group | str | Path) -> None:
        """Write the catalogue's fields as arrays in a zarr group.

        Each field is stored as a top-level array in *group*, named after
        the field.  Intended for sharded generation: pass a subgroup (e.g.
        one per sky tile) to write an independent, self-contained shard that
        can be created, read, and validated without touching any other
        shard.

        Parameters
        ----------
        group : zarr.Group or str or pathlib.Path
            An open zarr group to write into, or a store path/URL to open
            (in append mode, creating it if it doesn't exist).
        """
        if not isinstance(group, zarr.Group):
            group = zarr.open_group(store=str(group), mode="a")
        for field in _CATALOGUE_FIELDS:
            group.create_array(field, data=getattr(self, field), overwrite=True)

    @classmethod
    def from_zarr(cls, group: zarr.Group | str | Path) -> GalaxyCatalogue:
        """Load a catalogue from a zarr group written by :meth:`to_zarr`.

        Parameters
        ----------
        group : zarr.Group or str or pathlib.Path
            An open zarr group, or a store path/URL to open read-only.

        Returns
        -------
        GalaxyCatalogue
        """
        if not isinstance(group, zarr.Group):
            group = zarr.open_group(store=str(group), mode="r")
        data = {field: np.asarray(group[field]) for field in _CATALOGUE_FIELDS}
        return cls(**data)


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
                ra=empty,
                dec=empty,
            )

        d_L_pc = (
            cosmology.luminosity_distance(galaxies.redshifts)  # pyright: ignore[reportAttributeAccessIssue]
            .to("pc")
            .value
        )
        distance_modulus = 5.0 * np.log10(d_L_pc / 10.0)
        abs_magnitudes = self.m_sun - 2.5 * np.log10(galaxies.luminosities)
        apparent_magnitudes = abs_magnitudes + distance_modulus

        mask = apparent_magnitudes <= self.m_lim
        return GalaxyCatalogue(
            redshifts=galaxies.redshifts[mask],
            luminosities=galaxies.luminosities[mask],
            apparent_magnitudes=apparent_magnitudes[mask],
            ra=galaxies.ra[mask],
            dec=galaxies.dec[mask],
        )

    def completeness(
        self,
        redshifts: np.ndarray,
        ra: np.ndarray,
        dec: np.ndarray,
        cosmology: FLRW,
        luminosity: LuminosityModel,
    ) -> np.ndarray:
        r"""Return the survey completeness at each given position.

        The completeness is the probability that a galaxy at a given
        redshift would be brighter than the survey's limiting magnitude,
        i.e. the fraction of the luminosity function ``luminosity`` that
        lies above the apparent-magnitude threshold once the distance
        modulus at that redshift is accounted for:

        .. math::

            C(z) = \mathrm{CDF}_{\,L}\!\bigl(m_{\mathrm{lim}} - \mu(z)\bigr)

        This is used to determine, for a randomly drawn host galaxy of
        unknown luminosity, the probability that it would have been
        included in the observed catalogue.

        ``ra`` and ``dec`` are accepted but currently unused — completeness
        depends only on redshift for a uniform-depth, magnitude-limited
        survey.  They are part of the signature so that future sky-dependent
        completeness models (survey footprints, Galactic-plane masking) can
        be substituted without changing call sites.

        Parameters
        ----------
        redshifts : numpy.ndarray
            Redshifts at which to evaluate completeness, shape ``(n,)``.
        ra : numpy.ndarray
            Right ascensions in radians, shape ``(n,)``.  Currently unused.
        dec : numpy.ndarray
            Declinations in radians, shape ``(n,)``.  Currently unused.
        cosmology : astropy.cosmology.FLRW
            Background cosmology used to compute the distance modulus.
        luminosity : bagpuss.universe.LuminosityModel
            Luminosity model whose :meth:`~bagpuss.universe.LuminosityModel.cdf`
            gives the fraction of galaxies brighter than a given absolute
            magnitude.

        Returns
        -------
        numpy.ndarray
            Completeness at each position, shape ``(n,)``, values in
            ``[0, 1]``.
        """
        d_L_pc = cosmology.luminosity_distance(redshifts).to("pc").value  # pyright: ignore[reportAttributeAccessIssue]
        distance_modulus = 5.0 * np.log10(d_L_pc / 10.0)
        magnitude_threshold = self.m_lim - distance_modulus
        return luminosity.cdf(magnitude_threshold)


# ---------------------------------------------------------------------------
# Selection-weighted redshift sampling
# ---------------------------------------------------------------------------


def _selection_weight_grid(
    structure: Structure,
    cosmology: FLRW,
    luminosity: LuminosityModel,
    selection: MagnitudeLimitedSurvey,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the ``(z, weight)`` grid for the observable population.

    ``weight(z) = luminosity.number_density() * completeness(z) * dV_C/dz`` --
    the raw galaxy number density, restricted to the fraction bright enough
    to pass ``selection`` at that redshift, converted to a count per unit
    redshift. Mirrors :func:`bagpuss.population._rate_weight_grid`'s
    grid-based approach, with ``completeness(z)`` in place of the
    time-dilated merger-rate weight.
    """
    z_grid = np.linspace(0.0, structure.z_max, grid_size)
    dvc_dz = structure.differential_comoving_volume(z_grid, cosmology)
    completeness = selection.completeness(
        redshifts=z_grid,
        ra=np.zeros_like(z_grid),
        dec=np.zeros_like(z_grid),
        cosmology=cosmology,
        luminosity=luminosity,
    )
    weight = luminosity.number_density() * completeness * dvc_dz
    return z_grid, weight


def expected_n_observed(
    structure: Structure,
    cosmology: FLRW,
    luminosity: LuminosityModel,
    selection: MagnitudeLimitedSurvey,
    grid_size: int = _CATALOGUE_GRID_SIZE,
) -> float:
    r"""Return the expected number of galaxies that pass ``selection``.

    Computes

    .. math::

        N = \int_0^{z_{\max}} \phi \; C(z) \; \frac{dV_C}{dz} \, dz

    where :math:`\phi` is ``luminosity.number_density()`` and :math:`C(z)`
    is the survey completeness at redshift :math:`z`
    (:meth:`~bagpuss.catalogue.MagnitudeLimitedSurvey.completeness`). This is
    always less than (or equal to) the raw ``phi * survey_volume()`` figure,
    since :math:`C(z) \le 1` -- see :func:`build_catalogue`.

    Parameters
    ----------
    structure : bagpuss.universe.Structure
        Structure model providing ``z_max`` and
        :meth:`~bagpuss.universe.Structure.differential_comoving_volume`.
    cosmology : astropy.cosmology.FLRW
        Background cosmology.
    luminosity : bagpuss.universe.LuminosityModel
        Luminosity model.
    selection : MagnitudeLimitedSurvey
        Selection function.
    grid_size : int, optional
        Number of points used to numerically integrate over ``[0, z_max]``.

    Returns
    -------
    float
        Expected number of selection-passing galaxies.
    """
    z_grid, weight = _selection_weight_grid(
        structure, cosmology, luminosity, selection, grid_size
    )
    return float(np.trapezoid(weight, z_grid))


def sample_observed_redshifts(
    structure: Structure,
    cosmology: FLRW,
    luminosity: LuminosityModel,
    selection: MagnitudeLimitedSurvey,
    n: int,
    rng: np.random.Generator | None = None,
    grid_size: int = _CATALOGUE_GRID_SIZE,
) -> np.ndarray:
    r"""Draw redshifts weighted by survey completeness.

    Samples from :math:`p(z) \propto \phi \; C(z) \; dV_C/dz` (see
    :func:`_selection_weight_grid`) via inverse-CDF on a precomputed grid,
    the same technique :func:`bagpuss.population.sample_merger_redshifts`
    uses for the rate-weighted (injection) case.

    Parameters
    ----------
    structure : bagpuss.universe.Structure
        Structure model providing ``z_max`` and
        :meth:`~bagpuss.universe.Structure.differential_comoving_volume`.
    cosmology : astropy.cosmology.FLRW
        Background cosmology.
    luminosity : bagpuss.universe.LuminosityModel
        Luminosity model.
    selection : MagnitudeLimitedSurvey
        Selection function.
    n : int
        Number of redshifts to sample.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If *None*, ``numpy.random.default_rng()``
        is used.
    grid_size : int, optional
        Number of points used to build the inverse-CDF grid.

    Returns
    -------
    numpy.ndarray
        Redshifts, shape ``(n,)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    z_grid, weight = _selection_weight_grid(
        structure, cosmology, luminosity, selection, grid_size
    )
    dz = np.diff(z_grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[:-1] + weight[1:]) * dz)])
    cdf /= cdf[-1]

    u = rng.uniform(0.0, 1.0, size=n)
    return np.asarray(np.interp(u, cdf, z_grid))


# ---------------------------------------------------------------------------
# Catalogue builder
# ---------------------------------------------------------------------------


def build_catalogue(
    universe: Universe,
    selection: MagnitudeLimitedSurvey,
    rng: np.random.Generator | None = None,
) -> GalaxyCatalogue:
    """Build a Poisson realisation of the magnitude-limited galaxy catalogue.

    The expected total number of galaxies is
    :func:`expected_n_observed` -- the luminosity function's number density,
    restricted at each redshift to the fraction bright enough to pass
    ``selection`` (:meth:`~MagnitudeLimitedSurvey.completeness`), integrated
    over the survey volume. A Poisson draw from this expected count gives
    the catalogue size directly.

    Every returned galaxy satisfies the magnitude limit *by construction*:
    redshifts are drawn from the completeness-weighted distribution
    (:func:`sample_observed_redshifts`) and each galaxy's luminosity is then
    drawn conditioned on being bright enough to pass selection at its own
    redshift (:meth:`~bagpuss.universe.LuminosityModel.sample_brighter_than`).
    This never draws (and discards) the unobservable majority of the raw
    population -- unlike a naive "draw everything down to ``m_min``, then
    filter" approach, which is intractable at realistic survey depths and
    volumes (a full-``phi_star`` real-``z_max`` draw needs ~10¹¹ raw galaxies
    per unsharded tile before filtering, most of which are discarded).

    The catalogue size is not specified by the caller — it emerges from
    the physics of the luminosity function, survey volume, and selection
    function, just as it would in a real survey.  For this to give a
    meaningful absolute count, ``phi_star`` in the luminosity model must be
    supplied in physical units (Mpc⁻³ mag⁻¹).

    Parameters
    ----------
    universe : Universe
        Simulated universe providing the luminosity and structure models.
    selection : MagnitudeLimitedSurvey
        Magnitude-limited selection function to apply to the galaxy
        population.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If *None*, ``numpy.random.default_rng()``
        is used.

    Returns
    -------
    GalaxyCatalogue
        A Poisson realisation of the observable galaxy catalogue.  All
        returned galaxies satisfy the magnitude limit imposed by
        ``selection``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = int(
        rng.poisson(
            expected_n_observed(
                universe.structure, universe.cosmology, universe.luminosity, selection
            )
        )
    )

    if n_total == 0:
        empty: np.ndarray = np.array([], dtype=float)
        return GalaxyCatalogue(
            redshifts=empty,
            luminosities=empty,
            apparent_magnitudes=empty,
            ra=empty,
            dec=empty,
        )

    redshifts = sample_observed_redshifts(
        universe.structure,
        universe.cosmology,
        universe.luminosity,
        selection,
        n_total,
        rng,
    )
    ra, dec = universe.structure.sample_positions(n_total, rng)

    d_L_pc = universe.cosmology.luminosity_distance(redshifts).to("pc").value  # pyright: ignore[reportAttributeAccessIssue]
    distance_modulus = 5.0 * np.log10(d_L_pc / 10.0)
    magnitude_threshold = selection.m_lim - distance_modulus
    luminosities = universe.luminosity.sample_brighter_than(
        n_total, magnitude_threshold, rng
    )
    abs_magnitudes = selection.m_sun - 2.5 * np.log10(luminosities)
    apparent_magnitudes = abs_magnitudes + distance_modulus

    return GalaxyCatalogue(
        redshifts=redshifts,
        luminosities=luminosities,
        apparent_magnitudes=apparent_magnitudes,
        ra=ra,
        dec=dec,
    )
