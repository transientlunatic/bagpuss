"""GW injection set — Stage 5 of the bagpuss pipeline.

Assigns host galaxies from a :class:`~bagpuss.catalogue.GalaxyCatalogue` to
BBH events drawn from a :class:`~bagpuss.population.PopulationModel`, draws
the remaining extrinsic parameters, applies a
:class:`Detectable` filter, and produces an :class:`InjectionSet` that can
be written to HDF5 for use by downstream parameter-estimation and inference
tools (bilby, pycbc, etc.).

The pipeline is exposed at two levels of granularity:

* :func:`create_injection_set` — convenience wrapper that runs the full
  pipeline in one call.
* :func:`sample_host_galaxies` + :func:`assemble_injection_set` — the two
  composable stages, useful for notebook exploration or when the same host
  sample needs to be paired with multiple population models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import zarr
from astropy.cosmology import FLRW

from bagpuss.catalogue import GalaxyCatalogue, MagnitudeLimitedSurvey
from bagpuss.population import (
    BBHSet,
    MergerRate,
    PopulationModel,
    expected_n_mergers,
    sample_merger_redshifts,
)
from bagpuss.universe import Universe

__all__: list[str] = [
    "InjectionSet",
    "HostAssignment",
    "Detectable",
    "DistanceThreshold",
    "sample_host_galaxies",
    "assemble_injection_set",
    "create_injection_set",
    "build_injection_set",
    "GPS_O3_START",
    "GPS_O3_END",
]

#: Default GPS start of O3a (2019-04-01 15:00:00 UTC).
GPS_O3_START: float = 1_238_166_018.0
#: Default GPS end of O3b (2020-03-27 17:00:00 UTC).
GPS_O3_END: float = 1_269_363_618.0

_INJECTION_FIELDS: tuple[str, ...] = (
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
    "host_galaxy_index",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class InjectionSet:
    r"""A set of simulated GW injections with full extrinsic parameters.

    Each row corresponds to one BBH event assigned to a host galaxy drawn
    from a :class:`~bagpuss.catalogue.GalaxyCatalogue`.

    Parameters
    ----------
    m1_source, m2_source : numpy.ndarray
        Source-frame component masses in :math:`M_\odot`, shape ``(n,)``.
    a1, a2 : numpy.ndarray
        Spin magnitudes, shape ``(n,)``.
    cos_tilt1, cos_tilt2 : numpy.ndarray
        Cosines of spin tilt angles, shape ``(n,)``.
    phi12, phi_jl : numpy.ndarray
        Azimuthal spin angles in radians, shape ``(n,)``.
    theta_jn : numpy.ndarray
        Inclination angle (angle between total angular momentum and
        line-of-sight) in radians, shape ``(n,)``.  Values in ``[0, π]``.
    ra : numpy.ndarray
        Right ascension of the host galaxy in radians, shape ``(n,)``.
    dec : numpy.ndarray
        Declination of the host galaxy in radians, shape ``(n,)``.
    psi : numpy.ndarray
        Gravitational-wave polarisation angle in radians, shape ``(n,)``.
        Values in ``[0, π)``.
    geocent_time : numpy.ndarray
        Geocentric GPS merger time in seconds, shape ``(n,)``.
    redshift : numpy.ndarray
        Redshift of the host galaxy, shape ``(n,)``.
    luminosity_distance : numpy.ndarray
        Luminosity distance to the host galaxy in Mpc, shape ``(n,)``.
    host_galaxy_index : numpy.ndarray
        Integer index into the :class:`~bagpuss.catalogue.GalaxyCatalogue`
        used to generate these injections, shape ``(n,)``.  A value of
        ``-1`` means the host galaxy is not in the catalogue, i.e. it is
        fainter than the survey's magnitude limit.
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
    host_galaxy_index: np.ndarray

    def __len__(self) -> int:
        """Return the number of injections."""
        return int(self.m1_source.shape[0])

    def to_hdf5(self, path: str | Path) -> None:
        """Write the injection set to an HDF5 file.

        Each parameter is stored as a dataset inside a top-level
        ``/injections`` group.  The file is compatible with bilby's
        ``InjectionSet`` reader.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.  An existing file is overwritten.
        """
        with h5py.File(path, "w") as f:
            grp = f.create_group("injections")
            for field in _INJECTION_FIELDS:
                grp.create_dataset(field, data=getattr(self, field))

    @classmethod
    def from_hdf5(cls, path: str | Path) -> InjectionSet:
        """Load an injection set from an HDF5 file written by :meth:`to_hdf5`.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to an HDF5 file containing an ``/injections`` group.

        Returns
        -------
        InjectionSet
        """
        with h5py.File(path, "r") as f:
            grp = cast(h5py.Group, f["injections"])
            data = {
                field: cast(h5py.Dataset, grp[field])[()] for field in _INJECTION_FIELDS
            }
        return cls(**data)

    def to_zarr(self, group: zarr.Group | str | Path) -> None:
        """Write the injection set's fields as arrays in a zarr group.

        Each field is stored as a top-level array in *group*, named after
        the field. Intended for sharded generation: pass a subgroup (e.g.
        one per injection shard) to write an independent, self-contained
        shard that can be created, read, and validated without touching any
        other shard.

        Parameters
        ----------
        group : zarr.Group or str or pathlib.Path
            An open zarr group to write into, or a store path/URL to open
            (in append mode, creating it if it doesn't exist).
        """
        if not isinstance(group, zarr.Group):
            group = zarr.open_group(store=str(group), mode="a")
        for field in _INJECTION_FIELDS:
            group.create_array(field, data=getattr(self, field), overwrite=True)

    @classmethod
    def from_zarr(cls, group: zarr.Group | str | Path) -> InjectionSet:
        """Load an injection set from a zarr group written by :meth:`to_zarr`.

        Parameters
        ----------
        group : zarr.Group or str or pathlib.Path
            An open zarr group, or a store path/URL to open read-only.

        Returns
        -------
        InjectionSet
        """
        if not isinstance(group, zarr.Group):
            group = zarr.open_group(store=str(group), mode="r")
        data = {field: np.asarray(group[field]) for field in _INJECTION_FIELDS}
        return cls(**data)


@dataclass
class HostAssignment:
    """Host galaxy assignments for a set of GW events.

    The intermediate product of :func:`sample_host_galaxies`.  Each row
    holds the sky position and redshift of the assigned host galaxy —
    either a specific entry from the observed catalogue (for
    catalogue-hosted events) or a trial draw from the full underlying
    galaxy population (for field events too faint to be catalogued).

    Parameters
    ----------
    redshift : numpy.ndarray
        Redshift of the host galaxy, shape ``(n,)``.
    ra : numpy.ndarray
        Right ascension of the host galaxy in radians, shape ``(n,)``.
    dec : numpy.ndarray
        Declination of the host galaxy in radians, shape ``(n,)``.
    host_galaxy_index : numpy.ndarray
        Integer index into the :class:`~bagpuss.catalogue.GalaxyCatalogue`
        used to generate these assignments, shape ``(n,)``.  A value of
        ``-1`` means the host galaxy is not in the catalogue.
    """

    redshift: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    host_galaxy_index: np.ndarray

    def __len__(self) -> int:
        """Return the number of host assignments."""
        return int(self.redshift.shape[0])


# ---------------------------------------------------------------------------
# Detectability models
# ---------------------------------------------------------------------------


class Detectable(ABC):
    """Abstract base class for GW detectability models.

    A detectability model is a callable that maps a parameter dictionary to
    an array of detection probabilities in ``[0, 1]``.  Events are kept if
    ``rng.uniform() < p_det``.

    Subclasses must implement :meth:`__call__`.
    """

    @abstractmethod
    def __call__(self, params: dict[str, np.ndarray]) -> np.ndarray:
        """Return per-event detection probabilities.

        Parameters
        ----------
        params : dict
            Dictionary containing at least ``luminosity_distance`` (in Mpc)
            and any other fields required by the specific model.

        Returns
        -------
        numpy.ndarray
            Detection probabilities, shape ``(n,)``, values in ``[0, 1]``.
        """


class DistanceThreshold(Detectable):
    """Step-function detectability: all events within ``d_max`` are detected.

    This is the simplest possible detectability model — a hard cut on
    luminosity distance.  It is useful for quick sanity checks and as a
    baseline for comparison with more realistic models.

    Parameters
    ----------
    d_max : float
        Maximum luminosity distance in Mpc.  Events with
        ``luminosity_distance <= d_max`` are detected with probability 1;
        all others are not detected.

    Raises
    ------
    ValueError
        If ``d_max <= 0``.
    """

    def __init__(self, d_max: float) -> None:
        if d_max <= 0.0:
            raise ValueError(f"d_max must be positive, got {d_max!r}")
        self.d_max = d_max

    def __call__(self, params: dict[str, np.ndarray]) -> np.ndarray:
        """Return 1.0 for events within d_max, 0.0 for those beyond.

        Parameters
        ----------
        params : dict
            Must contain ``luminosity_distance`` in Mpc.

        Returns
        -------
        numpy.ndarray
            Detection probabilities, shape ``(n,)``.
        """
        return (params["luminosity_distance"] <= self.d_max).astype(float)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def sample_host_galaxies(
    universe: Universe,
    catalogue: GalaxyCatalogue,
    selection: MagnitudeLimitedSurvey,
    n: int,
    rng: np.random.Generator | None = None,
    redshift: np.ndarray | None = None,
) -> HostAssignment:
    """Draw host galaxy assignments for ``n`` potential GW events.

    For each event a trial host redshift and sky position are drawn from
    the full galaxy population.  The survey completeness at that redshift
    determines the probability that the host would have been bright enough
    to appear in ``catalogue``.  Events that pass this probabilistic cut
    are assigned a randomly chosen row from the actual catalogue; the rest
    keep their trial (uncatalogued) position.

    Parameters
    ----------
    universe : Universe
        Simulated universe providing the structure and luminosity models
        used to draw trial host redshifts/positions and evaluate
        completeness.
    catalogue : GalaxyCatalogue
        Observed galaxy catalogue, used as the host population for events
        that fall above the survey's magnitude limit.
    selection : MagnitudeLimitedSurvey
        Selection function used to compute the probability that a host at
        a given redshift is bright enough to be in ``catalogue``.
    n : int
        Number of host assignments to draw.  Ignored if ``redshift`` is
        given, in which case ``len(redshift)`` is used instead.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If *None*, ``numpy.random.default_rng()``
        is used.
    redshift : numpy.ndarray or None, optional
        Precomputed trial host redshifts, shape ``(n,)``.  If *None*
        (the default), redshifts are drawn from
        ``universe.structure.sample_redshifts`` as usual.  Pass this to
        substitute a different redshift distribution -- e.g. one weighted
        by a :class:`~bagpuss.population.MergerRate` via
        :func:`~bagpuss.population.sample_merger_redshifts`, as
        :func:`build_injection_set` does -- while still drawing sky
        positions and evaluating catalogue completeness the usual way.

    Returns
    -------
    HostAssignment
        Host redshifts, sky positions, and catalogue indices for ``n``
        events.  ``host_galaxy_index == -1`` for uncatalogued hosts.

    Notes
    -----
    A catalogued event's host is a uniformly random catalogue row, whatever
    its trial redshift. If ``redshift`` is drawn from a merger-rate model, the
    redshifts of catalogued events therefore follow the catalogue's own
    distribution rather than the rate's :math:`R(z)/(1+z)` weighting (for a
    constant rate, a factor of :math:`(1+z)`); uncatalogued events do follow it.
    The effect is small unless catalogued hosts are common.
    """
    if rng is None:
        rng = np.random.default_rng()

    cosmology = universe.cosmology
    n_cat = len(catalogue)

    if redshift is None:
        redshift = universe.structure.sample_redshifts(n, cosmology, rng)
    else:
        redshift = np.array(redshift, dtype=float, copy=True)
        n = redshift.shape[0]
    ra, dec = universe.structure.sample_positions(n, rng)

    completeness = selection.completeness(
        redshift, ra, dec, cosmology, universe.luminosity
    )
    host_observed = rng.uniform(0.0, 1.0, size=n) < completeness

    host_galaxy_index = np.full(n, -1, dtype=int)
    n_observed = int(host_observed.sum())
    # With an empty catalogue nothing can be catalogued: events flagged as
    # observed keep their trial (uncatalogued) position.
    if n_observed > 0 and n_cat > 0:
        cat_idx = rng.integers(0, n_cat, size=n_observed)
        redshift[host_observed] = catalogue.redshifts[cat_idx]
        ra[host_observed] = catalogue.ra[cat_idx]
        dec[host_observed] = catalogue.dec[cat_idx]
        host_galaxy_index[host_observed] = cat_idx

    return HostAssignment(
        redshift=redshift,
        ra=ra,
        dec=dec,
        host_galaxy_index=host_galaxy_index,
    )


def assemble_injection_set(
    hosts: HostAssignment,
    bbh: BBHSet,
    cosmology: FLRW,
    detectable: Detectable | None = None,
    rng: np.random.Generator | None = None,
    t_start: float = GPS_O3_START,
    t_end: float = GPS_O3_END,
) -> InjectionSet:
    """Combine host assignments and BBH parameters into an injection set.

    Computes luminosity distances from host redshifts, draws extrinsic
    orientation and timing parameters, applies an optional detectability
    filter, and returns the surviving events as an :class:`InjectionSet`.

    Parameters
    ----------
    hosts : HostAssignment
        Host galaxy redshifts, sky positions, and catalogue indices as
        returned by :func:`sample_host_galaxies`.
    bbh : BBHSet
        Intrinsic BBH parameters (masses and spins) as returned by
        :meth:`~bagpuss.population.PopulationModel.sample`.  Must have the
        same length as ``hosts``.
    cosmology : astropy.cosmology.FLRW
        Background cosmology used to convert redshifts to luminosity
        distances.
    detectable : Detectable or None, optional
        Detectability model.  If *None*, all events are kept.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If *None*, ``numpy.random.default_rng()``
        is used.
    t_start : float, optional
        Start of the observation window in GPS seconds.
    t_end : float, optional
        End of the observation window in GPS seconds.

    Returns
    -------
    InjectionSet
        The (filtered) set of simulated GW injections.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(hosts)
    luminosity_distance = (
        cosmology.luminosity_distance(hosts.redshift).to("Mpc").value  # pyright: ignore[reportAttributeAccessIssue]
    )

    cos_theta_jn = rng.uniform(-1.0, 1.0, size=n)
    theta_jn = np.arccos(cos_theta_jn)
    psi = rng.uniform(0.0, np.pi, size=n)
    geocent_time = rng.uniform(t_start, t_end, size=n)

    params = {
        "m1_source": bbh.m1_source,
        "m2_source": bbh.m2_source,
        "luminosity_distance": luminosity_distance,
        "redshift": hosts.redshift,
        "ra": hosts.ra,
        "dec": hosts.dec,
        "theta_jn": theta_jn,
    }

    if detectable is not None:
        p_det = detectable(params)
        keep = rng.uniform(0.0, 1.0, size=n) < p_det
    else:
        keep = np.ones(n, dtype=bool)

    return InjectionSet(
        m1_source=bbh.m1_source[keep],
        m2_source=bbh.m2_source[keep],
        a1=bbh.a1[keep],
        a2=bbh.a2[keep],
        cos_tilt1=bbh.cos_tilt1[keep],
        cos_tilt2=bbh.cos_tilt2[keep],
        phi12=bbh.phi12[keep],
        phi_jl=bbh.phi_jl[keep],
        theta_jn=theta_jn[keep],
        ra=hosts.ra[keep],
        dec=hosts.dec[keep],
        psi=psi[keep],
        geocent_time=geocent_time[keep],
        redshift=hosts.redshift[keep],
        luminosity_distance=luminosity_distance[keep],
        host_galaxy_index=hosts.host_galaxy_index[keep],
    )


def create_injection_set(
    universe: Universe,
    catalogue: GalaxyCatalogue,
    selection: MagnitudeLimitedSurvey,
    population: PopulationModel,
    n_draw: int,
    detectable: Detectable | None = None,
    rng: np.random.Generator | None = None,
    t_start: float = GPS_O3_START,
    t_end: float = GPS_O3_END,
) -> InjectionSet:
    """Create a simulated GW injection set from the full galaxy population.

    Convenience wrapper that runs the full pipeline in a single call:

    1. Draw intrinsic BBH parameters from ``population``.
    2. Assign host galaxies via :func:`sample_host_galaxies`.
    3. Assemble the injection set via :func:`assemble_injection_set`.

    For step-by-step control — e.g. to inspect the host sample before
    drawing BBH parameters, or to pair the same hosts with multiple
    population models — call the sub-functions directly.

    Parameters
    ----------
    universe : Universe
        Simulated universe providing the structure and luminosity models.
    catalogue : GalaxyCatalogue
        Observed galaxy catalogue.
    selection : MagnitudeLimitedSurvey
        Selection function used to compute survey completeness.
    population : PopulationModel
        BBH population model used to draw intrinsic parameters.
    n_draw : int
        Number of BBH events to draw before applying the detectability
        filter.
    detectable : Detectable or None, optional
        Detectability model.  If *None*, all ``n_draw`` events are kept.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If *None*, ``numpy.random.default_rng()``
        is used.
    t_start : float, optional
        Start of the observation window in GPS seconds.  Defaults to the
        O3 start (2019-04-01).
    t_end : float, optional
        End of the observation window in GPS seconds.  Defaults to the O3
        end (2020-03-27).

    Returns
    -------
    InjectionSet
        The (filtered) set of simulated GW injections.
    """
    if rng is None:
        rng = np.random.default_rng()

    bbh = population.sample(n_draw, rng)
    hosts = sample_host_galaxies(universe, catalogue, selection, n_draw, rng)
    return assemble_injection_set(
        hosts, bbh, universe.cosmology, detectable, rng, t_start, t_end
    )


def build_injection_set(
    universe: Universe,
    catalogue: GalaxyCatalogue,
    selection: MagnitudeLimitedSurvey,
    population: PopulationModel,
    rate: MergerRate,
    detectable: Detectable | None = None,
    rng: np.random.Generator | None = None,
    t_start: float = GPS_O3_START,
    t_end: float = GPS_O3_END,
) -> InjectionSet:
    r"""Create a Poisson realisation of the GW injection set from ``rate``.

    Unlike :func:`create_injection_set`, the number of events is not
    supplied by the caller: it is drawn from a Poisson distribution whose
    mean is set by ``rate`` (a
    :class:`~bagpuss.population.MergerRate`), ``universe.structure``'s
    survey volume, and the observation window
    ``[t_start, t_end]`` -- see
    :func:`~bagpuss.population.expected_n_mergers`. Event redshifts are
    then drawn from the rate-weighted distribution via
    :func:`~bagpuss.population.sample_merger_redshifts`, rather than the
    unweighted galaxy distribution used by :func:`sample_host_galaxies`
    (and hence :func:`create_injection_set`) by default.

    This mirrors :func:`~bagpuss.catalogue.build_catalogue`, which
    Poisson-realises the galaxy count from
    ``luminosity.number_density() * structure.survey_volume()`` instead of
    taking a caller-supplied target count.

    Parameters
    ----------
    universe : Universe
        Simulated universe providing the structure, luminosity, and
        cosmology used for host assignment.
    catalogue : GalaxyCatalogue
        Observed galaxy catalogue.
    selection : MagnitudeLimitedSurvey
        Selection function used to compute survey completeness.
    population : PopulationModel
        BBH population model used to draw intrinsic parameters.
    rate : MergerRate
        Merger-rate density model setting the expected number of events.
    detectable : Detectable or None, optional
        Detectability model.  If *None*, every drawn event is kept.
    rng : numpy.random.Generator or None, optional
        Random number generator.  If *None*, ``numpy.random.default_rng()``
        is used.
    t_start : float, optional
        Start of the observation window in GPS seconds.  Defaults to the
        O3 start (2019-04-01).
    t_end : float, optional
        End of the observation window in GPS seconds.  Defaults to the O3
        end (2020-03-27).

    Returns
    -------
    InjectionSet
        The (filtered) set of simulated GW injections.  Its size is a
        Poisson realisation of the rate-implied expected count, not a
        caller-chosen target.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_expected = expected_n_mergers(
        universe.structure, universe.cosmology, rate, t_start, t_end
    )
    n = int(rng.poisson(n_expected))

    redshift = sample_merger_redshifts(
        universe.structure, universe.cosmology, rate, n, rng
    )
    bbh = population.sample(n, rng)
    hosts = sample_host_galaxies(
        universe, catalogue, selection, n, rng, redshift=redshift
    )
    return assemble_injection_set(
        hosts, bbh, universe.cosmology, detectable, rng, t_start, t_end
    )
