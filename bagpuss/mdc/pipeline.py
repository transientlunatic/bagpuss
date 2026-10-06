r"""MDC pipeline stages: one function per HTCondor DAG node.

Each function here does exactly one unit of sharded work and is safe to
run as an independent condor job: it opens (or creates) the shared zarr
store, does its own draw with its own deterministic RNG stream
(:func:`shard_rng`), writes its own self-contained group, and returns.
Nothing here submits jobs or writes DAG/submit files -- see
:mod:`bagpuss.mdc.condor` for that.

DAG shape (see module docstring of :mod:`bagpuss.mdc.condor` for the
rendered version)::

    generate_catalogue_tile(tile_id) x n_tiles \
                                                  > consolidate_catalogue
    generate_catalogue_tile(tile_id) x n_tiles /

    consolidate_catalogue
        -> generate_injection_shard(shard_id) x n_injection_shards
            -> assemble_injections
                -> detect_injection_shard(shard_id) x n_injection_shards
                    -> assemble_detections
                        -> package_manifest
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
import zarr
from astropy.cosmology import FLRW

import bagpuss
import bagpuss.glade_export as glade_export
from bagpuss.catalogue import GalaxyCatalogue, MagnitudeLimitedSurvey, build_catalogue
from bagpuss.injection import (
    GPS_O3_END,
    GPS_O3_START,
    Detectable,
    DistanceThreshold,
    InjectionSet,
    build_injection_set,
)
from bagpuss.mdc.config import MDCConfig
from bagpuss.mdc.detection import DetectionBackend, MinkeBackend
from bagpuss.mdc.localization import BayestarLocalizer, Localizer
from bagpuss.population import (
    SECONDS_PER_YEAR,
    ConstantMergerRate,
    IsotropicSpinDistribution,
    MergerRate,
    PopulationModel,
    PowerLawPlusPeakMassDistribution,
    expected_n_mergers,
)
from bagpuss.universe import PointProcess, SkyPatch, Universe

__all__: list[str] = [
    "MDCValidationError",
    "tile_bounds",
    "shard_time_bounds",
    "shard_rng",
    "build_universe",
    "build_tile_universe",
    "build_selection",
    "build_population",
    "build_rate",
    "build_detectable",
    "expected_injection_count",
    "generate_catalogue_tile",
    "consolidate_catalogue",
    "load_consolidated_catalogue",
    "generate_injection_shard",
    "assemble_injections",
    "detect_injection_shard",
    "assemble_detections",
    "package_manifest",
    "export_glade_catalogue",
]

#: SeedSequence tag distinguishing the catalogue-tile RNG stream from the
#: injection-shard stream, so the two never collide even if a run happened
#: to use the same shard id for both.
_SEED_TAGS: dict[str, int] = {"tile": 1, "injection": 2, "duty": 3, "skymap": 4}


class MDCValidationError(Exception):
    """Raised when a consolidation/assembly stage's sanity checks fail.

    Carries the full validation report (as returned by
    :func:`consolidate_catalogue`/:func:`assemble_injections`) in
    :attr:`report`, so a POST script or caller can log exactly what failed.
    """

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__(f"MDC validation failed: {report['issues']}")
        self.report = report


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------


def tile_bounds(
    config: MDCConfig, tile_id: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the ``(ra_range, sin_dec_range)`` bounds of sky tile ``tile_id``.

    Tiles are laid out on a regular ``n_ra_tiles x n_dec_tiles`` grid in
    ``(ra, sin(dec))`` space (equal-area, per :class:`~bagpuss.universe.SkyPatch`),
    with ``tile_id = ra_index * n_dec_tiles + dec_index``.

    Parameters
    ----------
    config : MDCConfig
        Run configuration, providing ``n_ra_tiles``/``n_dec_tiles``.
    tile_id : int
        Flat tile index in ``[0, config.n_tiles)``.

    Returns
    -------
    ra_range : tuple[float, float]
    sin_dec_range : tuple[float, float]
    """
    if not (0 <= tile_id < config.n_tiles):
        raise ValueError(f"tile_id must be in [0, {config.n_tiles}), got {tile_id!r}")

    ra_index, dec_index = divmod(tile_id, config.n_dec_tiles)
    ra_edges = np.linspace(0.0, 2.0 * np.pi, config.n_ra_tiles + 1)
    u_edges = np.linspace(-1.0, 1.0, config.n_dec_tiles + 1)
    ra_range = (float(ra_edges[ra_index]), float(ra_edges[ra_index + 1]))
    sin_dec_range = (float(u_edges[dec_index]), float(u_edges[dec_index + 1]))
    return ra_range, sin_dec_range


def shard_time_bounds(config: MDCConfig, shard_id: int) -> tuple[float, float]:
    """Return the ``(t_start, t_end)`` GPS bounds of injection shard ``shard_id``.

    The run's full observation window (``config.t_start``/``config.t_end``,
    defaulting to bagpuss's O3 window when unset) is split into
    ``n_injection_shards`` equal time slices, so that each shard's Poisson
    draw covers a disjoint fraction of the window -- the injection-side
    analogue of how :func:`tile_bounds` splits the sky for the catalogue.
    A shard's expected event count therefore scales with
    ``1 / n_injection_shards`` for a fixed total observation time, the same
    way a catalogue tile's expected count scales with its
    :attr:`~bagpuss.universe.SkyPatch.sky_fraction`.

    Parameters
    ----------
    config : MDCConfig
        Run configuration, providing ``t_start``/``t_end``/``n_injection_shards``.
    shard_id : int
        Flat shard index in ``[0, config.n_injection_shards)``.

    Returns
    -------
    t_start : float
    t_end : float
    """
    if not (0 <= shard_id < config.n_injection_shards):
        raise ValueError(
            f"shard_id must be in [0, {config.n_injection_shards}), got {shard_id!r}"
        )

    t_start = config.t_start if config.t_start is not None else GPS_O3_START
    t_end = config.t_end if config.t_end is not None else GPS_O3_END
    edges = np.linspace(t_start, t_end, config.n_injection_shards + 1)
    return float(edges[shard_id]), float(edges[shard_id + 1])


def shard_rng(config: MDCConfig, kind: str, shard_id: int) -> np.random.Generator:
    """Return the deterministic, independent RNG stream for one shard.

    Derived from ``(config.master_seed, tag, shard_id)`` via
    :class:`numpy.random.SeedSequence`, so re-running the same shard id
    reproduces exactly the same draw, and different shards (even of
    different kinds) never share a stream.

    Parameters
    ----------
    config : MDCConfig
        Run configuration, providing ``master_seed``.
    kind : {"tile", "injection"}
        Which pipeline stage the shard belongs to.
    shard_id : int
        The shard's index within its stage.

    Returns
    -------
    numpy.random.Generator
    """
    tag = _SEED_TAGS[kind]
    seed_sequence = np.random.SeedSequence([config.master_seed, tag, shard_id])
    return np.random.default_rng(seed_sequence)


# ---------------------------------------------------------------------------
# Model builders (config -> bagpuss model objects)
# ---------------------------------------------------------------------------


def _load_cosmology(name: str) -> FLRW:
    """Return the astropy cosmology object named by *name*."""
    import astropy.cosmology as ac

    cosmology = getattr(ac, name, None)
    if cosmology is None:
        raise ValueError(f"Unknown cosmology {name!r}")
    return cosmology


def build_universe(config: MDCConfig) -> Universe:
    """Build the full-sky :class:`~bagpuss.universe.Universe` for *config*."""
    from bagpuss.luminosity import SchechterLuminosityModel

    cosmology = _load_cosmology(config.cosmology)
    luminosity = SchechterLuminosityModel(
        phi_star=config.phi_star,
        m_star=config.m_star,
        alpha=config.alpha,
        m_min=config.m_min,
        m_max=config.m_max,
        m_sun=config.m_sun,
    )
    return Universe(
        cosmology=cosmology,
        structure=PointProcess(z_max=config.z_max),
        luminosity=luminosity,
    )


def build_tile_universe(config: MDCConfig, tile_id: int) -> Universe:
    """Build the :class:`~bagpuss.universe.Universe` restricted to one sky tile.

    Same cosmology/luminosity model as :func:`build_universe`, but with the
    structure model wrapped in a :class:`~bagpuss.universe.SkyPatch` for
    ``tile_id`` (see :func:`tile_bounds`).
    """
    universe = build_universe(config)
    ra_range, sin_dec_range = tile_bounds(config, tile_id)
    tile_structure = SkyPatch(
        universe.structure, ra_range=ra_range, sin_dec_range=sin_dec_range
    )
    return Universe(
        cosmology=universe.cosmology,
        structure=tile_structure,
        luminosity=universe.luminosity,
    )


def build_selection(config: MDCConfig) -> MagnitudeLimitedSurvey:
    """Build the magnitude-limited selection function for *config*."""
    return MagnitudeLimitedSurvey(m_lim=config.m_lim, m_sun=config.m_sun)


def build_population(config: MDCConfig) -> PopulationModel:
    """Build the BBH population model for *config*."""
    mass = PowerLawPlusPeakMassDistribution(
        alpha=config.mass_alpha,
        beta_q=config.mass_beta_q,
        m_min=config.mass_m_min,
        m_max=config.mass_m_max,
        lambda_peak=config.mass_lambda_peak,
        mu_m=config.mass_mu_m,
        sigma_m=config.mass_sigma_m,
        delta_m=config.mass_delta_m,
    )
    spin = IsotropicSpinDistribution(a_max=config.spin_a_max)
    return PopulationModel(mass=mass, spin=spin)


def build_rate(config: MDCConfig) -> MergerRate:
    """Build the BBH merger-rate density model for *config*."""
    return ConstantMergerRate(rate_density_value=config.rate_density)


def build_detectable(config: MDCConfig) -> Detectable | None:
    """Build the detectability filter for *config*, or ``None`` if disabled."""
    if config.d_max is None:
        return None
    return DistanceThreshold(d_max=config.d_max)


def expected_injection_count(
    config: MDCConfig,
    d_max: float | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
) -> dict[str, float | None]:
    r"""Return the expected injection count implied by *config*, analytically.

    A pure closed-form calculation via :func:`~bagpuss.population.expected_n_mergers`
    -- no galaxies or events are drawn, no zarr store is touched. Useful for
    sanity-checking a proposed ``rate_density``/observation window/detector
    sensitivity *before* submitting an expensive DAG, the same way one might
    ask "what does this cosmology predict?" without running the simulation.

    ``d_max``/``t_start``/``t_end`` override the corresponding config fields
    for "what if" questions (e.g. a different observation window or a
    different detectability threshold) without editing the YAML file.

    Parameters
    ----------
    config : MDCConfig
        Run configuration providing ``z_max``, ``cosmology``, ``rate_density``,
        and (unless overridden below) ``t_start``/``t_end``/``d_max``.
    d_max : float or None, optional
        Distance-threshold detectability cutoff in Mpc, overriding
        ``config.d_max``. If neither is set, ``n_detected`` is ``None``.
    t_start, t_end : float or None, optional
        Observation window in GPS seconds, overriding ``config.t_start``/
        ``config.t_end`` (which in turn default to bagpuss's O3 window).

    Returns
    -------
    dict
        ``t_obs_years`` -- observation window length in years.
        ``n_total`` -- expected mergers in the full simulated volume
        (``config.z_max``), before any detectability cut.
        ``d_max`` -- the distance threshold actually used, or ``None``.
        ``n_detected`` -- expected mergers within ``d_max``, restricting the
        redshift integral to ``min(config.z_max, z(d_max))``; ``None`` if no
        distance threshold is configured. This mirrors
        :class:`~bagpuss.injection.DistanceThreshold`'s hard cut exactly, so
        it is only as realistic as that placeholder detectability model --
        see ``docs/mdc.rst``.
    """
    from astropy import units as u
    from astropy.cosmology import z_at_value

    universe = build_universe(config)
    rate = build_rate(config)

    resolved_t_start = (
        t_start
        if t_start is not None
        else (config.t_start if config.t_start is not None else GPS_O3_START)
    )
    resolved_t_end = (
        t_end
        if t_end is not None
        else (config.t_end if config.t_end is not None else GPS_O3_END)
    )
    resolved_d_max = d_max if d_max is not None else config.d_max

    n_total = expected_n_mergers(
        universe.structure, universe.cosmology, rate, resolved_t_start, resolved_t_end
    )

    n_detected: float | None = None
    if resolved_d_max is not None:
        z_at_dmax = float(
            z_at_value(universe.cosmology.luminosity_distance, resolved_d_max * u.Mpc)  # pyright: ignore[reportAttributeAccessIssue]
        )
        detectable_structure = PointProcess(z_max=min(config.z_max, z_at_dmax))
        n_detected = expected_n_mergers(
            detectable_structure,
            universe.cosmology,
            rate,
            resolved_t_start,
            resolved_t_end,
        )

    return {
        "t_obs_years": (resolved_t_end - resolved_t_start) / SECONDS_PER_YEAR,
        "n_total": n_total,
        "d_max": resolved_d_max,
        "n_detected": n_detected,
    }


def _open_store(config: MDCConfig, mode: Literal["r", "a"]) -> zarr.Group:
    """Open the run's zarr store in the given mode.

    Always bypasses consolidated metadata: our own read/write access always
    goes through exact, known group paths (``require_group(name)``,
    ``attrs["_index"]``) rather than listing, so consolidated metadata buys
    us nothing here -- and *using* it while also writing new groups after
    :func:`zarr.consolidate_metadata` has already run (as
    ``generate_injection_shard`` does, post-``consolidate_catalogue``) is
    actively unsafe: zarr resolves lookups against the frozen consolidated
    snapshot, so a group created after consolidation is invisible to one
    call and collides with itself on the next. Consolidated metadata is
    written for external distribution/consumer performance only (see
    :func:`package_manifest`), never read back by this module.
    """
    return zarr.open_group(store=config.store, mode=mode, use_consolidated=False)


def _provenance_attrs(config: MDCConfig, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the standard provenance block written into every shard's attrs."""
    return {
        "bagpuss_version": bagpuss.__version__,
        "config": asdict(config),
        **extra,
    }


# ---------------------------------------------------------------------------
# Stage 1: catalogue tiles
# ---------------------------------------------------------------------------


def generate_catalogue_tile(config: MDCConfig, tile_id: int) -> GalaxyCatalogue:
    """Generate and write one galaxy-catalogue sky tile (a Tier-1 DAG node).

    Draws a Poisson realisation of the observable galaxy population
    restricted to sky tile ``tile_id`` (see :func:`build_tile_universe`)
    and writes it to ``<store>/catalogue/tile_%04d`` as an independent,
    self-contained zarr group.

    Parameters
    ----------
    config : MDCConfig
        Run configuration.
    tile_id : int
        Flat tile index in ``[0, config.n_tiles)``.

    Returns
    -------
    GalaxyCatalogue
        The tile's catalogue (also written to the store as a side effect).
    """
    universe = build_tile_universe(config, tile_id)
    selection = build_selection(config)
    rng = shard_rng(config, "tile", tile_id)
    catalogue = build_catalogue(universe, selection, rng)

    ra_range, sin_dec_range = tile_bounds(config, tile_id)
    root = _open_store(config, mode="a")
    tile_group = root.require_group("catalogue").require_group(f"tile_{tile_id:04d}")
    catalogue.to_zarr(tile_group)
    tile_group.attrs.update(
        _provenance_attrs(
            config,
            tile_id=tile_id,
            ra_range=list(ra_range),
            sin_dec_range=list(sin_dec_range),
            n_galaxies=len(catalogue),
        )
    )
    return catalogue


# ---------------------------------------------------------------------------
# Stage 2: consolidate catalogue tiles
# ---------------------------------------------------------------------------


def consolidate_catalogue(
    config: MDCConfig, raise_on_failure: bool = True
) -> dict[str, Any]:
    """Validate and consolidate all catalogue tiles (the Tier-2 DAG node).

    Reads every ``<store>/catalogue/tile_%04d`` group written by
    :func:`generate_catalogue_tile`, records their row counts and
    cumulative offsets in ``<store>/catalogue.attrs["_index"]`` (this is
    what turns N independent tile shards into one logical catalogue for
    :func:`load_consolidated_catalogue` and the global ``host_galaxy_index``
    used by injections), runs sanity checks, and calls
    :func:`zarr.consolidate_metadata` on the whole store.

    Sanity checks:

    * every expected tile (``0 <= tile_id < config.n_tiles``) is present
    * no tile's row count is a >5-sigma Poisson outlier relative to the
      mean tile count (catches a corrupted or mis-seeded shard)
    * every galaxy's apparent magnitude satisfies the survey's ``m_lim``
    * every galaxy's redshift lies in ``[0, config.z_max]``
    * no NaN/Inf values in any field

    Parameters
    ----------
    config : MDCConfig
        Run configuration.
    raise_on_failure : bool, optional
        If ``True`` (default), raise :class:`MDCValidationError` when any
        check fails. If ``False``, return the report with ``ok=False``
        instead -- useful for callers that want to inspect the report
        themselves.

    Returns
    -------
    dict
        Report with keys ``ok``, ``n_tiles``, ``total_galaxies``, ``issues``.

    Raises
    ------
    MDCValidationError
        If ``raise_on_failure`` and any sanity check fails.
    """
    root = _open_store(config, mode="a")
    catalogue_group = root.require_group("catalogue")
    present = set(catalogue_group.group_keys())

    issues: list[str] = []
    index: list[dict[str, Any]] = []
    offset = 0
    magnitude_violations = 0
    redshift_violations = 0
    non_finite = 0

    for tile_id in range(config.n_tiles):
        name = f"tile_{tile_id:04d}"
        if name not in present:
            issues.append(f"missing tile: {name}")
            continue

        tile_group = catalogue_group.require_group(name)
        catalogue = GalaxyCatalogue.from_zarr(tile_group)
        n = len(catalogue)
        index.append({"tile_id": tile_id, "n_galaxies": n, "offset": offset})
        offset += n

        if n > 0:
            magnitude_violations += int(
                np.sum(catalogue.apparent_magnitudes > config.m_lim + 1e-9)
            )
            redshift_violations += int(
                np.sum(
                    (catalogue.redshifts < 0.0)
                    | (catalogue.redshifts > config.z_max + 1e-9)
                )
            )
            for field in (
                "redshifts",
                "luminosities",
                "apparent_magnitudes",
                "ra",
                "dec",
            ):
                non_finite += int(np.sum(~np.isfinite(getattr(catalogue, field))))

    total_galaxies = offset
    if index:
        counts = np.array([row["n_galaxies"] for row in index], dtype=float)
        mean_count = float(counts.mean())
        # 5-sigma Poisson outlier check; skip on empty/near-empty tiles where
        # sqrt(mean) is too small to be meaningful.
        if mean_count > 20.0:
            tolerance = 5.0 * np.sqrt(mean_count)
            outliers = np.where(np.abs(counts - mean_count) > tolerance)[0]
            for i in outliers:
                issues.append(
                    f"tile {index[int(i)]['tile_id']} galaxy count "
                    f"{index[int(i)]['n_galaxies']} is a >5-sigma outlier "
                    f"(mean={mean_count:.1f})"
                )

    if magnitude_violations:
        issues.append(f"{magnitude_violations} galaxies exceed m_lim={config.m_lim}")
    if redshift_violations:
        issues.append(f"{redshift_violations} galaxies outside [0, {config.z_max}]")
    if non_finite:
        issues.append(f"{non_finite} non-finite values across catalogue fields")

    catalogue_group.attrs.update({"_index": index, "_total_galaxies": total_galaxies})

    zarr.consolidate_metadata(root.store)

    report = {
        "ok": not issues,
        "n_tiles": len(index),
        "total_galaxies": total_galaxies,
        "issues": issues,
    }
    if not report["ok"] and raise_on_failure:
        raise MDCValidationError(report)
    return report


def load_consolidated_catalogue(config: MDCConfig) -> GalaxyCatalogue:
    """Load the full catalogue by concatenating all tile shards in order.

    Requires :func:`consolidate_catalogue` to have already run (its
    ``_index`` attribute gives the tile order). The result is small: only
    the *observed* (selection-passing) subset of galaxies is stored, so
    even at full survey scale this comfortably fits in memory.

    Parameters
    ----------
    config : MDCConfig
        Run configuration.

    Returns
    -------
    GalaxyCatalogue
        The concatenated catalogue, in tile-id order. Row ``i``'s global
        index (as referenced by ``InjectionSet.host_galaxy_index``) is its
        position in this concatenation.
    """
    root = _open_store(config, mode="r")
    catalogue_group = root.require_group("catalogue")
    index = cast(list[dict[str, Any]], catalogue_group.attrs["_index"])

    parts = [
        GalaxyCatalogue.from_zarr(
            catalogue_group.require_group(f"tile_{row['tile_id']:04d}")
        )
        for row in index
    ]
    if not parts:
        empty = np.array([], dtype=float)
        return GalaxyCatalogue(
            redshifts=empty,
            luminosities=empty,
            apparent_magnitudes=empty,
            ra=empty,
            dec=empty,
        )
    return GalaxyCatalogue(
        redshifts=np.concatenate([p.redshifts for p in parts]),
        luminosities=np.concatenate([p.luminosities for p in parts]),
        apparent_magnitudes=np.concatenate([p.apparent_magnitudes for p in parts]),
        ra=np.concatenate([p.ra for p in parts]),
        dec=np.concatenate([p.dec for p in parts]),
    )


# ---------------------------------------------------------------------------
# Stage 3: injection shards
# ---------------------------------------------------------------------------


def generate_injection_shard(config: MDCConfig, shard_id: int) -> InjectionSet:
    """Generate and write one injection shard (a Tier-3 DAG node).

    Poisson-realises the shard's event count from ``config.rate_density``
    over the shard's slice of the observation window (see
    :func:`shard_time_bounds`) via :func:`~bagpuss.injection.build_injection_set`
    -- the injection-side analogue of how :func:`generate_catalogue_tile`
    Poisson-realises a tile's galaxy count from the luminosity function
    rather than a caller-chosen target. Host galaxies are assigned against
    the *consolidated* catalogue (so ``host_galaxy_index`` is a global
    index, per :func:`load_consolidated_catalogue`), the configured
    detectability filter is applied, and the surviving events are written
    to ``<store>/injections/shard_%04d``.

    Requires :func:`consolidate_catalogue` to have already run.

    Parameters
    ----------
    config : MDCConfig
        Run configuration.
    shard_id : int
        Shard index in ``[0, config.n_injection_shards)``.

    Returns
    -------
    InjectionSet
        The shard's injections (also written to the store as a side effect).
    """
    universe = build_universe(config)
    selection = build_selection(config)
    population = build_population(config)
    rate = build_rate(config)
    detectable = build_detectable(config)
    catalogue = load_consolidated_catalogue(config)
    rng = shard_rng(config, "injection", shard_id)
    t_start, t_end = shard_time_bounds(config, shard_id)

    injections = build_injection_set(
        universe=universe,
        catalogue=catalogue,
        selection=selection,
        population=population,
        rate=rate,
        detectable=detectable,
        rng=rng,
        t_start=t_start,
        t_end=t_end,
        host_luminosity_weight=config.host_luminosity_weight,
    )

    root = _open_store(config, mode="a")
    shard_group = root.require_group("injections").require_group(
        f"shard_{shard_id:04d}"
    )
    injections.to_zarr(shard_group)
    shard_group.attrs.update(
        _provenance_attrs(config, shard_id=shard_id, n_injections=len(injections))
    )
    return injections


# ---------------------------------------------------------------------------
# Stage 4: assemble injection shards
# ---------------------------------------------------------------------------


def assemble_injections(
    config: MDCConfig, raise_on_failure: bool = True
) -> dict[str, Any]:
    """Validate and consolidate all injection shards (the Tier-4 DAG node).

    Mirrors :func:`consolidate_catalogue`: records each shard's row count
    and offset in ``<store>/injections.attrs["_index"]``, runs sanity
    checks, and calls :func:`zarr.consolidate_metadata`.

    Sanity checks:

    * every expected shard (``0 <= shard_id < config.n_injection_shards``)
      is present
    * no shard's surviving-event count is a >5-sigma binomial outlier
      relative to the mean (catches a corrupted or mis-seeded shard)
    * every ``host_galaxy_index`` is ``-1`` or a valid index into the
      consolidated catalogue
    * ``m1_source >= m2_source > 0`` for every event
    * no NaN/Inf values in any field

    Parameters
    ----------
    config : MDCConfig
        Run configuration.
    raise_on_failure : bool, optional
        If ``True`` (default), raise :class:`MDCValidationError` when any
        check fails.

    Returns
    -------
    dict
        Report with keys ``ok``, ``n_shards``, ``total_injections``, ``issues``.

    Raises
    ------
    MDCValidationError
        If ``raise_on_failure`` and any sanity check fails.
    """
    root = _open_store(config, mode="a")
    injections_group = root.require_group("injections")
    present = set(injections_group.group_keys())
    n_catalogue = int(
        cast(int, root.require_group("catalogue").attrs["_total_galaxies"])
    )

    issues: list[str] = []
    index: list[dict[str, Any]] = []
    offset = 0
    bad_host_index = 0
    bad_masses = 0
    non_finite = 0

    for shard_id in range(config.n_injection_shards):
        name = f"shard_{shard_id:04d}"
        if name not in present:
            issues.append(f"missing injection shard: {name}")
            continue

        shard_group = injections_group.require_group(name)
        injections = InjectionSet.from_zarr(shard_group)
        n = len(injections)
        index.append({"shard_id": shard_id, "n_injections": n, "offset": offset})
        offset += n

        if n > 0:
            host_idx = injections.host_galaxy_index
            bad_host_index += int(np.sum((host_idx < -1) | (host_idx >= n_catalogue)))
            bad_masses += int(np.sum(injections.m1_source < injections.m2_source))
            bad_masses += int(np.sum(injections.m2_source <= 0.0))
            for field in (
                "m1_source",
                "m2_source",
                "redshift",
                "luminosity_distance",
                "ra",
                "dec",
            ):
                non_finite += int(np.sum(~np.isfinite(getattr(injections, field))))

    total_injections = offset
    if index:
        counts = np.array([row["n_injections"] for row in index], dtype=float)
        mean_count = float(counts.mean())
        if mean_count > 20.0:
            tolerance = 5.0 * np.sqrt(mean_count)
            outliers = np.where(np.abs(counts - mean_count) > tolerance)[0]
            for i in outliers:
                issues.append(
                    f"shard {index[int(i)]['shard_id']} injection count "
                    f"{index[int(i)]['n_injections']} is a >5-sigma outlier "
                    f"(mean={mean_count:.1f})"
                )

    if bad_host_index:
        issues.append(f"{bad_host_index} injections have an invalid host_galaxy_index")
    if bad_masses:
        issues.append(f"{bad_masses} injections violate m1_source >= m2_source > 0")
    if non_finite:
        issues.append(f"{non_finite} non-finite values across injection fields")

    injections_group.attrs.update(
        {"_index": index, "_total_injections": total_injections}
    )

    zarr.consolidate_metadata(root.store)

    report = {
        "ok": not issues,
        "n_shards": len(index),
        "total_injections": total_injections,
        "issues": issues,
    }
    if not report["ok"] and raise_on_failure:
        raise MDCValidationError(report)
    return report


# ---------------------------------------------------------------------------
# Stage 5b: detection (SNRs, frames, blueprints)
# ---------------------------------------------------------------------------


def _shard_name(shard_id: int) -> str:
    return f"shard_{shard_id:04d}"


def detect_injection_shard(
    config: MDCConfig,
    shard_id: int,
    backend: DetectionBackend | None = None,
    localizer: Localizer | None = None,
) -> dict[str, Any]:
    """Compute SNRs for one injection shard and realise its detections.

    Per-shard DAG node, run after :func:`assemble_injections`. For every
    injection in ``<store>/injections/shard_NNNN`` it works out which
    detectors are observing at the event's GPS time (a duty-cycle schedule
    drawn from ``config.master_seed``, identical in every shard), injects
    the signal and computes the network SNR; an event with no detector
    observing gets SNR 0. The SNR of **every** injection is written to
    ``<store>/detections/shard_NNNN`` (arrays ``network_snr``,
    ``detectable`` and ``active``, row-aligned with the injection shard),
    so the threshold can be changed later without recomputing. Frame
    files and an asimov blueprint file are then written only for events
    with ``network_snr >= config.snr_threshold``, under
    ``config.output_dir``.

    Parameters
    ----------
    config : MDCConfig
        Run configuration.
    shard_id : int
        Shard index in ``[0, config.n_injection_shards)``.
    backend : DetectionBackend or None, optional
        Waveform/frame engine; defaults to :class:`~bagpuss.mdc.detection.MinkeBackend`.
    localizer : Localizer or None, optional
        Skymap generator, used when ``config.write_skymaps``; defaults to
        :class:`~bagpuss.mdc.localization.BayestarLocalizer`. A skymap is
        written for every event with ``network_snr >= config.snr_threshold``
        to ``<output_dir>/skymaps/shard_NNNN/``.

    Returns
    -------
    dict
        Summary with ``shard_id``, ``n_injections`` and ``n_detectable``.
    """
    if not 0 <= shard_id < config.n_injection_shards:
        raise ValueError(
            f"shard_id must be in [0, {config.n_injection_shards}), got {shard_id!r}"
        )
    backend = backend if backend is not None else MinkeBackend()
    name = _shard_name(shard_id)
    root = _open_store(config, mode="a")
    injections = InjectionSet.from_zarr(
        root.require_group("injections").require_group(name)
    )
    n = len(injections)
    detector_names = list(config.detectors)

    network_snr = np.zeros(n, dtype=np.float64)
    active = np.zeros((n, len(detector_names)), dtype=bool)
    out_dir = Path(config.output_dir).resolve()
    blueprint_path = out_dir / "blueprints" / f"{name}.yaml"
    blueprint_path.parent.mkdir(parents=True, exist_ok=True)

    detectable_params: list[dict[str, Any]] = []
    detectable_frames: list[dict[str, Any]] = []
    detectable_snrs: list[float] = []
    detectable_active: list[dict[str, str]] = []
    detectable_skymaps: list[Path | None] = []
    survivors: list[int] = []
    has_skymap = np.zeros(n, dtype=bool)

    if n > 0:
        params = backend.injection_parameters(injections, config.f_ref)
        t_start = config.t_start if config.t_start is not None else GPS_O3_START
        t_end = config.t_end if config.t_end is not None else GPS_O3_END
        schedules = backend.duty_schedules(
            config, t_start, t_end, shard_rng(config, "duty", 0)
        )

        # Pass 1: SNR of every event. No framefile, so nothing is written.
        active_sets: list[dict[str, str]] = []
        for i, event in enumerate(params):
            observing = backend.active_detectors(
                schedules, config.detectors, event["gpstime"]
            )
            active_sets.append(observing)
            active[i] = [d in observing for d in detector_names]
            if observing:
                _, network_snr[i] = backend.inject(event, observing, config, None)

        survivors = list(map(int, np.flatnonzero(network_snr >= config.snr_threshold)))
        if config.write_frames:
            # Pass 2: frames for survivors only. Each shard works in its own
            # directory because minke drops a basename-keyed cache/ in the cwd.
            frames_dir = out_dir / "frames" / name
            frames_dir.mkdir(parents=True, exist_ok=True)
            work_dir = out_dir / "work" / name
            work_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.chdir(work_dir):
                for i in survivors:
                    frame_files, snr = backend.inject(
                        params[i],
                        active_sets[i],
                        config,
                        str(frames_dir / f"s{shard_id:04d}_event_{i:06d}"),
                    )
                    detectable_params.append(params[i])
                    detectable_frames.append(frame_files)
                    detectable_snrs.append(snr)
                    detectable_active.append(active_sets[i])
        else:
            for i in survivors:
                detectable_params.append(params[i])
                detectable_frames.append({})
                detectable_snrs.append(float(network_snr[i]))
                detectable_active.append(active_sets[i])

    if config.write_skymaps and detectable_params:
        localizer = localizer or BayestarLocalizer(
            waveform=config.skymap_waveform, f_low=config.f_ref
        )
        sky_dir = out_dir / "skymaps" / name
        for event, observing, i in zip(
            detectable_params, detectable_active, survivors, strict=True
        ):
            abbreviations = {backend.abbreviation(d): d for d in observing}
            seed = int(
                np.random.SeedSequence(
                    [config.master_seed, _SEED_TAGS["skymap"], shard_id, i]
                ).generate_state(1)[0]
            )
            path = localizer.localize(
                event,
                abbreviations,
                {a: config.detectors[d] for a, d in abbreviations.items()},
                seed,
                sky_dir / f"inj_{event['gpstime']:.3f}.fits",
            )
            detectable_skymaps.append(path)
            has_skymap[i] = path is not None
    else:
        detectable_skymaps = [None] * len(detectable_params)

    if detectable_params:
        backend.write_blueprints(
            detectable_params,
            blueprint_path,
            config,
            detectable_frames,
            detectable_snrs,
            detectable_active,
            detectable_skymaps,
        )
    else:
        blueprint_path.write_text("")

    detectable = network_snr >= config.snr_threshold
    group = root.require_group("detections").require_group(name)
    group.create_array("network_snr", data=network_snr, overwrite=True)
    group.create_array("detectable", data=detectable, overwrite=True)
    group.create_array("active", data=active, overwrite=True)
    group.create_array("has_skymap", data=has_skymap, overwrite=True)
    group.attrs.update(
        _provenance_attrs(
            config,
            shard_id=shard_id,
            n_injections=n,
            n_detectable=int(detectable.sum()),
            n_skymaps=int(has_skymap.sum()),
            snr_threshold=config.snr_threshold,
            detectors=detector_names,
        )
    )
    return {
        "shard_id": shard_id,
        "n_injections": n,
        "n_detectable": int(detectable.sum()),
        "n_skymaps": int(has_skymap.sum()),
    }


def assemble_detections(
    config: MDCConfig, raise_on_failure: bool = True
) -> dict[str, Any]:
    """Validate all detection shards and merge their blueprints.

    Mirrors :func:`assemble_injections`. Records per-shard counts in
    ``<store>/detections.attrs["_index"]``, checks that every shard is
    present, has one SNR per injection, has no non-finite SNRs, and that
    its blueprint file holds exactly one document per detectable event,
    then concatenates the per-shard blueprints into
    ``<output_dir>/blueprints.yaml``.

    Raises
    ------
    MDCValidationError
        If ``raise_on_failure`` and any check fails.
    """
    root = _open_store(config, mode="a")
    detections = root.require_group("detections")
    injections = root.require_group("injections")
    present = set(detections.group_keys())
    out_dir = Path(config.output_dir)

    issues: list[str] = []
    index: list[dict[str, Any]] = []
    merged: list[str] = []
    offset = 0

    for shard_id in range(config.n_injection_shards):
        name = _shard_name(shard_id)
        if name not in present:
            issues.append(f"missing detection shard: {name}")
            continue
        group = detections.require_group(name)
        snr = np.asarray(group["network_snr"])
        n_detectable = int(np.sum(np.asarray(group["detectable"])))
        n_injected = len(np.asarray(injections.require_group(name)["m1_source"]))
        index.append(
            {
                "shard_id": shard_id,
                "n_injections": len(snr),
                "n_detectable": n_detectable,
                "n_skymaps": int(cast(int, group.attrs.get("n_skymaps", 0))),
                "offset": offset,
            }
        )
        offset += len(snr)

        if len(snr) != n_injected:
            issues.append(f"{name} has {len(snr)} SNRs for {n_injected} injections")
        if not np.all(np.isfinite(snr)):
            issues.append(f"{name} has non-finite SNRs")

        text = ""
        blueprint_file = out_dir / "blueprints" / f"{name}.yaml"
        if blueprint_file.exists():
            text = blueprint_file.read_text()
        docs = [d for d in yaml.safe_load_all(text) if d]
        if len(docs) != n_detectable:
            issues.append(
                f"{name} has {len(docs)} blueprints for "
                f"{n_detectable} detectable events"
            )
        elif text:
            merged.append(text)
        elif n_detectable:
            issues.append(f"missing blueprint file: {blueprint_file}")

    total_detectable = sum(row["n_detectable"] for row in index)
    total_skymaps = sum(row["n_skymaps"] for row in index)
    detections.attrs.update(
        {
            "_index": index,
            "_total_injections": offset,
            "_total_detectable": total_detectable,
            "_total_skymaps": total_skymaps,
            "_snr_threshold": config.snr_threshold,
        }
    )
    (out_dir / "blueprints.yaml").write_text("---\n".join(merged))
    if not config.write_frames:
        # One generic analysis, applied to every event: the per-event
        # injection settings live in the event blueprints.
        (out_dir / "analysis-minke.yaml").write_text(
            yaml.safe_dump(
                {
                    "kind": "analysis",
                    "name": "minke-frames",
                    "pipeline": "minke",
                    "waveform": {"approximant": "IMRPhenomXPHM"},
                },
                sort_keys=False,
            )
        )

    report = {
        "ok": not issues,
        "n_shards": len(index),
        "total_injections": offset,
        "total_detectable": total_detectable,
        "total_skymaps": total_skymaps,
        "n_without_skymap": total_detectable - total_skymaps
        if config.write_skymaps
        else None,
        "issues": issues,
    }
    if not report["ok"] and raise_on_failure:
        raise MDCValidationError(report)
    return report


# ---------------------------------------------------------------------------
# Stage 5: package manifest
# ---------------------------------------------------------------------------


def package_manifest(config: MDCConfig) -> dict[str, Any]:
    """Write the run's top-level manifest/provenance (the Tier-5 DAG node).

    Records the full config, bagpuss version, and summary counts in the
    store's root ``.zattrs`` -- so the data is self-documenting for anyone
    who receives just the zarr store directory, without the generating
    repo's docs. Does not compute per-chunk checksums itself (delegate to
    a store-aware tool such as ``rclone check`` or the eventual
    distribution target's own integrity checking, which can do it far more
    efficiently than iterating chunks in Python).

    Parameters
    ----------
    config : MDCConfig
        Run configuration.

    Returns
    -------
    dict
        The manifest that was written to the store's root attrs.
    """
    root = _open_store(config, mode="a")
    # The detection stage is optional (it needs minke); a run that stops at
    # injections still gets a manifest, with the detection fields null.
    detections_attrs = (
        root["detections"].attrs if "detections" in root.group_keys() else {}
    )
    manifest = {
        "bagpuss_version": bagpuss.__version__,
        "config": asdict(config),
        "n_tiles": root["catalogue"].attrs.get("_index", []) and config.n_tiles,
        "total_galaxies": root["catalogue"].attrs.get("_total_galaxies"),
        "n_injection_shards": config.n_injection_shards,
        "total_injections": root["injections"].attrs.get("_total_injections"),
        "total_detectable": detections_attrs.get("_total_detectable"),
        "snr_threshold": detections_attrs.get("_snr_threshold"),
    }
    root.attrs.update({"mdc_manifest": manifest})
    zarr.consolidate_metadata(root.store)

    manifest_path = f"{config.store}/manifest.json"
    try:
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
    except OSError:
        # config.store may be a remote URL rather than a local path; the
        # manifest already lives in the store's root attrs either way.
        pass

    return manifest


# ---------------------------------------------------------------------------
# GLADE+-style catalogue export
# ---------------------------------------------------------------------------


def export_glade_catalogue(config: MDCConfig, out_dir: str | Path) -> dict[str, Any]:
    """Export the consolidated catalogue as a GLADE+-style flat-ASCII product.

    Streams tile by tile (one tile's worth of galaxies in memory at a
    time), in the exact same order :func:`load_consolidated_catalogue`
    concatenates them in -- so row ``i`` of the exported catalogue is
    exactly the galaxy any :class:`~bagpuss.injection.InjectionSet`
    generated against this store's ``host_galaxy_index == i`` refers to.
    See :mod:`bagpuss.glade_export` for the column schema and disclaimer
    written into the companion README.

    .. warning::

       This correspondence only holds for injections generated *before*
       any subsequent re-run of :func:`generate_catalogue_tile`/
       :func:`consolidate_catalogue` against this store -- reconsolidating
       after the fact silently invalidates any existing injection set's
       ``host_galaxy_index`` values relative to a freshly exported
       catalogue.

    Parameters
    ----------
    config : MDCConfig
        Run configuration. Must already have been consolidated (see
        :func:`consolidate_catalogue`).
    out_dir : str or pathlib.Path
        Directory to write ``catalogue.dat``, ``completeness.dat``, and
        ``README.txt`` into. Created if it doesn't exist.

    Returns
    -------
    dict
        Provenance summary: ``n_galaxies``, ``out_dir``, ``config``.

    Raises
    ------
    MDCValidationError
        If the store hasn't been consolidated yet (no ``_index``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = _open_store(config, mode="r")
    catalogue_group = root.require_group("catalogue")
    index = catalogue_group.attrs.get("_index")
    if not index:
        raise MDCValidationError(
            {
                "ok": False,
                "issues": [
                    "catalogue has not been consolidated -- run "
                    "consolidate_catalogue() first"
                ],
            }
        )
    index = cast(list[dict[str, Any]], index)

    universe = build_universe(config)
    selection = build_selection(config)

    catalogue_path = out_dir / "catalogue.dat"
    n_galaxies = 0
    with open(catalogue_path, "w") as fileobj:
        for row in index:
            tile = GalaxyCatalogue.from_zarr(
                catalogue_group.require_group(f"tile_{row['tile_id']:04d}")
            )
            if len(tile) == 0:
                continue
            d_l_mpc = (
                universe.cosmology.luminosity_distance(tile.redshifts)  # pyright: ignore[reportAttributeAccessIssue]
                .to("Mpc")
                .value
            )
            absolute_magnitudes = glade_export.compute_absolute_magnitudes(
                tile.luminosities, config.m_sun
            )
            ids = np.arange(len(tile)) + n_galaxies
            glade_export.write_glade_rows(
                fileobj,
                ids=ids,
                ra_rad=tile.ra,
                dec_rad=tile.dec,
                redshifts=tile.redshifts,
                luminosity_distances_mpc=np.asarray(d_l_mpc),
                apparent_magnitudes=tile.apparent_magnitudes,
                absolute_magnitudes=absolute_magnitudes,
            )
            n_galaxies += len(tile)

    glade_export.write_completeness_curve(
        selection,
        universe.luminosity,
        universe.cosmology,
        config.z_max,
        out_dir / "completeness.dat",
    )
    glade_export.write_readme(
        out_dir / "README.txt",
        m_sun=config.m_sun,
        cosmology_name=config.cosmology,
        bagpuss_version=bagpuss.__version__,
        n_galaxies=n_galaxies,
        z_max=config.z_max,
        catalogue_filename="catalogue.dat",
        completeness_filename="completeness.dat",
    )

    return {
        "n_galaxies": n_galaxies,
        "out_dir": str(out_dir),
        "config": asdict(config),
    }
