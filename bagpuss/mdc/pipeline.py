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
                -> package_manifest
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Literal, cast

import numpy as np
import zarr
from astropy.cosmology import FLRW

import bagpuss
from bagpuss.catalogue import GalaxyCatalogue, MagnitudeLimitedSurvey, build_catalogue
from bagpuss.injection import (
    Detectable,
    DistanceThreshold,
    InjectionSet,
    create_injection_set,
)
from bagpuss.mdc.config import MDCConfig
from bagpuss.population import (
    IsotropicSpinDistribution,
    PopulationModel,
    PowerLawPlusPeakMassDistribution,
)
from bagpuss.universe import PointProcess, SkyPatch, Universe

__all__: list[str] = [
    "MDCValidationError",
    "tile_bounds",
    "shard_rng",
    "build_universe",
    "build_tile_universe",
    "build_selection",
    "build_population",
    "build_detectable",
    "generate_catalogue_tile",
    "consolidate_catalogue",
    "load_consolidated_catalogue",
    "generate_injection_shard",
    "assemble_injections",
    "package_manifest",
]

#: SeedSequence tag distinguishing the catalogue-tile RNG stream from the
#: injection-shard stream, so the two never collide even if a run happened
#: to use the same shard id for both.
_SEED_TAGS: dict[str, int] = {"tile": 1, "injection": 2}


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


def build_detectable(config: MDCConfig) -> Detectable | None:
    """Build the detectability filter for *config*, or ``None`` if disabled."""
    if config.d_max is None:
        return None
    return DistanceThreshold(d_max=config.d_max)


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

    Draws ``config.n_draw_per_shard`` BBH events, assigns host galaxies
    against the *consolidated* catalogue (so ``host_galaxy_index`` is a
    global index, per :func:`load_consolidated_catalogue`), applies the
    configured detectability filter, and writes the surviving events to
    ``<store>/injections/shard_%04d``.

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
    detectable = build_detectable(config)
    catalogue = load_consolidated_catalogue(config)
    rng = shard_rng(config, "injection", shard_id)

    kwargs: dict[str, Any] = {}
    if config.t_start is not None:
        kwargs["t_start"] = config.t_start
    if config.t_end is not None:
        kwargs["t_end"] = config.t_end

    injections = create_injection_set(
        universe=universe,
        catalogue=catalogue,
        selection=selection,
        population=population,
        n_draw=config.n_draw_per_shard,
        detectable=detectable,
        rng=rng,
        **kwargs,
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
    manifest = {
        "bagpuss_version": bagpuss.__version__,
        "config": asdict(config),
        "n_tiles": root["catalogue"].attrs.get("_index", []) and config.n_tiles,
        "total_galaxies": root["catalogue"].attrs.get("_total_galaxies"),
        "n_injection_shards": config.n_injection_shards,
        "total_injections": root["injections"].attrs.get("_total_injections"),
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
