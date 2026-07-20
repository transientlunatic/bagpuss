"""Configuration for the MDC generation pipeline.

A single :class:`MDCConfig`, loaded from one YAML file, is the sole source
of truth for a Mock Data Challenge run: physical parameters (cosmology,
luminosity function, selection, BBH population), sharding (how many sky
tiles and injection-draw shards to split the run into), and reproducibility
(the master RNG seed). Every pipeline stage in :mod:`bagpuss.mdc.pipeline`
and every condor job rendered by :mod:`bagpuss.mdc.condor` takes the same
config, so a run is fully reproducible from the config file plus the
bagpuss version that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

__all__: list[str] = ["MDCConfig", "load_config"]


@dataclass(frozen=True)
class MDCConfig:
    """Full configuration for one MDC generation run.

    All fields have defaults matching bagpuss's existing small-scale demo
    parameters (see ``faketc/scripts/make_injections.py``); a run-specific
    YAML file only needs to override the fields that differ.

    Parameters
    ----------
    cosmology : str
        Name of a built-in astropy cosmology (e.g. ``"Planck18"``).
    z_max : float
        Maximum redshift of the simulated volume (``PointProcess``).
    phi_star, m_star, alpha, m_min, m_max, m_sun : float
        Schechter luminosity function parameters. ``phi_star`` must be in
        physical units (Mpc⁻³ mag⁻¹) for :func:`bagpuss.catalogue.build_catalogue`
        to give a physically meaningful galaxy count.
    m_lim : float
        Limiting apparent magnitude of the magnitude-limited survey used
        as the selection function.
    mass_alpha, mass_beta_q, mass_m_min, mass_m_max : float
        :class:`~bagpuss.population.PowerLawPlusPeakMassDistribution`
        hyperparameters.
    mass_lambda_peak, mass_mu_m, mass_sigma_m, mass_delta_m : float
        More :class:`~bagpuss.population.PowerLawPlusPeakMassDistribution`
        hyperparameters.
    spin_a_max : float
        Maximum spin magnitude for the isotropic spin distribution.
    d_max : float or None
        Distance-threshold detectability cutoff in Mpc. ``None`` disables
        the detectability filter (every drawn event is kept).
    t_start, t_end : float or None
        Observation window in GPS seconds. ``None`` uses bagpuss's O3
        defaults.
    n_ra_tiles, n_dec_tiles : int
        Number of equal-area sky tiles along right ascension and
        ``sin(dec)`` respectively. Stage-1 (catalogue) generation runs one
        condor job per ``n_ra_tiles * n_dec_tiles`` tile.
    n_injection_shards : int
        Number of independent injection-generation shards (stage-3).
    n_draw_per_shard : int
        Number of BBH events drawn (before the detectability filter) in
        each injection shard.
    master_seed : int
        Root seed. Every shard derives an independent, deterministic RNG
        stream from this seed plus its own stage/shard id
        (:func:`bagpuss.mdc.pipeline.shard_rng`).
    store : str
        Path or URL of the zarr store the run writes to.
    """

    cosmology: str = "Planck18"

    z_max: float = 1.0

    phi_star: float = 1.61e-2
    m_star: float = -19.66
    alpha: float = -1.16
    m_min: float = -25.0
    m_max: float = -14.0
    m_sun: float = 4.83

    m_lim: float = 19.5

    mass_alpha: float = 3.5
    mass_beta_q: float = 1.4
    mass_m_min: float = 5.0
    mass_m_max: float = 87.0
    mass_lambda_peak: float = 0.03
    mass_mu_m: float = 34.0
    mass_sigma_m: float = 3.6
    mass_delta_m: float = 4.8

    spin_a_max: float = 1.0

    d_max: float | None = None

    t_start: float | None = None
    t_end: float | None = None

    n_ra_tiles: int = 1
    n_dec_tiles: int = 1
    n_injection_shards: int = 1
    n_draw_per_shard: int = 1000

    master_seed: int = 0
    store: str = "mdc.zarr"

    def __post_init__(self) -> None:
        if self.n_ra_tiles < 1 or self.n_dec_tiles < 1:
            raise ValueError(
                "n_ra_tiles and n_dec_tiles must both be >= 1, got "
                f"n_ra_tiles={self.n_ra_tiles!r}, n_dec_tiles={self.n_dec_tiles!r}"
            )
        if self.n_injection_shards < 1:
            raise ValueError(
                f"n_injection_shards must be >= 1, got {self.n_injection_shards!r}"
            )

    @property
    def n_tiles(self) -> int:
        """Return the total number of sky tiles (``n_ra_tiles * n_dec_tiles``)."""
        return self.n_ra_tiles * self.n_dec_tiles


def load_config(path: str | Path) -> MDCConfig:
    """Load an :class:`MDCConfig` from a YAML file.

    The file's top-level keys are applied over :class:`MDCConfig`'s
    defaults; any key that isn't a recognised field raises an error, to
    catch typos early rather than silently ignoring them.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a YAML config file.

    Returns
    -------
    MDCConfig

    Raises
    ------
    ValueError
        If the file contains a key that isn't a field of :class:`MDCConfig`.
    """
    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    known = {f.name for f in fields(MDCConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown MDCConfig key(s) in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    return replace(MDCConfig(), **raw)
