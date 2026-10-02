"""Configuration for the MDC generation pipeline.

A single :class:`MDCConfig`, loaded from one YAML file, is the sole source
of truth for a Mock Data Challenge run: physical parameters (cosmology,
luminosity function, selection, BBH population), the detector network and
frame-generation settings used to turn injections into detections,
sharding (how many sky tiles and injection-draw shards to split the run
into), and reproducibility (the master RNG seed). Every pipeline stage in
:mod:`bagpuss.mdc.pipeline` and every condor job rendered by
:mod:`bagpuss.mdc.condor` takes the same config, so a run is fully
reproducible from the config file plus the bagpuss version that produced
it. The ``detection`` fields (``detectors``, ``duty_cycles``,
``snr_threshold``, ...) are read by the detection stage
(:func:`bagpuss.mdc.pipeline.detect_injection_shard`), which delegates to
minke; the ``estimation`` fields are carried for the downstream PE
hand-off only.

On disk, a config file is a nested YAML mapping whose top-level sections
mirror the pipeline's stages (``galaxies``, ``population``,
``observation``, ``detection``, ``estimation``, ``sharding``, ``run``);
:func:`load_config` flattens it against :data:`_SCHEMA` into
:class:`MDCConfig`'s (flat) fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
        defaults. Split into ``n_injection_shards`` equal time slices by
        :func:`bagpuss.mdc.pipeline.shard_time_bounds`.
    detectors : dict[str, str]
        Detector network for stage 5b (frame generation + detectability),
        mapping each detector name to the PSD used to generate its noise
        (e.g. ``{"AdvancedLIGOHanford": "AdvancedLIGO_O4"}``). Used by the
        detection stage, which turns injection parameters into frame files
        via minke.
    duty_cycles : dict[str, float]
        Per-detector target long-run duty cycle (fraction of time locked),
        keyed the same as ``detectors``. Used by the detection stage (via
        ``minke.duty_cycle.generate_duty_cycle_schedule``) to
        decide which detectors are actually active for each event, rather
        than assuming every detector in ``detectors`` observes every event.
    duty_cycle_mean_lock_duration : float
        Mean duration in seconds of a detector's locked (observing)
        segment, shared across all detectors, used by the same duty-cycle
        renewal-process model as ``duty_cycles``.
    injection_channel : str
        Channel name signals are injected into, prefixed per-detector by
        the downstream frame-generation tooling (e.g. ``"H1:Injection"``).
    snr_threshold : float
        Network matched-filter SNR above which an injected event is
        considered detected; :func:`bagpuss.mdc.pipeline.detect_injection_shard`
        writes frames and blueprints only for events at or above it (the
        SNR of every event is stored regardless).
    write_frames : bool
        Whether the detection stage writes frame files for events above
        ``snr_threshold``. With ``False`` it only computes and stores SNRs
        (and writes data-less event blueprints), leaving frame generation
        to a later step (e.g. an asimov analysis).
    write_skymaps : bool
        Whether the detection stage writes a BAYESTAR skymap for every event
        at or above ``snr_threshold`` (see :mod:`bagpuss.mdc.localization`),
        recorded in its blueprint as ``localization file``. simple-pe takes
        its sky localisation from such a file.
    skymap_waveform : str
        Template approximant BAYESTAR uses when localising.
    f_ref : float
        Reference frequency in Hz at which spin components are defined,
        used by the downstream frame-generation/PE tooling.
    frame_duration : float
        Length in seconds of each generated frame file.
    frame_sample_rate : float
        Sample rate in Hz of each generated frame file.
    frame_pre_signal_padding : float
        Seconds of data included before an injected signal's GPS time when
        generating its frame (frame epoch = ``geocent_time -
        frame_pre_signal_padding``).
    chirp_mass_margin : float
        Fractional margin around the true chirp mass used to set the
        downstream PE chirp-mass prior range (stage 6), as
        ``[Mc / (1 + margin), Mc * (1 + margin)]``.
    trigger_eta_max : float or None
        If set, the starting "trigger" written into each event blueprint has
        its symmetric mass ratio capped at this value (chirp mass held
        fixed); the injected truth is kept under ``trigger truth``. simple-pe
        fails on triggers at the equal-mass limit (``eta = 0.25``), where
        its metric scaling steps past the physical bound, so ~0.249 avoids
        that. ``None`` leaves the trigger at the injected values.
    n_ra_tiles, n_dec_tiles : int
        Number of equal-area sky tiles along right ascension and
        ``sin(dec)`` respectively. Stage-1 (catalogue) generation runs one
        condor job per ``n_ra_tiles * n_dec_tiles`` tile.
    n_injection_shards : int
        Number of independent injection-generation shards (stage-3). Each
        shard covers an equal slice of the observation window
        (``t_start``/``t_end``), not a fixed event count -- the count per
        shard is a Poisson realisation from ``rate_density``.
    rate_density : float
        BBH merger-rate density in Mpc⁻³ yr⁻¹ (source-frame), used by
        :func:`bagpuss.mdc.pipeline.build_rate` to Poisson-realise the
        injection count physically rather than as a caller-chosen target
        -- see :func:`bagpuss.injection.build_injection_set`. Defaults to
        the local-Universe BBH rate density implied by LVK's GWTC-3
        population estimate (~24 Gpc⁻³ yr⁻¹).
    master_seed : int
        Root seed. Every shard derives an independent, deterministic RNG
        stream from this seed plus its own stage/shard id
        (:func:`bagpuss.mdc.pipeline.shard_rng`).
    store : str
        Path or URL of the zarr store the run writes to.
    output_dir : str
        Local directory (visible to every execute node, e.g. under the
        shared home area) that the detection stage writes frame files and
        asimov blueprints into, as ``<output_dir>/frames/shard_NNNN/`` and
        ``<output_dir>/blueprints/``. Per-injection SNRs go in the zarr
        store, not here.
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

    detectors: dict[str, str] = field(
        default_factory=lambda: {
            "AdvancedLIGOHanford": "AdvancedLIGO_O4",
            "AdvancedLIGOLivingston": "AdvancedLIGO_O4",
        }
    )
    duty_cycles: dict[str, float] = field(
        default_factory=lambda: {
            "AdvancedLIGOHanford": 0.75,
            "AdvancedLIGOLivingston": 0.75,
        }
    )
    duty_cycle_mean_lock_duration: float = 28_800.0  # 8 hours
    injection_channel: str = "Injection"
    snr_threshold: float = 8.0
    write_frames: bool = True
    write_skymaps: bool = False
    skymap_waveform: str = "IMRPhenomD"
    f_ref: float = 20.0
    frame_duration: float = 32.0
    frame_sample_rate: float = 4096.0
    frame_pre_signal_padding: float = 30.0
    chirp_mass_margin: float = 0.5
    trigger_eta_max: float | None = None

    n_ra_tiles: int = 1
    n_dec_tiles: int = 1
    n_injection_shards: int = 1
    rate_density: float = 2.4e-8  # Mpc^-3 yr^-1, ~ GWTC-3 local BBH rate

    master_seed: int = 0
    store: str = "mdc.zarr"
    output_dir: str = "mdc_out"

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


#: Nested-YAML-section -> flat :class:`MDCConfig` field-name schema, mirroring
#: the pipeline's own stages. Each leaf is the flat dataclass field name a
#: key maps to; each non-leaf is a further nested section. A leaf's *value*
#: is taken as-is (so ``detection.network``, whose value is itself a
#: mapping, is assigned directly to ``detectors`` without being validated
#: against this schema -- only the schema's own structure is validated).
_SCHEMA: dict[str, Any] = {
    "cosmology": "cosmology",
    "galaxies": {
        "z_max": "z_max",
        "luminosity_function": {
            "phi_star": "phi_star",
            "m_star": "m_star",
            "alpha": "alpha",
            "m_min": "m_min",
            "m_max": "m_max",
            "m_sun": "m_sun",
        },
        "selection": {"m_lim": "m_lim"},
    },
    "population": {
        "mass": {
            "alpha": "mass_alpha",
            "beta_q": "mass_beta_q",
            "m_min": "mass_m_min",
            "m_max": "mass_m_max",
            "lambda_peak": "mass_lambda_peak",
            "mu_m": "mass_mu_m",
            "sigma_m": "mass_sigma_m",
            "delta_m": "mass_delta_m",
        },
        "spin": {"a_max": "spin_a_max"},
        "rate_density": "rate_density",
    },
    "observation": {"t_start": "t_start", "t_end": "t_end", "d_max": "d_max"},
    "detection": {
        "network": "detectors",
        "duty_cycles": "duty_cycles",
        "duty_cycle_mean_lock_duration": "duty_cycle_mean_lock_duration",
        "channel": "injection_channel",
        "snr_threshold": "snr_threshold",
        "write_frames": "write_frames",
        "write_skymaps": "write_skymaps",
        "skymap_waveform": "skymap_waveform",
        "f_ref": "f_ref",
        "frame_duration": "frame_duration",
        "frame_sample_rate": "frame_sample_rate",
        "frame_pre_signal_padding": "frame_pre_signal_padding",
    },
    "estimation": {
        "chirp_mass_margin": "chirp_mass_margin",
        "trigger_eta_max": "trigger_eta_max",
    },
    "sharding": {
        "n_ra_tiles": "n_ra_tiles",
        "n_dec_tiles": "n_dec_tiles",
        "n_injection_shards": "n_injection_shards",
    },
    "run": {
        "master_seed": "master_seed",
        "store": "store",
        "output_dir": "output_dir",
    },
}


def _flatten(
    raw: dict[str, Any], schema: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Recursively flatten *raw* against *schema* into flat field->value pairs.

    Raises
    ------
    ValueError
        If a key isn't in *schema* at that nesting level, or a section
        expected to be a mapping isn't one.
    """
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        full_key = f"{prefix}{key}"
        if key not in schema:
            valid = sorted(schema)
            raise ValueError(
                f"Unknown MDCConfig key {full_key!r}. Valid keys here: {valid}"
            )

        target = schema[key]
        if isinstance(target, dict):
            if not isinstance(value, dict):
                raise ValueError(
                    f"MDCConfig key {full_key!r} must be a mapping, "
                    f"got {type(value).__name__}"
                )
            flat.update(_flatten(value, target, prefix=f"{full_key}."))
        else:
            flat[target] = value
    return flat


def load_config(path: str | Path) -> MDCConfig:
    """Load an :class:`MDCConfig` from a hierarchical YAML file.

    The file's keys are matched against :data:`_SCHEMA`'s nested sections
    (``galaxies``, ``population``, ``observation``, ``detection``,
    ``estimation``, ``sharding``, ``run``) and applied over
    :class:`MDCConfig`'s defaults; any key that isn't recognised at its
    nesting level raises an error, to catch typos early rather than
    silently ignoring them.

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
        If the file contains a key that isn't part of the schema, or a
        section that should be a mapping isn't one.
    """
    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    flat = _flatten(raw, _SCHEMA)
    return replace(MDCConfig(), **flat)
