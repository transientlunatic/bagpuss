"""Command-line interface for bagpuss.

Provides the ``bagpuss`` CLI entry point, with subcommands for generating
plots from the simulation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np
import yaml

from bagpuss.luminosity import SchechterLuminosityModel
from bagpuss.universe import LuminosityModel, PointProcess, Structure, Universe

__all__: list[str] = ["main"]

# ---------------------------------------------------------------------------
# Model selection constants
# ---------------------------------------------------------------------------

_STRUCTURE_CHOICES: tuple[str, ...] = ("point_process",)
_LUMINOSITY_CHOICES: tuple[str, ...] = ("schechter",)

_COSMOLOGY_NAMES: tuple[str, ...] = (
    "Planck13",
    "Planck15",
    "Planck18",
    "WMAP1",
    "WMAP3",
    "WMAP5",
    "WMAP7",
    "WMAP9",
)

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _load_cosmology(name: str) -> Any:  # noqa: ANN401
    """Return the astropy cosmology object for *name*.

    Parameters
    ----------
    name : str
        Name of a built-in astropy cosmology (e.g. ``"Planck18"``).

    Returns
    -------
    astropy.cosmology.Cosmology
        The named cosmology object.

    Raises
    ------
    click.BadParameter
        If *name* is not a recognised astropy built-in cosmology.
    """
    import astropy.cosmology as ac

    obj = getattr(ac, name, None)
    if obj is None:
        raise click.BadParameter(
            f"Unknown cosmology {name!r}. Valid choices: {', '.join(_COSMOLOGY_NAMES)}",
            param_hint="--cosmology",
        )
    return obj


def _build_structure(cfg: dict[str, Any]) -> Structure:
    """Instantiate the structure model named in *cfg*.

    Parameters
    ----------
    cfg : dict[str, Any]
        Merged configuration dictionary.

    Returns
    -------
    Structure
        Instantiated structure model.

    Raises
    ------
    click.BadParameter
        If the structure model name is not recognised.
    """
    name = cfg["structure"]
    if name == "point_process":
        return PointProcess(z_max=float(cfg["z_max"]))
    raise click.BadParameter(
        f"Unknown structure model {name!r}. "
        f"Valid choices: {', '.join(_STRUCTURE_CHOICES)}",
        param_hint="--structure",
    )


def _build_luminosity(cfg: dict[str, Any]) -> LuminosityModel:
    """Instantiate the luminosity model named in *cfg*.

    Parameters
    ----------
    cfg : dict[str, Any]
        Merged configuration dictionary.

    Returns
    -------
    LuminosityModel
        Instantiated luminosity model.

    Raises
    ------
    click.BadParameter
        If the luminosity model name is not recognised.
    """
    name = cfg["luminosity"]
    if name == "schechter":
        return SchechterLuminosityModel(
            phi_star=float(cfg["phi_star"]),
            m_star=float(cfg["m_star"]),
            alpha=float(cfg["alpha"]),
            m_min=float(cfg["m_min"]),
            m_max=float(cfg["m_max"]),
            m_sun=float(cfg["m_sun"]),
        )
    raise click.BadParameter(
        f"Unknown luminosity model {name!r}. "
        f"Valid choices: {', '.join(_LUMINOSITY_CHOICES)}",
        param_hint="--luminosity",
    )


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "cosmology": "Planck18",
    "structure": "point_process",
    "luminosity": "schechter",
    "n_galaxies": 10000,
    "seed": None,
    "z_max": 1.0,
    "phi_star": 1.61e-2,
    "m_star": -19.66,
    "alpha": -1.16,
    "m_min": -25.0,
    "m_max": -14.0,
    "m_sun": 4.83,
    "bins": 50,
    "title": "Galaxy density heatmap",
    "colormap": "viridis",
    "colorbar_label": "Count",
    "output": None,
}


def _merge_config(
    config_path: Path | None,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge defaults, YAML file values, and CLI overrides.

    Priority (highest last): hard-coded defaults → YAML file → CLI flags.
    CLI flags are only applied when their value is not ``None``.

    Parameters
    ----------
    config_path : pathlib.Path or None
        Path to a YAML configuration file, or ``None`` if not provided.
    cli_overrides : dict[str, Any]
        Mapping of parameter name to CLI-provided value (``None`` if not set).

    Returns
    -------
    dict[str, Any]
        Fully-merged configuration dictionary.

    Raises
    ------
    click.BadParameter
        If the config file cannot be read or contains invalid YAML.
    """
    cfg: dict[str, Any] = dict(_DEFAULTS)

    if config_path is not None:
        try:
            with open(config_path) as fh:
                yaml_data: dict[str, Any] = yaml.safe_load(fh) or {}
        except OSError as exc:
            raise click.BadParameter(
                f"Cannot read config file: {exc}", param_hint="--config"
            ) from exc
        except yaml.YAMLError as exc:
            raise click.BadParameter(
                f"Invalid YAML in config file: {exc}", param_hint="--config"
            ) from exc

        cfg.update(yaml_data.get("model", {}))
        cfg.update(yaml_data.get("structure", {}).get("point_process", {}))
        cfg.update(yaml_data.get("luminosity", {}).get("schechter", {}))
        cfg.update(yaml_data.get("plot", {}))

    for key, val in cli_overrides.items():
        if val is not None:
            cfg[key] = val

    return cfg


# ---------------------------------------------------------------------------
# Click command tree
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Bagpuss — synthetic galaxy catalogue tools."""


@main.group()
def plot() -> None:
    """Run plotting subcommands."""


@plot.command("heatmap")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML config file.",
)
@click.option(
    "--cosmology",
    type=str,
    default=None,
    help="Astropy cosmology name (e.g. Planck18).",
)
@click.option(
    "--structure",
    type=click.Choice(list(_STRUCTURE_CHOICES)),
    default=None,
    help="Structure model.",
)
@click.option(
    "--luminosity",
    type=click.Choice(list(_LUMINOSITY_CHOICES)),
    default=None,
    help="Luminosity model.",
)
@click.option(
    "--n-galaxies", type=int, default=None, help="Number of galaxies to sample."
)
@click.option("--seed", type=int, default=None, help="RNG seed for reproducibility.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (PDF/PNG). Omit to display interactively.",
)
@click.option(
    "--z-max", type=float, default=None, help="Maximum redshift (PointProcess)."
)
@click.option("--bins", type=int, default=None, help="Histogram bins along each axis.")
@click.option("--title", type=str, default=None, help="Plot title.")
@click.option("--colormap", type=str, default=None, help="Matplotlib colormap name.")
@click.option("--colorbar-label", type=str, default=None, help="Colorbar axis label.")
def heatmap_cmd(
    config: Path | None,
    cosmology: str | None,
    structure: str | None,
    luminosity: str | None,
    n_galaxies: int | None,
    seed: int | None,
    output: Path | None,
    z_max: float | None,
    bins: int | None,
    title: str | None,
    colormap: str | None,
    colorbar_label: str | None,
) -> None:
    """Generate a 2-D redshift–luminosity density heatmap."""
    import matplotlib.pyplot as plt

    from bagpuss.plotting import plot_universe_heatmap

    cli_overrides: dict[str, Any] = {
        "cosmology": cosmology,
        "structure": structure,
        "luminosity": luminosity,
        "n_galaxies": n_galaxies,
        "seed": seed,
        "output": output,
        "z_max": z_max,
        "bins": bins,
        "title": title,
        "colormap": colormap,
        "colorbar_label": colorbar_label,
    }

    cfg = _merge_config(config, cli_overrides)

    cosmo = _load_cosmology(str(cfg["cosmology"]))
    struct = _build_structure(cfg)
    lum_model = _build_luminosity(cfg)
    universe = Universe(cosmology=cosmo, structure=struct, luminosity=lum_model)

    rng = np.random.default_rng(cfg["seed"])
    galaxies = universe.sample(int(cfg["n_galaxies"]), rng=rng)

    fig = plot_universe_heatmap(
        galaxies,
        bins=int(cfg["bins"]),
        title=str(cfg["title"]),
        colormap=str(cfg["colormap"]),
        colorbar_label=str(cfg["colorbar_label"]),
    )

    if cfg["output"] is not None:
        fig.savefig(cfg["output"], dpi=150, bbox_inches="tight")
        click.echo(f"Saved heatmap to {cfg['output']}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# MDC (Mock Data Challenge) pipeline
# ---------------------------------------------------------------------------


@main.group()
def mdc() -> None:
    """Mock Data Challenge (MDC) generation pipeline.

    One subcommand per stage of the sharded, condor-friendly pipeline in
    :mod:`bagpuss.mdc.pipeline`. Every subcommand takes ``--config``, a YAML
    file loaded via :func:`bagpuss.mdc.config.load_config` -- see
    ``docs/mdc.rst`` for the full config schema and pipeline shape.
    """


_CONFIG_OPTION = click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="YAML MDCConfig file (see bagpuss.mdc.config.MDCConfig).",
)


@mdc.command("generate-tile")
@_CONFIG_OPTION
@click.option("--tile-id", type=int, required=True, help="Sky-tile index to generate.")
def mdc_generate_tile(config: Path, tile_id: int) -> None:
    """Generate one galaxy-catalogue sky tile (a Tier-1 DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import generate_catalogue_tile

    cfg = load_config(config)
    catalogue = generate_catalogue_tile(cfg, tile_id)
    click.echo(f"tile {tile_id}: {len(catalogue)} galaxies written to {cfg.store}")


@mdc.command("consolidate")
@_CONFIG_OPTION
def mdc_consolidate(config: Path) -> None:
    """Validate and consolidate all catalogue tiles (the Tier-2 DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import MDCValidationError, consolidate_catalogue

    cfg = load_config(config)
    try:
        report = consolidate_catalogue(cfg)
    except MDCValidationError as exc:
        click.echo(f"Catalogue validation FAILED: {exc.report['issues']}", err=True)
        raise SystemExit(1) from exc
    click.echo(
        f"consolidated {report['n_tiles']} tiles, {report['total_galaxies']} galaxies"
    )


@mdc.command("generate-injections")
@_CONFIG_OPTION
@click.option(
    "--shard-id", type=int, required=True, help="Injection-shard index to generate."
)
def mdc_generate_injections(config: Path, shard_id: int) -> None:
    """Generate one injection shard (a Tier-3 DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import generate_injection_shard

    cfg = load_config(config)
    injections = generate_injection_shard(cfg, shard_id)
    click.echo(f"shard {shard_id}: {len(injections)} injections written to {cfg.store}")


@mdc.command("assemble-injections")
@_CONFIG_OPTION
def mdc_assemble_injections(config: Path) -> None:
    """Validate and consolidate all injection shards (the Tier-4 DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import MDCValidationError, assemble_injections

    cfg = load_config(config)
    try:
        report = assemble_injections(cfg)
    except MDCValidationError as exc:
        click.echo(f"Injection validation FAILED: {exc.report['issues']}", err=True)
        raise SystemExit(1) from exc
    click.echo(
        f"consolidated {report['n_shards']} shards, "
        f"{report['total_injections']} injections"
    )


@mdc.command("detect-injections")
@_CONFIG_OPTION
@click.option(
    "--shard-id", type=int, required=True, help="Injection-shard index to process."
)
def mdc_detect_injections(config: Path, shard_id: int) -> None:
    """Compute SNRs and write frames/blueprints for one shard (a DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import detect_injection_shard

    cfg = load_config(config)
    summary = detect_injection_shard(cfg, shard_id)
    click.echo(
        f"shard {shard_id}: {summary['n_detectable']}/{summary['n_injections']} "
        f"injections above SNR {cfg.snr_threshold}"
    )


@mdc.command("assemble-detections")
@_CONFIG_OPTION
def mdc_assemble_detections(config: Path) -> None:
    """Validate detection shards and merge their blueprints (a DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import MDCValidationError, assemble_detections

    cfg = load_config(config)
    try:
        report = assemble_detections(cfg)
    except MDCValidationError as exc:
        click.echo(f"Detection validation FAILED: {exc.report['issues']}", err=True)
        raise SystemExit(1) from exc
    click.echo(
        f"{report['total_detectable']}/{report['total_injections']} injections "
        f"detectable across {report['n_shards']} shards"
    )


@mdc.command("export-release")
@_CONFIG_OPTION
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to assemble the release in (must be empty or absent).",
)
@click.option(
    "--skymap-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Per-shard skymap directory (default: <output_dir>/skymaps).",
)
@click.option("--name", default="bagpuss-mdc-z3", show_default=True)
@click.option("--release-version", default="1.0.0", show_default=True)
@click.option("--title", default=None, help="Zenodo title.")
@click.option(
    "--creator",
    "creators",
    multiple=True,
    help="Zenodo creator as 'Family, Given;Affiliation' (repeatable).",
)
@click.option("--license", "license_id", default="cc-by-4.0", show_default=True)
@click.option("--glade", is_flag=True, help="Also include a GLADE+-style catalogue.")
@click.option("--overwrite", is_flag=True, help="Replace a non-empty --out-dir.")
def mdc_export_release(  # noqa: PLR0913
    config: Path,
    out_dir: Path,
    skymap_dir: Path | None,
    name: str,
    release_version: str,
    title: str | None,
    creators: tuple[str, ...],
    license_id: str,
    glade: bool,
    overwrite: bool,
) -> None:
    """Assemble the data release (the files that go on Zenodo) from the store."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import MDCValidationError
    from bagpuss.mdc.release import build_release, verify_release

    cfg = load_config(config)
    parsed = [
        {"name": n.strip(), "affiliation": a.strip()}
        for n, _, a in (c.partition(";") for c in creators)
    ]
    try:
        manifest = build_release(
            cfg,
            out_dir,
            config_path=config,
            skymap_dir=skymap_dir,
            name=name,
            version=release_version,
            title=title,
            creators=parsed or None,
            license_id=license_id,
            glade=glade,
            overwrite=overwrite,
        )
    except (MDCValidationError, FileExistsError) as exc:
        click.echo(f"Release FAILED: {exc}", err=True)
        raise SystemExit(1) from exc
    report = verify_release(out_dir)
    for issue in report["issues"]:
        click.echo(f"verify: {issue}", err=True)
    c = manifest["counts"]
    click.echo(
        f"wrote {len(manifest['files'])} files to {out_dir}: "
        f"{c['n_galaxies']} galaxies, {c['n_injections']} injections, "
        f"{c['n_detectable']} detectable, "
        f"{c['n_skymaps']} skymaps; verification {'OK' if report['ok'] else 'FAILED'}"
    )
    click.echo(
        "Upload every file except zenodo_metadata.json, which holds the metadata "
        "for the deposit form or API."
    )
    if not report["ok"]:
        raise SystemExit(1)


@mdc.command("verify-release")
@click.argument("release_dir", type=click.Path(exists=True, path_type=Path))
def mdc_verify_release(release_dir: Path) -> None:
    """Check a release directory against its own manifest and checksums."""
    from bagpuss.mdc.release import verify_release

    report = verify_release(release_dir)
    for issue in report["issues"]:
        click.echo(issue, err=True)
    click.echo("release OK" if report["ok"] else "release FAILED")
    if not report["ok"]:
        raise SystemExit(1)


@mdc.command("package")
@_CONFIG_OPTION
def mdc_package(config: Path) -> None:
    """Write the run's manifest/provenance (the Tier-5 DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import package_manifest

    cfg = load_config(config)
    manifest = package_manifest(cfg)
    click.echo(f"wrote manifest for {cfg.store}: {manifest}")


@mdc.command("export-glade")
@_CONFIG_OPTION
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write catalogue.dat, completeness.dat, and README.txt into.",
)
def mdc_export_glade(config: Path, out_dir: Path) -> None:
    """Export the consolidated catalogue as a GLADE+-style flat-ASCII product."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import MDCValidationError, export_glade_catalogue

    cfg = load_config(config)
    try:
        report = export_glade_catalogue(cfg, out_dir)
    except MDCValidationError as exc:
        click.echo(f"Export FAILED: {exc.report['issues']}", err=True)
        raise SystemExit(1) from exc
    click.echo(f"exported {report['n_galaxies']} galaxies to {report['out_dir']}")


@mdc.command("expected-count")
@_CONFIG_OPTION
@click.option(
    "--d-max",
    type=float,
    default=None,
    help="Override config.d_max (Mpc) for the detected-count estimate.",
)
@click.option(
    "--t-obs-years",
    type=float,
    default=None,
    help="Observation window length in years, overriding config.t_start/t_end "
    "(the window still starts at config.t_start, defaulting to the O3 start).",
)
def mdc_expected_count(
    config: Path, d_max: float | None, t_obs_years: float | None
) -> None:
    """Print the expected injection count implied by a config -- no simulation run.

    A pure closed-form calculation (see
    :func:`bagpuss.mdc.pipeline.expected_injection_count`): useful for
    sanity-checking a proposed ``rate_density``/observation-window/detector
    sensitivity before submitting an expensive DAG.
    """
    from bagpuss.injection import GPS_O3_START
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import expected_injection_count
    from bagpuss.population import SECONDS_PER_YEAR

    cfg = load_config(config)

    t_end = None
    if t_obs_years is not None:
        t_start_base = cfg.t_start if cfg.t_start is not None else GPS_O3_START
        t_end = t_start_base + t_obs_years * SECONDS_PER_YEAR

    result = expected_injection_count(cfg, d_max=d_max, t_end=t_end)

    click.echo(f"observation window: {result['t_obs_years']:.3f} years")
    click.echo(
        f"expected mergers in survey volume (z_max={cfg.z_max}): "
        f"{result['n_total']:.2f}"
    )
    if result["d_max"] is not None:
        click.echo(
            f"expected DETECTED mergers (d_max={result['d_max']:.1f} Mpc, "
            f"hard distance-threshold placeholder): {result['n_detected']:.2f}"
        )
    else:
        click.echo("no d_max configured or passed -- detected count not estimated")


@mdc.command("make-dag")
@_CONFIG_OPTION
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to write mdc.dag and the per-stage .sub files into.",
)
@click.option(
    "--bagpuss-executable",
    type=str,
    required=True,
    help="Absolute path to the bagpuss entry point on the execute node "
    "(e.g. a venv's bin/bagpuss).",
)
@click.option(
    "--skip-detection",
    is_flag=True,
    help="Omit the detection stage (SNRs/frames/blueprints) from the DAG.",
)
@click.option(
    "--detection-only",
    is_flag=True,
    help="Render only the detection stage, for an already-generated store "
    "(never regenerates the catalogue or injections).",
)
def mdc_make_dag(
    config: Path,
    out_dir: Path,
    bagpuss_executable: str,
    skip_detection: bool,
    detection_only: bool,
) -> None:
    """Render the HTCondor DAG and submit files for a run, without submitting."""
    from bagpuss.mdc.condor import write_dag
    from bagpuss.mdc.config import load_config

    cfg = load_config(config)
    dag_path = write_dag(
        cfg,
        config,
        out_dir,
        bagpuss_executable,
        include_detection=not skip_detection,
        include_generation=not detection_only,
    )
    click.echo(
        f"Wrote DAG ({cfg.n_tiles} tiles, {cfg.n_injection_shards} shards) "
        f"to {dag_path}"
    )
