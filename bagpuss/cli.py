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


@mdc.command("package")
@_CONFIG_OPTION
def mdc_package(config: Path) -> None:
    """Write the run's manifest/provenance (the Tier-5 DAG node)."""
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import package_manifest

    cfg = load_config(config)
    manifest = package_manifest(cfg)
    click.echo(f"wrote manifest for {cfg.store}: {manifest}")


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
def mdc_make_dag(config: Path, out_dir: Path, bagpuss_executable: str) -> None:
    """Render the HTCondor DAG and submit files for a run, without submitting."""
    from bagpuss.mdc.condor import write_dag
    from bagpuss.mdc.config import load_config

    cfg = load_config(config)
    dag_path = write_dag(cfg, config, out_dir, bagpuss_executable)
    click.echo(
        f"Wrote DAG ({cfg.n_tiles} tiles, {cfg.n_injection_shards} shards) "
        f"to {dag_path}"
    )
