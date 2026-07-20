"""HTCondor submit-file and DAG rendering for the MDC pipeline.

Turns an :class:`~bagpuss.mdc.config.MDCConfig` into an HTCondor DAGMan
file (``mdc.dag``) plus one submit description per pipeline stage, wiring
up the shape described in :mod:`bagpuss.mdc.pipeline`::

    catalogue_tile_0000 .. catalogue_tile_NNNN   (parallel)
        PARENT of  consolidate_catalogue
        CHILD of   injection_shard_0000 .. injection_shard_MMMM (parallel)
            CHILD of  assemble_injections
                CHILD of  package_manifest

Each node's own exit status is the go/no-go gate between tiers:
``consolidate_catalogue``/``assemble_injections`` raise
:class:`~bagpuss.mdc.pipeline.MDCValidationError` (nonzero exit, via the
``bagpuss mdc`` CLI) when their sanity checks fail, which DAGMan already
treats as a failed node -- so no separate POST script is needed to gate
the DAG.

This module only *renders* files; it never calls ``condor_submit`` or
``condor_submit_dag`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bagpuss.mdc.config import MDCConfig

__all__: list[str] = ["CondorResources", "DEFAULT_RESOURCES", "write_dag"]


@dataclass(frozen=True)
class CondorResources:
    """Per-node HTCondor resource request.

    Parameters
    ----------
    request_cpus : int
        Number of CPU cores to request.
    request_memory : str
        Memory request, HTCondor units (e.g. ``"4GB"``).
    request_disk : str
        Disk request, HTCondor units (e.g. ``"4GB"``).
    """

    request_cpus: int = 1
    request_memory: str = "2GB"
    request_disk: str = "2GB"


#: Sensible starting points per stage; override with real numbers once a
#: calibration run (see the MDC plan) has measured actual usage.
DEFAULT_RESOURCES: dict[str, CondorResources] = {
    "catalogue_tile": CondorResources(
        request_cpus=1, request_memory="4GB", request_disk="4GB"
    ),
    "consolidate_catalogue": CondorResources(
        request_cpus=1, request_memory="4GB", request_disk="2GB"
    ),
    "injection_shard": CondorResources(
        request_cpus=1, request_memory="4GB", request_disk="2GB"
    ),
    "assemble_injections": CondorResources(
        request_cpus=1, request_memory="4GB", request_disk="2GB"
    ),
    "package_manifest": CondorResources(
        request_cpus=1, request_memory="2GB", request_disk="2GB"
    ),
}

_SUBMIT_TEMPLATE = """\
executable              = {executable}
arguments               = "{arguments}"
should_transfer_files   = NO
getenv                  = True
request_cpus            = {request_cpus}
request_memory          = {request_memory}
request_disk            = {request_disk}
log                     = {log_dir}/{log_stem}.log
output                  = {log_dir}/{log_stem}.out
error                   = {log_dir}/{log_stem}.err
queue
"""

#: Log-file stem per stage. The two sharded stages interpolate the
#: per-job condor macro, so concurrently-running tile/shard jobs each get
#: their own log files rather than clobbering a single shared one; the
#: four singleton stages just use their own name.
_LOG_STEMS: dict[str, str] = {
    "catalogue_tile": "catalogue_tile_$(tile_id)",
    "consolidate_catalogue": "consolidate_catalogue",
    "injection_shard": "injection_shard_$(shard_id)",
    "assemble_injections": "assemble_injections",
    "package_manifest": "package_manifest",
}


def _render_submit_file(
    log_stem: str,
    executable: str,
    arguments: str,
    log_dir: Path,
    resources: CondorResources,
) -> str:
    """Render a single HTCondor submit description."""
    return _SUBMIT_TEMPLATE.format(
        executable=executable,
        arguments=arguments,
        log_dir=log_dir,
        log_stem=log_stem,
        request_cpus=resources.request_cpus,
        request_memory=resources.request_memory,
        request_disk=resources.request_disk,
    )


def _write_submit_files(
    config: MDCConfig,
    config_path: str | Path,
    out_dir: Path,
    log_dir: Path,
    bagpuss_executable: str,
    resources: dict[str, CondorResources],
) -> dict[str, Path]:
    """Write one .sub file per pipeline stage; return {stage: path}."""
    specs = {
        "catalogue_tile": (
            f"mdc generate-tile --config {config_path} --tile-id $(tile_id)"
        ),
        "consolidate_catalogue": f"mdc consolidate --config {config_path}",
        "injection_shard": (
            f"mdc generate-injections --config {config_path} --shard-id $(shard_id)"
        ),
        "assemble_injections": f"mdc assemble-injections --config {config_path}",
        "package_manifest": f"mdc package --config {config_path}",
    }
    paths: dict[str, Path] = {}
    for name, arguments in specs.items():
        text = _render_submit_file(
            _LOG_STEMS[name], bagpuss_executable, arguments, log_dir, resources[name]
        )
        path = out_dir / f"{name}.sub"
        path.write_text(text)
        paths[name] = path
    return paths


def _render_dag(config: MDCConfig, submit_paths: dict[str, Path]) -> str:
    """Render the full ``mdc.dag`` contents for *config*."""
    lines: list[str] = []

    tile_names = [f"catalogue_tile_{i:04d}" for i in range(config.n_tiles)]
    for i, name in enumerate(tile_names):
        lines.append(f"JOB {name} {submit_paths['catalogue_tile']}")
        lines.append(f'VARS {name} tile_id="{i:04d}"')
        lines.append(f"RETRY {name} 2")
    lines.append("")

    lines.append(f"JOB consolidate_catalogue {submit_paths['consolidate_catalogue']}")
    lines.append("RETRY consolidate_catalogue 2")
    lines.append(f"PARENT {' '.join(tile_names)} CHILD consolidate_catalogue")
    lines.append("")

    shard_names = [f"injection_shard_{i:04d}" for i in range(config.n_injection_shards)]
    for i, name in enumerate(shard_names):
        lines.append(f"JOB {name} {submit_paths['injection_shard']}")
        lines.append(f'VARS {name} shard_id="{i:04d}"')
        lines.append(f"RETRY {name} 2")
    lines.append(f"PARENT consolidate_catalogue CHILD {' '.join(shard_names)}")
    lines.append("")

    lines.append(f"JOB assemble_injections {submit_paths['assemble_injections']}")
    lines.append("RETRY assemble_injections 2")
    lines.append(f"PARENT {' '.join(shard_names)} CHILD assemble_injections")
    lines.append("")

    lines.append(f"JOB package_manifest {submit_paths['package_manifest']}")
    lines.append("RETRY package_manifest 2")
    lines.append("PARENT assemble_injections CHILD package_manifest")
    lines.append("")

    return "\n".join(lines)


def write_dag(
    config: MDCConfig,
    config_path: str | Path,
    out_dir: str | Path,
    bagpuss_executable: str,
    resources: dict[str, CondorResources] | None = None,
) -> Path:
    """Render and write the full DAG plus submit files for *config*.

    Parameters
    ----------
    config : MDCConfig
        Run configuration (determines ``n_tiles``/``n_injection_shards``,
        i.e. the DAG's shape).
    config_path : str or pathlib.Path
        Path to the YAML file *config* was loaded from -- passed through
        as ``--config`` to every ``bagpuss mdc`` job, so jobs re-load the
        config themselves rather than it being baked into the DAG.
    out_dir : str or pathlib.Path
        Directory to write ``mdc.dag`` and the per-stage ``.sub`` files
        into. Created if it doesn't exist. A ``logs/`` subdirectory is
        created for job logs.
    bagpuss_executable : str
        Absolute path to the ``bagpuss`` entry point to run on the execute
        node (e.g. a venv's ``bin/bagpuss``). Required rather than
        defaulting to a bare ``"bagpuss"``, since condor execute nodes
        don't inherit the submitting shell's ``PATH`` resolution.
    resources : dict[str, CondorResources] or None, optional
        Per-stage resource overrides. Falls back to
        :data:`DEFAULT_RESOURCES` for any stage not given.

    Returns
    -------
    pathlib.Path
        Path to the written ``mdc.dag`` file.
    """
    out_dir = Path(out_dir)
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    merged_resources = dict(DEFAULT_RESOURCES)
    if resources is not None:
        merged_resources.update(resources)

    submit_paths = _write_submit_files(
        config, config_path, out_dir, log_dir, bagpuss_executable, merged_resources
    )
    dag_text = _render_dag(config, submit_paths)

    dag_path = out_dir / "mdc.dag"
    dag_path.write_text(dag_text)
    return dag_path
