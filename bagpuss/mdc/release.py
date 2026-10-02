"""Assemble an MDC data release (the files that go on Zenodo).

The run's zarr store is the generation archive: catalogue tiles, injection
shards and detection shards as written by the condor jobs. This module turns
it into a small number of ordinary files that a downstream user can open
directly, without knowing the shard layout:

``catalogue.h5``
    One dataset per field over every catalogue galaxy, in tile order, so
    row ``i`` is the galaxy that ``host_galaxy_index == i`` refers to.
``injections.h5``
    One table of every injection, in shard order, with the detection results
    (network SNR, whether it was detectable, which detectors were observing,
    skymap) alongside. The ``/injections`` group is also readable by
    :meth:`bagpuss.injection.InjectionSet.from_hdf5`.
``events.yaml``
    The asimov event blueprints of the detectable events.
``skymaps.tar``
    The BAYESTAR skymaps (one FITS file per event with a skymap).
``config.yaml``, ``README.md``, ``SHA256SUMS``, ``MANIFEST.json``
    Provenance, documentation and integrity.
``zenodo_metadata.json``
    Upload metadata for the deposit form or API; not itself part of the upload.

Nothing here modifies the store, so the guarantees recorded in it (notably
the catalogue/injection ``host_galaxy_index`` correspondence) are untouched.
:func:`verify_release` re-reads a release directory and checks it against its
own manifest.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import re
import shutil
import tarfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import yaml

import bagpuss
from bagpuss.injection import _INJECTION_FIELDS, InjectionSet
from bagpuss.mdc.config import MDCConfig
from bagpuss.mdc.pipeline import (
    MDCValidationError,
    _open_store,
    export_glade_catalogue,
    load_consolidated_catalogue,
)

__all__: list[str] = ["build_release", "verify_release"]

_CHUNK_ROWS = 1_000_000

#: Upload metadata for the Zenodo deposit form or API. Not part of the payload
#: (it is neither checksummed nor meant to be uploaded).
ZENODO_METADATA = "zenodo_metadata.json"

#: Creators written to ``.zenodo.json`` unless the caller supplies their own.
_DEFAULT_CREATORS: list[dict[str, str]] = [
    {
        "name": "Williams, Daniel",
        "affiliation": "Institute for Gravitational Research, University of Glasgow",
    }
]


def _portable_config(text: str) -> str:
    """Replace the machine-specific store and output paths in a config's text."""
    text = re.sub(r"(?m)^(\s*store:\s*).*$", r"\1mdc.zarr", text)
    return re.sub(r"(?m)^(\s*output_dir:\s*).*$", r"\1mdc_out", text)


def _dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return the dataset *name* of *handle* (h5py's indexing is typed as a union)."""
    return cast(h5py.Dataset, handle[name])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _event_name(gps: float) -> str:
    """Return the asimov event name for an injection at GPS time *gps*."""
    return f"inj_{gps:.3f}"


def _write_catalogue(config: MDCConfig, path: Path) -> dict[str, Any]:
    """Write ``catalogue.h5`` and return its summary."""
    root = _open_store(config, mode="r")
    index = cast(list[dict[str, Any]], root.require_group("catalogue").attrs["_index"])
    catalogue = load_consolidated_catalogue(config)
    tile_id = np.concatenate(
        [np.full(row["n_galaxies"], row["tile_id"], dtype=np.int16) for row in index]
        or [np.array([], dtype=np.int16)]
    )
    fields = {
        "ra": (catalogue.ra, "radians"),
        "dec": (catalogue.dec, "radians"),
        "redshift": (catalogue.redshifts, "dimensionless"),
        "luminosity": (catalogue.luminosities, "solar luminosities"),
        "apparent_magnitude": (catalogue.apparent_magnitudes, "mag (survey band)"),
        "tile_id": (tile_id, "sky tile the galaxy was drawn in"),
    }
    n = len(catalogue)
    with h5py.File(path, "w") as fh:
        grp = fh.create_group("catalogue")
        for name, (values, unit) in fields.items():
            chunks = (min(_CHUNK_ROWS, n),) if n else None
            ds = grp.create_dataset(
                name,
                data=values,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            ds.attrs["unit"] = unit
        grp.attrs["n_galaxies"] = n
        grp.attrs["row_order"] = (
            "tile order; row i is the galaxy that injections.h5 host_galaxy_index == i "
            "refers to"
        )
        grp.attrs["cosmology"] = config.cosmology
        grp.attrs["m_lim"] = config.m_lim
        grp.attrs["z_max"] = config.z_max
    return {"n_galaxies": n, "n_tiles": len(index)}


def _load_injections(
    config: MDCConfig,
) -> tuple[dict[str, np.ndarray], list[str], list[dict[str, Any]]]:
    """Concatenate injection shards (and detection shards, if present)."""
    root = _open_store(config, mode="r")
    injections = root.require_group("injections")
    index = cast(list[dict[str, Any]], injections.attrs["_index"])
    has_detections = "detections" in root.group_keys()
    detector_names = list(config.detectors)

    columns: dict[str, list[np.ndarray]] = {f: [] for f in _INJECTION_FIELDS}
    columns.update(
        {"shard_id": [], "network_snr": [], "detectable": [], "has_skymap": []}
    )
    active: list[np.ndarray] = []
    for row in index:
        name = f"shard_{row['shard_id']:04d}"
        shard = InjectionSet.from_zarr(injections.require_group(name))
        n = len(shard)
        for field in _INJECTION_FIELDS:
            columns[field].append(getattr(shard, field))
        columns["shard_id"].append(np.full(n, row["shard_id"], dtype=np.int16))
        if has_detections:
            group = root.require_group("detections").require_group(name)
            for field in ("network_snr", "detectable", "has_skymap"):
                columns[field].append(np.asarray(group[field]))
            active.append(np.asarray(group["active"]))
    data = {k: np.concatenate(v) for k, v in columns.items() if v}
    if active:
        data["active"] = np.concatenate(active)
    return data, detector_names, index


def _write_injections(
    data: dict[str, np.ndarray],
    detector_names: list[str],
    names: np.ndarray,
    skymap_files: np.ndarray,
    path: Path,
) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as fh:
        grp = fh.create_group("injections")
        for field in _INJECTION_FIELDS:
            grp.create_dataset(field, data=data[field])
        grp.create_dataset("shard_id", data=data["shard_id"])
        grp.create_dataset("event_name", data=names, dtype=str_dtype)
        if "network_snr" in data:
            grp.create_dataset("network_snr", data=data["network_snr"])
            grp.create_dataset("detectable", data=data["detectable"])
            active = grp.create_dataset("active", data=data["active"])
            active.attrs["detectors"] = detector_names
            grp.create_dataset("has_skymap", data=data["has_skymap"])
            grp.create_dataset("skymap_file", data=skymap_files, dtype=str_dtype)
        grp.attrs["n_injections"] = len(names)
        grp.attrs["row_order"] = "shard order (shards are contiguous time slices)"


def _pack_skymaps(
    config: MDCConfig,
    data: dict[str, np.ndarray],
    names: np.ndarray,
    skymap_dir: Path,
    tar_path: Path,
) -> np.ndarray:
    """Pack the skymaps into a tar; return each injection's path inside it."""
    paths = np.full(len(names), "", dtype=object)
    wanted: Any = np.flatnonzero(data["has_skymap"]) if "has_skymap" in data else []
    missing: list[str] = []
    with tarfile.open(tar_path, "w") as tar:
        for i in map(int, wanted):
            name = str(names[i])
            source = (
                skymap_dir / f"shard_{int(data['shard_id'][i]):04d}" / f"{name}.fits"
            )
            if not source.exists():
                missing.append(str(source))
                continue
            arcname = f"skymaps/{name}.fits"
            tar.add(source, arcname=arcname)
            paths[i] = arcname
    if missing:
        raise MDCValidationError(
            {"issues": [f"{len(missing)} skymap files are missing, e.g. {missing[0]}"]}
        )
    return paths


def _write_events(
    config: MDCConfig, names: np.ndarray, skymap_paths: np.ndarray, path: Path
) -> int:
    """Write ``events.yaml`` (blueprints with skymap paths made relative)."""
    source = Path(config.output_dir) / "blueprints.yaml"
    if not source.exists():
        return 0
    relative = {str(n): str(p) for n, p in zip(names, skymap_paths, strict=True) if p}
    documents = [d for d in yaml.safe_load_all(source.read_text()) if d]
    for doc in documents:
        if "localization file" in doc:
            if doc["name"] not in relative:
                raise MDCValidationError(
                    {"issues": [f"blueprint {doc['name']} names an unknown skymap"]}
                )
            doc["localization file"] = relative[doc["name"]]
    with open(path, "w") as fh:
        yaml.safe_dump_all(documents, fh, sort_keys=False)
    return len(documents)


def _readme(
    config: MDCConfig,
    title: str,
    version: str,
    counts: dict[str, Any],
) -> str:
    n_unlocalised = counts["n_detectable"] - counts["n_skymaps"]
    unlocalised = (
        f" {n_unlocalised} detectable events have no skymap (BAYESTAR produced no "
        "coincidence for them)."
        if n_unlocalised
        else ""
    )
    trigger_note = (
        "Where a `trigger truth` block is present, the trigger's mass ratio was moved "
        "off the equal-mass limit (symmetric mass ratio capped at "
        f"{config.trigger_eta_max}, chirp mass held fixed) so that simple-pe can run, "
        "and `trigger truth` holds the injected masses."
        if config.trigger_eta_max is not None
        else "The trigger holds the injected parameters."
    )
    t0 = config.t_start
    t1 = config.t_end
    return f"""# {title}

Version {version}. Generated with bagpuss {bagpuss.__version__}.

A simulated galaxy catalogue and gravitational-wave injection set for testing
cosmological inference pipelines. **The galaxies and events are simulated;
nothing here is real data.**

## Files

| file | contents |
|---|---|
| `catalogue.h5` | {counts["n_galaxies"]:,} galaxies passing the survey selection (apparent magnitude < {config.m_lim}), one dataset per field |
| `injections.h5` | all {counts["n_injections"]:,} simulated binary black hole mergers, with detection results |
| `events.yaml` | asimov event blueprints for the {counts["n_detectable"]:,} detectable events (injection parameters, trigger, SNR) |
| `skymaps.tar` | {counts["n_skymaps"]:,} BAYESTAR skymaps (`skymaps/<event name>.fits`) |
| `config.yaml` | the configuration this run was generated from |
| `SHA256SUMS`, `MANIFEST.json` | checksums and a machine-readable summary |

## Catalogue (`catalogue.h5`, group `/catalogue`)

`ra`, `dec` (radians), `redshift`, `luminosity` (solar luminosities),
`apparent_magnitude` (survey band), and `tile_id` (the equal-area sky tile the
galaxy was drawn in). Rows are in tile order. Only the *observed*
(selection-passing) galaxies are stored. Distances follow from the redshift
with the `{config.cosmology}` cosmology.

## Injections (`injections.h5`, group `/injections`)

Source-frame masses `m1_source`, `m2_source`; spin magnitudes `a1`, `a2`;
`cos_tilt1`, `cos_tilt2`, `phi12`, `phi_jl`; inclination `theta_jn`; sky
position `ra`, `dec` (radians); polarisation `psi`; geocentric GPS merger time
`geocent_time`; `redshift` and `luminosity_distance` (Mpc) of the host.
`host_galaxy_index` is the row of `catalogue.h5` of the host galaxy, or `-1`
if the host is fainter than the survey limit (not in the catalogue).

Detection results: `network_snr` (optimal network SNR in the detectors that
were observing, 0 if none were), `detectable` (`network_snr >= {config.snr_threshold}`),
`active` (shape `(n, {len(config.detectors)})`: which detectors were observing, in
the order of the dataset's `detectors` attribute), `has_skymap`, `skymap_file`
(path inside `skymaps.tar`, empty if none), `event_name`, and `shard_id`.

The observation window is GPS {t0} to {t1}; the merger rate density is
{config.rate_density:g} Mpc^-3 yr^-1, held constant with redshift, out to
z = {config.z_max}. The group is also readable with `bagpuss.injection.InjectionSet.from_hdf5`.

## Events and skymaps

`events.yaml` holds one asimov event blueprint per detectable event.
{trigger_note}

The skymaps are BAYESTAR skymaps *simulated from the injection parameters and
the detector noise curves* in the way a search pipeline would produce them
(Gaussian measurement errors on the single-detector SNR, phase and arrival
time); they are not derived from the frames' noise realisation.{unlocalised}

## Integrity

`sha256sum -c SHA256SUMS` checks every file.
"""  # noqa: E501


def build_release(
    config: MDCConfig,
    out_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    skymap_dir: str | Path | None = None,
    name: str = "bagpuss-mdc-z3",
    version: str = "1.0.0",
    title: str | None = None,
    creators: list[dict[str, str]] | None = None,
    license_id: str = "cc-by-4.0",
    glade: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Assemble the data release for *config*'s store into *out_dir*.

    Parameters
    ----------
    config : MDCConfig
        Run configuration; its ``store`` must have been through
        ``consolidate`` and ``assemble-injections`` (and, for the detection
        columns and skymaps, the detection stage).
    out_dir : str or pathlib.Path
        Directory to write the release into. Must be empty or absent unless
        *overwrite* is set.
    config_path : str or pathlib.Path or None, optional
        The YAML file *config* was loaded from, copied verbatim into the
        release. Defaults to a YAML dump of *config*.
    skymap_dir : str or pathlib.Path or None, optional
        Where the per-shard skymaps live; defaults to
        ``<config.output_dir>/skymaps``.
    name : str, optional
        Release name, used for the Zenodo title.
    version : str, optional
        Release version.
    title : str or None, optional
        Zenodo title; defaults to a title built from *name*.
    creators : list of dict or None, optional
        Zenodo creators, as ``{"name": "Family, Given", "affiliation": ...}``.
    license_id : str, optional
        Zenodo license identifier for the data.
    glade : bool, optional
        Also export the catalogue as GLADE+-style flat text (gzipped) into
        the release.
    overwrite : bool, optional
        Replace an existing, non-empty *out_dir*.

    Returns
    -------
    dict
        The release manifest (also written to ``MANIFEST.json``).

    Raises
    ------
    MDCValidationError
        If the store is incomplete or skymap files are missing.
    FileExistsError
        If *out_dir* is not empty and *overwrite* is false.
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty; pass overwrite=True")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    root = _open_store(config, mode="r")
    for group in ("catalogue", "injections"):
        if "_index" not in root.require_group(group).attrs:
            raise MDCValidationError(
                {"issues": [f"{group} has not been consolidated (no _index)"]}
            )

    catalogue_summary = _write_catalogue(config, out / "catalogue.h5")
    data, detector_names, _ = _load_injections(config)
    names = np.array(
        [_event_name(float(t)) for t in data["geocent_time"]], dtype=object
    )
    if len(set(names)) != len(names):
        raise MDCValidationError({"issues": ["event names are not unique"]})

    sky_dir = Path(skymap_dir) if skymap_dir else Path(config.output_dir) / "skymaps"
    skymap_files = _pack_skymaps(config, data, names, sky_dir, out / "skymaps.tar")
    _write_injections(data, detector_names, names, skymap_files, out / "injections.h5")
    n_events = _write_events(config, names, skymap_files, out / "events.yaml")

    if config_path is not None:
        text = Path(config_path).read_text()
    else:
        text = yaml.safe_dump(asdict(config))
    (out / "config.yaml").write_text(_portable_config(text))

    if glade:
        glade_dir = out / "glade"
        export_glade_catalogue(config, glade_dir)
        with (
            open(glade_dir / "catalogue.dat", "rb") as src,
            gzip.open(out / "glade_catalogue.dat.gz", "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
        shutil.copy(glade_dir / "README.txt", out / "glade_README.txt")
        shutil.copy(glade_dir / "completeness.dat", out / "glade_completeness.dat")
        shutil.rmtree(glade_dir)

    counts = {
        "n_galaxies": catalogue_summary["n_galaxies"],
        "n_tiles": catalogue_summary["n_tiles"],
        "n_injections": len(names),
        "n_detectable": int(data["detectable"].sum()) if "detectable" in data else 0,
        "n_skymaps": int((skymap_files != "").sum()),
        "n_events": n_events,
    }
    release_title = title or f"{name}: simulated galaxy catalogue and GW injections"
    (out / "README.md").write_text(_readme(config, release_title, version, counts))

    _write_zenodo_metadata(
        out / ZENODO_METADATA, release_title, version, creators, license_id, counts
    )
    return _write_manifest(out, name, version, counts, config)


def _write_zenodo_metadata(
    path: Path,
    title: str,
    version: str,
    creators: list[dict[str, str]] | None,
    license_id: str,
    counts: dict[str, Any],
) -> None:
    description = (
        "<p>A simulated galaxy catalogue and gravitational-wave injection set for "
        "testing cosmological inference pipelines, generated with "
        '<a href="https://github.com/transientlunatic/bagpuss">bagpuss</a>. '
        "<b>The galaxies and events are simulated; nothing here is real data.</b></p>"
        f"<p>The catalogue has {counts['n_galaxies']:,} galaxies; the injection "
        f"set has {counts['n_injections']:,} binary black hole mergers, of which "
        f"{counts['n_detectable']:,} are detectable, with {counts['n_skymaps']:,} "
        "BAYESTAR skymaps. See README.md for the file layout and field definitions.</p>"
    )
    metadata = {
        "title": title,
        "upload_type": "dataset",
        "description": description,
        "creators": creators or _DEFAULT_CREATORS,
        "license": license_id,
        "access_right": "open",
        "version": version,
        "keywords": [
            "gravitational waves",
            "mock data challenge",
            "galaxy catalogue",
            "dark sirens",
            "cosmology",
            "simulation",
        ],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def _write_manifest(
    out: Path, name: str, version: str, counts: dict[str, Any], config: MDCConfig
) -> dict[str, Any]:
    files = sorted(
        p
        for p in out.iterdir()
        if p.is_file() and p.name not in {"SHA256SUMS", ZENODO_METADATA}
    )
    checksums = {p.name: _sha256(p) for p in files if p.name != "MANIFEST.json"}
    (out / "SHA256SUMS").write_text(
        "".join(f"{digest}  {fname}\n" for fname, digest in sorted(checksums.items()))
    )
    manifest = {
        "name": name,
        "version": version,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "bagpuss_version": bagpuss.__version__,
        "counts": counts,
        "master_seed": config.master_seed,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": checksums.get(p.name)}
            for p in files
        },
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_release(out_dir: str | Path) -> dict[str, Any]:
    """Check a release directory against its own manifest.

    Verifies every checksum in ``SHA256SUMS``, that the dataset row counts match
    ``MANIFEST.json``, that every catalogued host's sky position and redshift in
    ``injections.h5`` agree with its row of ``catalogue.h5``, and that
    ``skymaps.tar`` holds exactly the skymaps the injection table names.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        A directory written by :func:`build_release`.

    Returns
    -------
    dict
        ``{"ok": bool, "issues": [...]}``.
    """
    out = Path(out_dir)
    issues: list[str] = []
    manifest = json.loads((out / "MANIFEST.json").read_text())
    counts = manifest["counts"]

    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, _, fname = line.partition("  ")
        if not (out / fname).exists():
            issues.append(f"missing file: {fname}")
        elif _sha256(out / fname) != digest:
            issues.append(f"checksum mismatch: {fname}")

    with (
        h5py.File(out / "catalogue.h5", "r") as cat,
        h5py.File(out / "injections.h5", "r") as inj,
    ):
        n_gal = _dataset(cat, "catalogue/ra").shape[0]
        n_inj = _dataset(inj, "injections/geocent_time").shape[0]
        if n_gal != counts["n_galaxies"]:
            issues.append(
                f"catalogue has {n_gal} rows, manifest says {counts['n_galaxies']}"
            )
        if n_inj != counts["n_injections"]:
            issues.append(
                f"injections has {n_inj} rows, manifest says {counts['n_injections']}"
            )

        host = _dataset(inj, "injections/host_galaxy_index")[()]
        hosted = np.flatnonzero(host >= 0)
        if len(hosted):
            rows = host[hosted]
            if rows.max() >= n_gal:
                issues.append("host_galaxy_index points past the end of the catalogue")
            else:
                order = np.argsort(rows)
                rows_sorted = rows[order]
                for field, cat_field in (
                    ("ra", "ra"),
                    ("dec", "dec"),
                    ("redshift", "redshift"),
                ):
                    expected = _dataset(cat, f"catalogue/{cat_field}")[rows_sorted]
                    got = _dataset(inj, f"injections/{field}")[hosted][order]
                    if not np.allclose(expected, got):
                        issues.append(f"host {field} disagrees with the catalogue")

        listed = (
            {
                s.decode() if isinstance(s, bytes) else s
                for s in _dataset(inj, "injections/skymap_file")[()]
            }
            - {""}
            if "injections/skymap_file" in inj
            else set()
        )
    with tarfile.open(out / "skymaps.tar") as tar:
        members = {m.name for m in tar.getmembers() if m.isfile()}
    if members != listed:
        issues.append(
            f"skymaps.tar has {len(members)} files, injection table names {len(listed)}"
        )
    if len(members) != counts["n_skymaps"]:
        issues.append(
            f"skymaps.tar has {len(members)} files, manifest says {counts['n_skymaps']}"
        )

    return {"ok": not issues, "issues": issues}
