"""Tests for bagpuss.mdc.release (assembling the data release)."""

# zarr and h5py index into a union of Group | Array | Dataset; the tests know which.
# pyright: reportIndexIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportReturnType=false, reportOptionalSubscript=false, reportGeneralTypeIssues=false

import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml
from click.testing import CliRunner

from bagpuss.cli import main
from bagpuss.injection import GPS_O3_END, GPS_O3_START, InjectionSet
from bagpuss.mdc.config import MDCConfig
from bagpuss.mdc.pipeline import (
    MDCValidationError,
    assemble_detections,
    assemble_injections,
    consolidate_catalogue,
    detect_injection_shard,
    generate_catalogue_tile,
    generate_injection_shard,
    load_consolidated_catalogue,
)
from bagpuss.mdc.release import _sha256, build_release, verify_release
from tests.test_mdc_pipeline import _FakeBackend, _FakeLocalizer, _tiny_config


def _run_pipeline(tmp: str) -> MDCConfig:
    """Generate a tiny store with detections and skymaps (fake waveform engine)."""
    cfg = _tiny_config(
        n_ra_tiles=2,
        n_injection_shards=3,
        store=str(Path(tmp) / "store.zarr"),
        output_dir=str(Path(tmp) / "out"),
        snr_threshold=8.0,
        write_frames=False,
        write_skymaps=True,
    )
    for tile in range(cfg.n_tiles):
        generate_catalogue_tile(cfg, tile)
    consolidate_catalogue(cfg)
    for shard in range(cfg.n_injection_shards):
        generate_injection_shard(cfg, shard)
    assemble_injections(cfg)
    for shard in range(cfg.n_injection_shards):
        detect_injection_shard(
            cfg, shard, backend=_FakeBackend(), localizer=_FakeLocalizer()
        )
    assemble_detections(cfg)
    return cfg


class TestBuildRelease(unittest.TestCase):
    """build_release assembles flat files from the sharded store."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one release for the whole class."""
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cfg = _run_pipeline(cls._tmp.name)
        cls.out = Path(cls._tmp.name) / "release"
        cls.manifest = build_release(
            cls.cfg,
            cls.out,
            name="test-release",
            version="0.1.0",
            creators=[{"name": "Doe, Jane", "affiliation": "Somewhere"}],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the temporary store."""
        cls._tmp.cleanup()

    def test_expected_files_written(self) -> None:
        """Every file of the release is present."""
        names = {p.name for p in self.out.iterdir()}
        self.assertTrue(
            {
                "catalogue.h5",
                "injections.h5",
                "events.yaml",
                "skymaps.tar",
                "config.yaml",
                "README.md",
                "SHA256SUMS",
                "MANIFEST.json",
                "zenodo_metadata.json",
            }
            <= names
        )

    def test_release_verifies(self) -> None:
        """A freshly built release passes its own verification."""
        report = verify_release(self.out)
        self.assertEqual(report["issues"], [])
        self.assertTrue(report["ok"])

    def test_catalogue_matches_store(self) -> None:
        """The catalogue has every galaxy of every tile, with tile ids."""
        with h5py.File(self.out / "catalogue.h5", "r") as fh:
            n = fh["catalogue/ra"].shape[0]
            tile_ids = fh["catalogue/tile_id"][()]
        self.assertEqual(n, self.manifest["counts"]["n_galaxies"])
        self.assertGreater(n, 0)
        self.assertEqual(set(tile_ids), set(range(self.cfg.n_tiles)))
        self.assertTrue(np.all(np.diff(tile_ids) >= 0))

    def test_injections_readable_as_injection_set(self) -> None:
        """/injections is the format InjectionSet.from_hdf5 reads."""
        injections = InjectionSet.from_hdf5(self.out / "injections.h5")
        self.assertEqual(len(injections), self.manifest["counts"]["n_injections"])
        self.assertGreater(len(injections), 0)

    def test_injection_detection_columns(self) -> None:
        """Detection results sit alongside each injection."""
        with h5py.File(self.out / "injections.h5", "r") as fh:
            grp = fh["injections"]
            n = grp["geocent_time"].shape[0]
            self.assertEqual(grp["network_snr"].shape, (n,))
            self.assertEqual(grp["active"].shape, (n, len(self.cfg.detectors)))
            self.assertEqual(
                list(grp["active"].attrs["detectors"]), list(self.cfg.detectors)
            )
            self.assertEqual(
                int(grp["detectable"][()].sum()),
                self.manifest["counts"]["n_detectable"],
            )
            names = [s.decode() for s in grp["event_name"][()]]
        self.assertEqual(len(set(names)), n)

    def test_skymaps_packed_and_named(self) -> None:
        """skymaps.tar holds one file per injection that has a skymap."""
        with h5py.File(self.out / "injections.h5", "r") as fh:
            listed = {s.decode() for s in fh["injections/skymap_file"][()] if s}
        with tarfile.open(self.out / "skymaps.tar") as tar:
            members = {m.name for m in tar.getmembers() if m.isfile()}
        self.assertEqual(members, listed)
        self.assertEqual(len(members), self.manifest["counts"]["n_skymaps"])
        self.assertGreater(len(members), 0)

    def test_events_use_relative_skymap_paths(self) -> None:
        """events.yaml carries no absolute paths into the generating machine."""
        docs = [
            d for d in yaml.safe_load_all((self.out / "events.yaml").read_text()) if d
        ]
        self.assertEqual(len(docs), self.manifest["counts"]["n_detectable"])
        with tarfile.open(self.out / "skymaps.tar") as tar:
            members = {m.name for m in tar.getmembers()}
        for doc in docs:
            self.assertFalse(Path(doc["localization file"]).is_absolute())
            self.assertIn(doc["localization file"], members)

    def test_catalogue_streamed_identically(self) -> None:
        """Writing tile by tile gives exactly the concatenated catalogue."""
        catalogue = load_consolidated_catalogue(self.cfg)
        with h5py.File(self.out / "catalogue.h5", "r") as fh:
            grp = fh["catalogue"]
            np.testing.assert_array_equal(grp["ra"][()], catalogue.ra)
            np.testing.assert_array_equal(grp["dec"][()], catalogue.dec)
            np.testing.assert_array_equal(grp["redshift"][()], catalogue.redshifts)
            np.testing.assert_array_equal(grp["luminosity"][()], catalogue.luminosities)
            np.testing.assert_array_equal(
                grp["apparent_magnitude"][()], catalogue.apparent_magnitudes
            )

    def test_readme_resolves_default_observation_window(self) -> None:
        """A config with no explicit window documents the O3 default, not None."""
        self.assertIsNone(self.cfg.t_start)
        readme = (self.out / "README.md").read_text()
        self.assertNotIn("None", readme)
        self.assertIn(f"GPS {GPS_O3_START} to {GPS_O3_END}", readme)

    def test_zenodo_metadata(self) -> None:
        """The upload metadata has the fields Zenodo requires."""
        meta = json.loads((self.out / "zenodo_metadata.json").read_text())
        self.assertEqual(meta["upload_type"], "dataset")
        self.assertEqual(meta["version"], "0.1.0")
        self.assertEqual(meta["creators"][0]["name"], "Doe, Jane")
        self.assertEqual(meta["license"], "cc-by-4.0")
        self.assertIn("test-release", meta["title"])

    def test_config_has_no_machine_paths(self) -> None:
        """The released config does not leak the generating machine's paths."""
        config = (self.out / "config.yaml").read_text()
        self.assertNotIn(self.cfg.store, config)
        self.assertNotIn(self.cfg.output_dir, config)

    def test_manifest_lists_every_file_with_checksum(self) -> None:
        """SHA256SUMS covers every file except itself and the manifest."""
        sums = (self.out / "SHA256SUMS").read_text().splitlines()
        covered = {line.split("  ")[1] for line in sums}
        on_disk = {
            p.name
            for p in self.out.iterdir()
            if p.is_file()
            and p.name not in {"SHA256SUMS", "MANIFEST.json", "zenodo_metadata.json"}
        }
        self.assertEqual(covered, on_disk)


class TestReleaseIntegrity(unittest.TestCase):
    """Verification catches damage; building refuses to clobber."""

    def setUp(self) -> None:
        """Generate a store and build a release from it."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = _run_pipeline(self._tmp.name)
        self.out = Path(self._tmp.name) / "release"
        build_release(self.cfg, self.out)

    def test_corrupted_file_detected(self) -> None:
        """Flipping a byte makes the checksum fail."""
        path = self.out / "catalogue.h5"
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        report = verify_release(self.out)
        self.assertFalse(report["ok"])
        self.assertTrue(any("checksum mismatch" in i for i in report["issues"]))

    def test_missing_file_detected(self) -> None:
        """A deleted file is reported."""
        (self.out / "events.yaml").unlink()
        report = verify_release(self.out)
        self.assertTrue(any("missing file" in i for i in report["issues"]))

    def test_wrong_host_index_detected(self) -> None:
        """If a host index points at the wrong galaxy the release is flagged."""
        with h5py.File(self.out / "injections.h5", "r+") as fh:
            host = fh["injections/host_galaxy_index"]
            hosted = np.flatnonzero(host[()] >= 0)
            if len(hosted) == 0:
                self.skipTest("this tiny store has no catalogued hosts")
            fh["injections/redshift"][hosted[0]] += 1.0
        report = verify_release(self.out)
        self.assertTrue(
            any("disagrees with the catalogue" in i for i in report["issues"])
        )

    def _rewrite_events(self, keep: int) -> None:
        """Truncate events.yaml and refresh its checksum, as a stale file would be."""
        path = self.out / "events.yaml"
        docs = [d for d in yaml.safe_load_all(path.read_text()) if d]
        with open(path, "w") as fh:
            yaml.safe_dump_all(docs[:keep], fh, sort_keys=False)
        lines = (self.out / "SHA256SUMS").read_text().splitlines()
        fixed = [
            f"{_sha256(path)}  events.yaml" if line.endswith("  events.yaml") else line
            for line in lines
        ]
        (self.out / "SHA256SUMS").write_text("\n".join(fixed) + "\n")

    def test_verify_flags_events_that_are_not_the_detectable_ones(self) -> None:
        """An events.yaml missing a detectable event fails the semantic check."""
        self._rewrite_events(keep=1)
        report = verify_release(self.out)
        self.assertFalse(report["ok"])
        self.assertTrue(any("events.yaml describes" in i for i in report["issues"]))

    def test_build_rejects_stale_blueprints(self) -> None:
        """A blueprints.yaml that omits a detectable event cannot be released."""
        path = Path(self.cfg.output_dir) / "blueprints.yaml"
        docs = [d for d in yaml.safe_load_all(path.read_text()) if d]
        with open(path, "w") as fh:
            yaml.safe_dump_all(docs[:-1], fh, sort_keys=False)
        with self.assertRaises(MDCValidationError) as ctx:
            build_release(self.cfg, self.out, overwrite=True)
        self.assertTrue(
            any("lacks" in i for i in ctx.exception.report["issues"]), ctx.exception
        )

    def test_build_rejects_missing_blueprints(self) -> None:
        """Detectable events with no blueprints file at all cannot be released."""
        (Path(self.cfg.output_dir) / "blueprints.yaml").unlink()
        with self.assertRaises(MDCValidationError):
            build_release(self.cfg, self.out, overwrite=True)

    def test_malformed_tar_is_reported_not_raised(self) -> None:
        """A truncated skymaps.tar is exactly what verification must diagnose."""
        (self.out / "skymaps.tar").write_bytes(b"not a tar")
        report = verify_release(self.out)
        self.assertFalse(report["ok"])
        self.assertTrue(any("cannot read skymaps.tar" in i for i in report["issues"]))

    def test_unreadable_hdf5_is_reported_not_raised(self) -> None:
        """A corrupt HDF5 table is reported as an issue."""
        (self.out / "injections.h5").write_bytes(b"garbage")
        report = verify_release(self.out)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("cannot read the HDF5 tables" in i for i in report["issues"])
        )

    def test_missing_manifest_is_reported_not_raised(self) -> None:
        """A directory with no manifest fails cleanly."""
        (self.out / "MANIFEST.json").unlink()
        report = verify_release(self.out)
        self.assertFalse(report["ok"])

    def test_refuses_non_empty_directory(self) -> None:
        """An existing release is not overwritten by accident."""
        with self.assertRaises(FileExistsError):
            build_release(self.cfg, self.out)

    def test_overwrite_replaces(self) -> None:
        """overwrite=True rebuilds in place."""
        build_release(self.cfg, self.out, overwrite=True)
        self.assertTrue(verify_release(self.out)["ok"])

    def test_missing_skymap_file_raises(self) -> None:
        """A skymap the detection stage recorded but that is gone is an error."""
        shutil.rmtree(Path(self.cfg.output_dir) / "skymaps")
        with self.assertRaises(MDCValidationError):
            build_release(self.cfg, self.out, overwrite=True)


class TestVerifyReleaseCli(unittest.TestCase):
    """``bagpuss mdc verify-release``."""

    def test_verify_cli(self) -> None:
        """The command reports OK for a good release and fails for a bad one."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _run_pipeline(tmp)
            out = Path(tmp) / "release"
            build_release(cfg, out)
            runner = CliRunner()
            good = runner.invoke(main, ["mdc", "verify-release", str(out)])
            self.assertEqual(good.exit_code, 0, msg=good.output)
            (out / "skymaps.tar").write_bytes(b"not a tar")
            bad = runner.invoke(main, ["mdc", "verify-release", str(out)])
            self.assertNotEqual(bad.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
