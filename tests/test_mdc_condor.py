"""Tests for bagpuss.mdc.condor."""

import tempfile
import unittest
from pathlib import Path

from bagpuss.mdc.condor import CondorResources, write_dag
from bagpuss.mdc.config import MDCConfig


class TestWriteDag(unittest.TestCase):
    """Tests for write_dag's rendered DAG and submit-file output."""

    def _write(self, cfg: MDCConfig, tmp: str) -> tuple[Path, Path]:
        out_dir = Path(tmp) / "dag"
        dag_path = write_dag(cfg, "config.yaml", out_dir, "/opt/venv/bin/bagpuss")
        return dag_path, out_dir

    def test_dag_file_written(self) -> None:
        """write_dag writes mdc.dag and returns its path."""
        cfg = MDCConfig(n_ra_tiles=2, n_dec_tiles=1, n_injection_shards=2)
        with tempfile.TemporaryDirectory() as tmp:
            dag_path, _ = self._write(cfg, tmp)
            self.assertTrue(dag_path.exists())
            self.assertEqual(dag_path.name, "mdc.dag")

    def test_all_submit_files_written(self) -> None:
        """One .sub file is written per pipeline stage."""
        cfg = MDCConfig(n_ra_tiles=2, n_dec_tiles=1, n_injection_shards=2)
        with tempfile.TemporaryDirectory() as tmp:
            _, out_dir = self._write(cfg, tmp)
            for stage in (
                "catalogue_tile",
                "consolidate_catalogue",
                "injection_shard",
                "assemble_injections",
                "package_manifest",
            ):
                self.assertTrue((out_dir / f"{stage}.sub").exists(), stage)

    def test_logs_directory_created(self) -> None:
        """A logs/ subdirectory is created alongside the DAG."""
        cfg = MDCConfig()
        with tempfile.TemporaryDirectory() as tmp:
            _, out_dir = self._write(cfg, tmp)
            self.assertTrue((out_dir / "logs").is_dir())

    def test_one_job_per_tile_and_shard(self) -> None:
        """The DAG has exactly n_tiles tile jobs and n_injection_shards shard jobs."""
        cfg = MDCConfig(n_ra_tiles=3, n_dec_tiles=2, n_injection_shards=4)
        with tempfile.TemporaryDirectory() as tmp:
            dag_path, _ = self._write(cfg, tmp)
            text = dag_path.read_text()
        self.assertEqual(text.count("JOB catalogue_tile_"), 6)
        self.assertEqual(text.count("JOB injection_shard_"), 4)
        self.assertEqual(text.count("JOB consolidate_catalogue "), 1)
        self.assertEqual(text.count("JOB assemble_injections "), 1)
        self.assertEqual(text.count("JOB package_manifest "), 1)

    def test_tile_jobs_are_parents_of_consolidate(self) -> None:
        """Every tile job appears in consolidate_catalogue's PARENT line."""
        cfg = MDCConfig(n_ra_tiles=2, n_dec_tiles=1)
        with tempfile.TemporaryDirectory() as tmp:
            dag_path, _ = self._write(cfg, tmp)
            text = dag_path.read_text()
        parent_line = next(
            line
            for line in text.splitlines()
            if line.startswith("PARENT catalogue_tile")
        )
        self.assertIn("catalogue_tile_0000", parent_line)
        self.assertIn("catalogue_tile_0001", parent_line)
        self.assertIn("CHILD consolidate_catalogue", parent_line)

    def test_dag_chain_ends_in_package_manifest(self) -> None:
        """package_manifest is the final DAG node, parented by assemble_injections."""
        cfg = MDCConfig()
        with tempfile.TemporaryDirectory() as tmp:
            dag_path, _ = self._write(cfg, tmp)
            text = dag_path.read_text()
        self.assertIn("PARENT assemble_injections CHILD package_manifest", text)

    def test_per_job_log_paths_are_unique(self) -> None:
        """Tile/shard submit files interpolate the per-job condor macro in logs."""
        cfg = MDCConfig(n_ra_tiles=2, n_dec_tiles=1, n_injection_shards=2)
        with tempfile.TemporaryDirectory() as tmp:
            _, out_dir = self._write(cfg, tmp)
            tile_sub = (out_dir / "catalogue_tile.sub").read_text()
            shard_sub = (out_dir / "injection_shard.sub").read_text()
        self.assertIn("$(tile_id).log", tile_sub)
        self.assertIn("$(shard_id).log", shard_sub)

    def test_executable_and_config_path_used(self) -> None:
        """Submit files reference the given executable and config path."""
        cfg = MDCConfig()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dag"
            write_dag(cfg, "/some/run.yaml", out_dir, "/opt/venv/bin/bagpuss")
            text = (out_dir / "catalogue_tile.sub").read_text()
        self.assertIn("executable              = /opt/venv/bin/bagpuss", text)
        self.assertIn("/some/run.yaml", text)

    def test_resource_overrides_applied(self) -> None:
        """A per-stage resource override is reflected in its submit file."""
        cfg = MDCConfig()
        overrides = {"catalogue_tile": CondorResources(request_memory="64GB")}
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "dag"
            write_dag(
                cfg,
                "config.yaml",
                out_dir,
                "/opt/venv/bin/bagpuss",
                resources=overrides,
            )
            text = (out_dir / "catalogue_tile.sub").read_text()
        memory_line = next(
            line for line in text.splitlines() if line.startswith("request_memory")
        )
        self.assertEqual(memory_line.split("=")[1].strip(), "64GB")

    def test_retry_present_for_every_job(self) -> None:
        """Every JOB line has a matching RETRY line."""
        cfg = MDCConfig(n_ra_tiles=2, n_dec_tiles=1, n_injection_shards=2)
        with tempfile.TemporaryDirectory() as tmp:
            dag_path, _ = self._write(cfg, tmp)
            text = dag_path.read_text()
        job_names = [
            line.split()[1] for line in text.splitlines() if line.startswith("JOB ")
        ]
        for name in job_names:
            self.assertIn(f"RETRY {name} ", text)
