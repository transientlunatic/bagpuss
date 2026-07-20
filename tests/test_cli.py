"""Tests for bagpuss.cli."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from click.testing import CliRunner  # noqa: E402

from bagpuss.cli import _merge_config, main  # noqa: E402


class TestMergeConfigDefaults(unittest.TestCase):
    """_merge_config returns sensible defaults with no file or CLI flags."""

    def test_default_cosmology(self) -> None:
        """Default cosmology is Planck18."""
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["cosmology"], "Planck18")

    def test_default_structure(self) -> None:
        """Default structure model is point_process."""
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["structure"], "point_process")

    def test_default_luminosity(self) -> None:
        """Default luminosity model is schechter."""
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["luminosity"], "schechter")

    def test_default_n_galaxies(self) -> None:
        """Default galaxy count is 10 000."""
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["n_galaxies"], 10000)

    def test_default_seed_is_none(self) -> None:
        """Default RNG seed is None (non-reproducible)."""
        cfg = _merge_config(None, {})
        self.assertIsNone(cfg["seed"])

    def test_default_z_max(self) -> None:
        """Default maximum redshift is 1.0."""
        cfg = _merge_config(None, {})
        self.assertAlmostEqual(cfg["z_max"], 1.0)

    def test_default_bins(self) -> None:
        """Default histogram bin count is 50."""
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["bins"], 50)


class TestMergeConfigYamlOverrides(unittest.TestCase):
    """YAML file values override defaults."""

    def _write_yaml(self, content: str) -> Path:
        """Write *content* to a temporary YAML file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_yaml_cosmology(self) -> None:
        """YAML model.cosmology overrides the default."""
        p = self._write_yaml("model:\n  cosmology: WMAP9\n")
        cfg = _merge_config(p, {})
        self.assertEqual(cfg["cosmology"], "WMAP9")

    def test_yaml_n_galaxies(self) -> None:
        """YAML model.n_galaxies overrides the default."""
        p = self._write_yaml("model:\n  n_galaxies: 500\n")
        cfg = _merge_config(p, {})
        self.assertEqual(cfg["n_galaxies"], 500)

    def test_yaml_z_max(self) -> None:
        """YAML structure.point_process.z_max overrides the default."""
        p = self._write_yaml("structure:\n  point_process:\n    z_max: 2.5\n")
        cfg = _merge_config(p, {})
        self.assertAlmostEqual(cfg["z_max"], 2.5)

    def test_yaml_bins(self) -> None:
        """YAML plot.bins overrides the default."""
        p = self._write_yaml("plot:\n  bins: 20\n")
        cfg = _merge_config(p, {})
        self.assertEqual(cfg["bins"], 20)

    def test_yaml_schechter_params(self) -> None:
        """YAML luminosity.schechter.alpha overrides the default."""
        p = self._write_yaml("luminosity:\n  schechter:\n    alpha: -1.5\n")
        cfg = _merge_config(p, {})
        self.assertAlmostEqual(cfg["alpha"], -1.5)


class TestMergeConfigCliOverrides(unittest.TestCase):
    """CLI overrides take precedence over YAML values."""

    def _write_yaml(self, content: str) -> Path:
        """Write *content* to a temporary YAML file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_cli_overrides_yaml_cosmology(self) -> None:
        """A CLI cosmology value beats the YAML value."""
        p = self._write_yaml("model:\n  cosmology: WMAP9\n")
        cfg = _merge_config(p, {"cosmology": "Planck13"})
        self.assertEqual(cfg["cosmology"], "Planck13")

    def test_cli_none_does_not_override_yaml(self) -> None:
        """A CLI value of None leaves the YAML value in place."""
        p = self._write_yaml("model:\n  n_galaxies: 999\n")
        cfg = _merge_config(p, {"n_galaxies": None})
        self.assertEqual(cfg["n_galaxies"], 999)

    def test_cli_none_does_not_override_defaults(self) -> None:
        """A CLI value of None leaves the default in place."""
        cfg = _merge_config(None, {"bins": None})
        self.assertEqual(cfg["bins"], 50)


class TestHeatmapCommand(unittest.TestCase):
    """Integration tests for ``bagpuss plot heatmap``."""

    def setUp(self) -> None:
        """Create a shared CliRunner."""
        self.runner = CliRunner()

    def test_help_exits_zero(self) -> None:
        """``--help`` exits with code 0 and mentions --output."""
        result = self.runner.invoke(main, ["plot", "heatmap", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--output", result.output)

    def test_save_to_file(self) -> None:
        """Running with --output creates the output file."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                main,
                [
                    "plot",
                    "heatmap",
                    "--n-galaxies",
                    "100",
                    "--seed",
                    "7",
                    "--z-max",
                    "0.5",
                    "--output",
                    "out.png",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path("out.png").exists())

    def test_save_echoes_path(self) -> None:
        """A success message containing the output path is printed."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                main,
                [
                    "plot",
                    "heatmap",
                    "--n-galaxies",
                    "50",
                    "--seed",
                    "1",
                    "--output",
                    "out.png",
                ],
            )
            self.assertIn("out.png", result.output)

    def test_invalid_cosmology_exits_nonzero(self) -> None:
        """An unknown cosmology name causes a non-zero exit."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                main,
                [
                    "plot",
                    "heatmap",
                    "--cosmology",
                    "NotACosmology",
                    "--output",
                    "x.png",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)

    def test_nonexistent_config_exits_nonzero(self) -> None:
        """A missing --config path is rejected by click before running."""
        result = self.runner.invoke(
            main,
            [
                "plot",
                "heatmap",
                "--config",
                "/nonexistent/path.yaml",
                "--output",
                "x.png",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_yaml_config_file(self) -> None:
        """A YAML config file drives the full run successfully."""
        with self.runner.isolated_filesystem():
            Path("cfg.yaml").write_text(
                "model:\n  n_galaxies: 50\n  seed: 1\n"
                "structure:\n  point_process:\n    z_max: 0.3\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                main,
                ["plot", "heatmap", "--config", "cfg.yaml", "--output", "out.png"],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path("out.png").exists())

    def test_cli_seed_overrides_yaml_seed(self) -> None:
        """A CLI --seed beats the seed set in the YAML config."""
        with self.runner.isolated_filesystem():
            Path("cfg.yaml").write_text("model:\n  seed: 99\n", encoding="utf-8")
            result = self.runner.invoke(
                main,
                [
                    "plot",
                    "heatmap",
                    "--config",
                    "cfg.yaml",
                    "--seed",
                    "0",
                    "--n-galaxies",
                    "50",
                    "--output",
                    "out.png",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)


# ---------------------------------------------------------------------------
# mdc command group
# ---------------------------------------------------------------------------

_TINY_MDC_CONFIG = (
    "z_max: 0.1\nphi_star: 1.0e-5\nm_lim: 25.0\nn_ra_tiles: 2\nn_dec_tiles: 1\n"
    "n_injection_shards: 2\nn_draw_per_shard: 20\nd_max: 5000.0\nstore: store.zarr\n"
)


class TestMdcCommands(unittest.TestCase):
    """Integration tests for the ``bagpuss mdc`` command group."""

    def setUp(self) -> None:
        """Create a shared CliRunner."""
        self.runner = CliRunner()

    def test_help_exits_zero(self) -> None:
        """``bagpuss mdc --help`` exits 0 and lists every subcommand."""
        result = self.runner.invoke(main, ["mdc", "--help"])
        self.assertEqual(result.exit_code, 0)
        for name in (
            "generate-tile",
            "consolidate",
            "generate-injections",
            "assemble-injections",
            "package",
            "make-dag",
        ):
            self.assertIn(name, result.output)

    def test_full_pipeline_via_cli(self) -> None:
        """The full tile->consolidate->shard->assemble->package chain succeeds."""
        with self.runner.isolated_filesystem():
            Path("config.yaml").write_text(_TINY_MDC_CONFIG, encoding="utf-8")

            for tile_id in (0, 1):
                result = self.runner.invoke(
                    main,
                    [
                        "mdc",
                        "generate-tile",
                        "--config",
                        "config.yaml",
                        "--tile-id",
                        str(tile_id),
                    ],
                )
                self.assertEqual(result.exit_code, 0, msg=result.output)

            result = self.runner.invoke(
                main, ["mdc", "consolidate", "--config", "config.yaml"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)

            for shard_id in (0, 1):
                result = self.runner.invoke(
                    main,
                    [
                        "mdc",
                        "generate-injections",
                        "--config",
                        "config.yaml",
                        "--shard-id",
                        str(shard_id),
                    ],
                )
                self.assertEqual(result.exit_code, 0, msg=result.output)

            result = self.runner.invoke(
                main, ["mdc", "assemble-injections", "--config", "config.yaml"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("2 shards", result.output)

            result = self.runner.invoke(
                main, ["mdc", "package", "--config", "config.yaml"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path("store.zarr/manifest.json").exists())

    def test_consolidate_missing_tile_exits_nonzero(self) -> None:
        """Consolidating with a tile missing exits non-zero and reports why."""
        with self.runner.isolated_filesystem():
            Path("config.yaml").write_text(_TINY_MDC_CONFIG, encoding="utf-8")
            result = self.runner.invoke(
                main,
                ["mdc", "generate-tile", "--config", "config.yaml", "--tile-id", "0"],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)

            result = self.runner.invoke(
                main, ["mdc", "consolidate", "--config", "config.yaml"]
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("FAILED", result.output)

    def test_make_dag(self) -> None:
        """``make-dag`` renders a DAG without touching condor."""
        with self.runner.isolated_filesystem():
            Path("config.yaml").write_text(_TINY_MDC_CONFIG, encoding="utf-8")
            result = self.runner.invoke(
                main,
                [
                    "mdc",
                    "make-dag",
                    "--config",
                    "config.yaml",
                    "--out-dir",
                    "dag",
                    "--bagpuss-executable",
                    "/opt/venv/bin/bagpuss",
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path("dag/mdc.dag").exists())

    def test_generate_tile_requires_config(self) -> None:
        """Omitting --config is rejected by click before running."""
        result = self.runner.invoke(main, ["mdc", "generate-tile", "--tile-id", "0"])
        self.assertNotEqual(result.exit_code, 0)
