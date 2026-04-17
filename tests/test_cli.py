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
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["cosmology"], "Planck18")

    def test_default_structure(self) -> None:
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["structure"], "point_process")

    def test_default_luminosity(self) -> None:
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["luminosity"], "schechter")

    def test_default_n_galaxies(self) -> None:
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["n_galaxies"], 10000)

    def test_default_seed_is_none(self) -> None:
        cfg = _merge_config(None, {})
        self.assertIsNone(cfg["seed"])

    def test_default_z_max(self) -> None:
        cfg = _merge_config(None, {})
        self.assertAlmostEqual(cfg["z_max"], 1.0)

    def test_default_bins(self) -> None:
        cfg = _merge_config(None, {})
        self.assertEqual(cfg["bins"], 50)


class TestMergeConfigYamlOverrides(unittest.TestCase):
    """YAML file values override defaults."""

    def _write_yaml(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_yaml_cosmology(self) -> None:
        p = self._write_yaml("model:\n  cosmology: WMAP9\n")
        cfg = _merge_config(p, {})
        self.assertEqual(cfg["cosmology"], "WMAP9")

    def test_yaml_n_galaxies(self) -> None:
        p = self._write_yaml("model:\n  n_galaxies: 500\n")
        cfg = _merge_config(p, {})
        self.assertEqual(cfg["n_galaxies"], 500)

    def test_yaml_z_max(self) -> None:
        p = self._write_yaml("structure:\n  point_process:\n    z_max: 2.5\n")
        cfg = _merge_config(p, {})
        self.assertAlmostEqual(cfg["z_max"], 2.5)

    def test_yaml_bins(self) -> None:
        p = self._write_yaml("plot:\n  bins: 20\n")
        cfg = _merge_config(p, {})
        self.assertEqual(cfg["bins"], 20)

    def test_yaml_schechter_params(self) -> None:
        p = self._write_yaml("luminosity:\n  schechter:\n    alpha: -1.5\n")
        cfg = _merge_config(p, {})
        self.assertAlmostEqual(cfg["alpha"], -1.5)


class TestMergeConfigCliOverrides(unittest.TestCase):
    """CLI overrides take precedence over YAML values."""

    def _write_yaml(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_cli_overrides_yaml_cosmology(self) -> None:
        p = self._write_yaml("model:\n  cosmology: WMAP9\n")
        cfg = _merge_config(p, {"cosmology": "Planck13"})
        self.assertEqual(cfg["cosmology"], "Planck13")

    def test_cli_none_does_not_override_yaml(self) -> None:
        p = self._write_yaml("model:\n  n_galaxies: 999\n")
        cfg = _merge_config(p, {"n_galaxies": None})
        self.assertEqual(cfg["n_galaxies"], 999)

    def test_cli_none_does_not_override_defaults(self) -> None:
        cfg = _merge_config(None, {"bins": None})
        self.assertEqual(cfg["bins"], 50)


class TestHeatmapCommand(unittest.TestCase):
    """Integration tests for ``bagpuss plot heatmap``."""

    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_help_exits_zero(self) -> None:
        result = self.runner.invoke(main, ["plot", "heatmap", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--output", result.output)

    def test_save_to_file(self) -> None:
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
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                main,
                ["plot", "heatmap", "--n-galaxies", "50", "--seed", "1", "--output", "out.png"],
            )
            self.assertIn("out.png", result.output)

    def test_invalid_cosmology_exits_nonzero(self) -> None:
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                main,
                ["plot", "heatmap", "--cosmology", "NotACosmology", "--output", "x.png"],
            )
            self.assertNotEqual(result.exit_code, 0)

    def test_nonexistent_config_exits_nonzero(self) -> None:
        result = self.runner.invoke(
            main,
            ["plot", "heatmap", "--config", "/nonexistent/path.yaml", "--output", "x.png"],
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_yaml_config_file(self) -> None:
        with self.runner.isolated_filesystem():
            Path("cfg.yaml").write_text(
                "model:\n  n_galaxies: 50\n  seed: 1\nstructure:\n  point_process:\n    z_max: 0.3\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                main,
                ["plot", "heatmap", "--config", "cfg.yaml", "--output", "out.png"],
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path("out.png").exists())

    def test_cli_seed_overrides_yaml_seed(self) -> None:
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
