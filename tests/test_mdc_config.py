"""Tests for bagpuss.mdc.config."""

import tempfile
import unittest
from pathlib import Path

from bagpuss.mdc.config import MDCConfig, load_config


class TestMDCConfigDefaults(unittest.TestCase):
    """MDCConfig's defaults match bagpuss's existing small-scale demo params."""

    def test_default_cosmology(self) -> None:
        """Default cosmology is Planck18."""
        self.assertEqual(MDCConfig().cosmology, "Planck18")

    def test_default_phi_star(self) -> None:
        """Default phi_star matches the B-band Schechter normalisation."""
        self.assertAlmostEqual(MDCConfig().phi_star, 1.61e-2)

    def test_default_single_tile(self) -> None:
        """Defaults describe a single, unsharded tile/shard run."""
        cfg = MDCConfig()
        self.assertEqual(cfg.n_ra_tiles, 1)
        self.assertEqual(cfg.n_dec_tiles, 1)
        self.assertEqual(cfg.n_injection_shards, 1)

    def test_default_d_max_is_none(self) -> None:
        """No detectability filter is applied by default."""
        self.assertIsNone(MDCConfig().d_max)


class TestMDCConfigValidation(unittest.TestCase):
    """MDCConfig rejects invalid sharding parameters at construction time."""

    def test_zero_ra_tiles_rejected(self) -> None:
        """n_ra_tiles = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            MDCConfig(n_ra_tiles=0)

    def test_zero_dec_tiles_rejected(self) -> None:
        """n_dec_tiles = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            MDCConfig(n_dec_tiles=0)

    def test_zero_injection_shards_rejected(self) -> None:
        """n_injection_shards = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            MDCConfig(n_injection_shards=0)


class TestMDCConfigNTiles(unittest.TestCase):
    """n_tiles is the product of n_ra_tiles and n_dec_tiles."""

    def test_n_tiles_product(self) -> None:
        """n_tiles multiplies the two axis tile counts."""
        cfg = MDCConfig(n_ra_tiles=4, n_dec_tiles=3)
        self.assertEqual(cfg.n_tiles, 12)

    def test_n_tiles_default(self) -> None:
        """Default configuration has exactly one tile."""
        self.assertEqual(MDCConfig().n_tiles, 1)


class TestLoadConfig(unittest.TestCase):
    """Tests for load_config's YAML-over-defaults merging."""

    def _write_yaml(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=tempfile.gettempdir()
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_overrides_apply(self) -> None:
        """YAML values override the corresponding defaults."""
        path = self._write_yaml("z_max: 3.0\nn_ra_tiles: 8\nmaster_seed: 42\n")
        try:
            cfg = load_config(path)
        finally:
            path.unlink()
        self.assertAlmostEqual(cfg.z_max, 3.0)
        self.assertEqual(cfg.n_ra_tiles, 8)
        self.assertEqual(cfg.master_seed, 42)

    def test_unspecified_fields_keep_defaults(self) -> None:
        """Fields not present in the YAML file keep MDCConfig's defaults."""
        path = self._write_yaml("z_max: 3.0\n")
        try:
            cfg = load_config(path)
        finally:
            path.unlink()
        self.assertAlmostEqual(cfg.phi_star, MDCConfig().phi_star)

    def test_empty_file_gives_defaults(self) -> None:
        """An empty YAML file loads as MDCConfig()."""
        path = self._write_yaml("")
        try:
            cfg = load_config(path)
        finally:
            path.unlink()
        self.assertEqual(cfg, MDCConfig())

    def test_unknown_key_raises(self) -> None:
        """A key that isn't an MDCConfig field raises ValueError."""
        path = self._write_yaml("z_maxx: 3.0\n")
        try:
            with self.assertRaises(ValueError):
                load_config(path)
        finally:
            path.unlink()
