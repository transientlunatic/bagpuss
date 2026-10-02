"""Tests for bagpuss.mdc.config."""

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from bagpuss.mdc.config import _SCHEMA, MDCConfig, load_config


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

    def test_default_rate_density_positive(self) -> None:
        """The default merger-rate density is a positive Mpc^-3 yr^-1 value."""
        self.assertGreater(MDCConfig().rate_density, 0.0)

    def test_default_detector_network(self) -> None:
        """Default detector network is the two-detector O4 configuration."""
        self.assertEqual(
            MDCConfig().detectors,
            {
                "AdvancedLIGOHanford": "AdvancedLIGO_O4",
                "AdvancedLIGOLivingston": "AdvancedLIGO_O4",
            },
        )

    def test_default_duty_cycles(self) -> None:
        """Default duty cycles match the two-detector O4 network."""
        self.assertEqual(
            MDCConfig().duty_cycles,
            {"AdvancedLIGOHanford": 0.75, "AdvancedLIGOLivingston": 0.75},
        )

    def test_default_duty_cycle_mean_lock_duration(self) -> None:
        """Default mean lock duration is 8 hours."""
        self.assertEqual(MDCConfig().duty_cycle_mean_lock_duration, 28_800.0)

    def test_duty_cycles_default_is_independent_per_instance(self) -> None:
        """Mutating one instance's duty_cycles dict doesn't affect another's."""
        cfg = MDCConfig()
        cfg.duty_cycles["Virgo"] = 0.7
        self.assertNotIn("Virgo", MDCConfig().duty_cycles)

    def test_default_injection_channel(self) -> None:
        """Default injection channel is 'Injection'."""
        self.assertEqual(MDCConfig().injection_channel, "Injection")

    def test_default_snr_threshold(self) -> None:
        """Default detection threshold is the conventional network SNR of 8."""
        self.assertEqual(MDCConfig().snr_threshold, 8.0)

    def test_default_frame_generation_settings(self) -> None:
        """Default frame-generation settings match faketc's prior hardcoded values."""
        cfg = MDCConfig()
        self.assertEqual(cfg.f_ref, 20.0)
        self.assertEqual(cfg.frame_duration, 32.0)
        self.assertEqual(cfg.frame_sample_rate, 4096.0)
        self.assertEqual(cfg.frame_pre_signal_padding, 30.0)

    def test_default_chirp_mass_margin(self) -> None:
        """Default PE chirp-mass prior margin is 0.5."""
        self.assertEqual(MDCConfig().chirp_mass_margin, 0.5)

    def test_detectors_default_is_independent_per_instance(self) -> None:
        """Mutating one instance's detectors dict doesn't affect another's."""
        cfg = MDCConfig()
        cfg.detectors["Virgo"] = "AdVirgo_O4"
        self.assertNotIn("Virgo", MDCConfig().detectors)


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
        """Nested YAML values override the corresponding defaults."""
        path = self._write_yaml(
            "galaxies:\n"
            "  z_max: 3.0\n"
            "sharding:\n"
            "  n_ra_tiles: 8\n"
            "run:\n"
            "  master_seed: 42\n"
            "  output_dir: /somewhere/out\n"
        )
        try:
            cfg = load_config(path)
        finally:
            path.unlink()
        self.assertEqual(cfg.output_dir, "/somewhere/out")
        self.assertAlmostEqual(cfg.z_max, 3.0)
        self.assertEqual(cfg.n_ra_tiles, 8)
        self.assertEqual(cfg.master_seed, 42)

    def test_nested_section_overrides_apply(self) -> None:
        """Overrides apply correctly across every section of the schema."""
        path = self._write_yaml(
            "cosmology: Planck13\n"
            "galaxies:\n"
            "  z_max: 2.0\n"
            "  luminosity_function:\n"
            "    phi_star: 0.02\n"
            "  selection:\n"
            "    m_lim: 20.0\n"
            "population:\n"
            "  mass:\n"
            "    alpha: 4.0\n"
            "  spin:\n"
            "    a_max: 0.9\n"
            "  rate_density: 1.0e-8\n"
            "observation:\n"
            "  t_start: 100.0\n"
            "  t_end: 200.0\n"
            "detection:\n"
            "  network:\n"
            "    Virgo: AdVirgo_O4\n"
            "  duty_cycles:\n"
            "    Virgo: 0.7\n"
            "  duty_cycle_mean_lock_duration: 3600.0\n"
            "  channel: TestChannel\n"
            "  snr_threshold: 10.0\n"
            "estimation:\n"
            "  chirp_mass_margin: 0.25\n"
        )
        try:
            cfg = load_config(path)
        finally:
            path.unlink()
        self.assertEqual(cfg.cosmology, "Planck13")
        self.assertEqual(cfg.z_max, 2.0)
        self.assertAlmostEqual(cfg.phi_star, 0.02)
        self.assertEqual(cfg.m_lim, 20.0)
        self.assertEqual(cfg.mass_alpha, 4.0)
        self.assertEqual(cfg.spin_a_max, 0.9)
        self.assertAlmostEqual(cfg.rate_density, 1.0e-8)
        self.assertEqual(cfg.t_start, 100.0)
        self.assertEqual(cfg.t_end, 200.0)
        self.assertEqual(cfg.detectors, {"Virgo": "AdVirgo_O4"})
        self.assertEqual(cfg.duty_cycles, {"Virgo": 0.7})
        self.assertEqual(cfg.duty_cycle_mean_lock_duration, 3600.0)
        self.assertEqual(cfg.injection_channel, "TestChannel")
        self.assertEqual(cfg.snr_threshold, 10.0)
        self.assertEqual(cfg.chirp_mass_margin, 0.25)

    def test_unspecified_fields_keep_defaults(self) -> None:
        """Fields not present in the YAML file keep MDCConfig's defaults."""
        path = self._write_yaml("galaxies:\n  z_max: 3.0\n")
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

    def test_unknown_top_level_key_raises(self) -> None:
        """A top-level key that isn't a schema section raises ValueError."""
        path = self._write_yaml("cosmologyy: Planck18\n")
        try:
            with self.assertRaises(ValueError):
                load_config(path)
        finally:
            path.unlink()

    def test_unknown_nested_key_raises_with_dotted_path(self) -> None:
        """A typo'd nested key raises ValueError naming its full dotted path."""
        path = self._write_yaml("galaxies:\n  z_maxx: 3.0\n")
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
        finally:
            path.unlink()
        self.assertIn("galaxies.z_maxx", str(ctx.exception))

    def test_section_given_scalar_instead_of_mapping_raises(self) -> None:
        """A section that should be a mapping raises ValueError if given a scalar."""
        path = self._write_yaml("galaxies: 3.0\n")
        try:
            with self.assertRaises(ValueError):
                load_config(path)
        finally:
            path.unlink()


class TestSchemaCompleteness(unittest.TestCase):
    """_SCHEMA and MDCConfig's fields must not drift apart."""

    def _schema_leaves(self, schema: dict) -> list[str]:
        leaves: list[str] = []
        for value in schema.values():
            if isinstance(value, dict):
                leaves.extend(self._schema_leaves(value))
            else:
                leaves.append(value)
        return leaves

    def test_every_field_reachable_exactly_once(self) -> None:
        """Every MDCConfig field is exactly one _SCHEMA leaf, and vice versa."""
        field_names = {f.name for f in fields(MDCConfig)}
        leaves = self._schema_leaves(_SCHEMA)
        self.assertEqual(
            len(leaves), len(set(leaves)), "a field is reachable via >1 schema key"
        )
        self.assertEqual(field_names, set(leaves))
