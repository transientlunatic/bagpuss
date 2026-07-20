"""Tests for bagpuss.mdc.pipeline."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from bagpuss.mdc.config import MDCConfig
from bagpuss.mdc.pipeline import (
    MDCValidationError,
    assemble_injections,
    build_detectable,
    build_population,
    build_selection,
    build_tile_universe,
    build_universe,
    consolidate_catalogue,
    generate_catalogue_tile,
    generate_injection_shard,
    load_consolidated_catalogue,
    package_manifest,
    shard_rng,
    tile_bounds,
)
from bagpuss.universe import SkyPatch

# A tiny, fast-drawing configuration: small z_max and a scaled-down phi_star
# (per the project's existing test convention for build_catalogue) so tile
# draws are O(1000) galaxies rather than millions.
_TINY_KWARGS: dict[str, float | int | str | None] = {
    "z_max": 0.1,
    "phi_star": 1.0e-5,
    "m_lim": 25.0,
}


def _tiny_config(**overrides: float | int | str | None) -> MDCConfig:
    kwargs = dict(_TINY_KWARGS)
    kwargs.update(overrides)
    return MDCConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# tile_bounds
# ---------------------------------------------------------------------------


class TestTileBounds(unittest.TestCase):
    """Tests for the tile_id -> (ra_range, sin_dec_range) grid mapping."""

    def test_single_tile_covers_full_sky(self) -> None:
        """With one tile, its bounds cover the full sky."""
        cfg = _tiny_config(n_ra_tiles=1, n_dec_tiles=1)
        ra_range, sin_dec_range = tile_bounds(cfg, 0)
        self.assertEqual(ra_range, (0.0, 2.0 * np.pi))
        self.assertEqual(sin_dec_range, (-1.0, 1.0))

    def test_grid_tiles_partition_sky(self) -> None:
        """A 2x2 grid's tiles exactly tile [0, 2pi) x [-1, 1] with no gaps."""
        cfg = _tiny_config(n_ra_tiles=2, n_dec_tiles=2)
        bounds = [tile_bounds(cfg, i) for i in range(cfg.n_tiles)]
        ra_ranges = sorted({b[0] for b in bounds})
        u_ranges = sorted({b[1] for b in bounds})
        self.assertEqual(ra_ranges, [(0.0, np.pi), (np.pi, 2.0 * np.pi)])
        self.assertEqual(u_ranges, [(-1.0, 0.0), (0.0, 1.0)])

    def test_out_of_range_tile_id_raises(self) -> None:
        """A tile_id outside [0, n_tiles) raises ValueError."""
        cfg = _tiny_config(n_ra_tiles=2, n_dec_tiles=2)
        with self.assertRaises(ValueError):
            tile_bounds(cfg, 4)

    def test_negative_tile_id_raises(self) -> None:
        """A negative tile_id raises ValueError."""
        cfg = _tiny_config()
        with self.assertRaises(ValueError):
            tile_bounds(cfg, -1)


# ---------------------------------------------------------------------------
# shard_rng
# ---------------------------------------------------------------------------


class TestShardRNG(unittest.TestCase):
    """Tests for deterministic, independent per-shard RNG streams."""

    def test_reproducible(self) -> None:
        """The same (config, kind, shard_id) always gives the same stream."""
        cfg = _tiny_config(master_seed=99)
        a = shard_rng(cfg, "tile", 3).uniform(size=5)
        b = shard_rng(cfg, "tile", 3).uniform(size=5)
        np.testing.assert_array_equal(a, b)

    def test_different_shard_ids_differ(self) -> None:
        """Different shard ids give different streams."""
        cfg = _tiny_config(master_seed=99)
        a = shard_rng(cfg, "tile", 0).uniform(size=5)
        b = shard_rng(cfg, "tile", 1).uniform(size=5)
        self.assertFalse(np.array_equal(a, b))

    def test_different_kinds_differ(self) -> None:
        """The tile and injection streams differ even for the same shard id."""
        cfg = _tiny_config(master_seed=99)
        a = shard_rng(cfg, "tile", 0).uniform(size=5)
        b = shard_rng(cfg, "injection", 0).uniform(size=5)
        self.assertFalse(np.array_equal(a, b))

    def test_different_master_seed_differs(self) -> None:
        """A different master_seed gives a different stream."""
        a = shard_rng(_tiny_config(master_seed=1), "tile", 0).uniform(size=5)
        b = shard_rng(_tiny_config(master_seed=2), "tile", 0).uniform(size=5)
        self.assertFalse(np.array_equal(a, b))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


class TestModelBuilders(unittest.TestCase):
    """Tests for the config -> bagpuss model object builders."""

    def test_build_universe_uses_config_z_max(self) -> None:
        """build_universe's structure model uses config.z_max."""
        cfg = _tiny_config(z_max=2.5)
        universe = build_universe(cfg)
        self.assertAlmostEqual(universe.structure.z_max, 2.5)  # type: ignore[attr-defined]

    def test_build_tile_universe_wraps_in_sky_patch(self) -> None:
        """build_tile_universe wraps the structure in a SkyPatch."""
        cfg = _tiny_config(n_ra_tiles=2, n_dec_tiles=1)
        universe = build_tile_universe(cfg, 0)
        self.assertIsInstance(universe.structure, SkyPatch)

    def test_build_tile_universe_volume_scaled(self) -> None:
        """A tile universe's survey volume is the full volume / n_tiles."""
        cfg = _tiny_config(n_ra_tiles=2, n_dec_tiles=2)
        full = build_universe(cfg)
        tile = build_tile_universe(cfg, 0)
        full_volume = full.structure.survey_volume(full.cosmology)
        tile_volume = tile.structure.survey_volume(tile.cosmology)
        self.assertAlmostEqual(tile_volume, full_volume / 4.0)

    def test_build_selection_uses_config_m_lim(self) -> None:
        """build_selection uses config.m_lim."""
        cfg = _tiny_config(m_lim=21.0)
        self.assertAlmostEqual(build_selection(cfg).m_lim, 21.0)

    def test_build_population_returns_nonempty_sample(self) -> None:
        """build_population produces a usable PopulationModel."""
        cfg = _tiny_config()
        population = build_population(cfg)
        bbh = population.sample(10, rng=np.random.default_rng(0))
        self.assertEqual(len(bbh), 10)

    def test_build_detectable_none_when_d_max_unset(self) -> None:
        """build_detectable returns None when d_max is not configured."""
        cfg = _tiny_config(d_max=None)
        self.assertIsNone(build_detectable(cfg))

    def test_build_detectable_present_when_d_max_set(self) -> None:
        """build_detectable returns a Detectable when d_max is configured."""
        cfg = _tiny_config(d_max=1000.0)
        detectable = build_detectable(cfg)
        self.assertIsNotNone(detectable)


# ---------------------------------------------------------------------------
# End-to-end pipeline (small, on a temp directory zarr store)
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd(unittest.TestCase):
    """Runs the full tile -> consolidate -> shard -> assemble -> package chain."""

    def _run_full_pipeline(self, cfg: MDCConfig) -> dict[str, Any]:
        for tile_id in range(cfg.n_tiles):
            generate_catalogue_tile(cfg, tile_id)
        catalogue_report = consolidate_catalogue(cfg)
        for shard_id in range(cfg.n_injection_shards):
            generate_injection_shard(cfg, shard_id)
        injection_report = assemble_injections(cfg)
        manifest = package_manifest(cfg)
        return {
            "catalogue_report": catalogue_report,
            "injection_report": injection_report,
            "manifest": manifest,
        }

    def test_full_pipeline_succeeds(self) -> None:
        """A small multi-tile, multi-shard run completes and validates clean."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=2,
                n_dec_tiles=2,
                n_injection_shards=3,
                n_draw_per_shard=30,
                d_max=5000.0,
                store=str(Path(tmp) / "store.zarr"),
            )
            result = self._run_full_pipeline(cfg)

        self.assertTrue(result["catalogue_report"]["ok"])
        self.assertEqual(result["catalogue_report"]["n_tiles"], 4)
        self.assertGreater(result["catalogue_report"]["total_galaxies"], 0)

        self.assertTrue(result["injection_report"]["ok"])
        self.assertEqual(result["injection_report"]["n_shards"], 3)
        self.assertEqual(result["injection_report"]["total_injections"], 90)

        self.assertEqual(
            result["manifest"]["total_galaxies"],
            result["catalogue_report"]["total_galaxies"],
        )
        self.assertEqual(result["manifest"]["total_injections"], 90)

    def test_load_consolidated_catalogue_matches_report_count(self) -> None:
        """load_consolidated_catalogue's length matches the consolidation report."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=2, n_dec_tiles=1, store=str(Path(tmp) / "store.zarr")
            )
            for tile_id in range(cfg.n_tiles):
                generate_catalogue_tile(cfg, tile_id)
            report = consolidate_catalogue(cfg)
            catalogue = load_consolidated_catalogue(cfg)

        self.assertEqual(len(catalogue), report["total_galaxies"])

    def test_injections_have_valid_host_indices(self) -> None:
        """Every injection's host_galaxy_index is -1 or a valid catalogue row."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=1,
                n_dec_tiles=1,
                n_injection_shards=1,
                n_draw_per_shard=50,
                store=str(Path(tmp) / "store.zarr"),
            )
            generate_catalogue_tile(cfg, 0)
            consolidate_catalogue(cfg)
            injections = generate_injection_shard(cfg, 0)
            n_catalogue = len(load_consolidated_catalogue(cfg))

        valid = (injections.host_galaxy_index == -1) | (
            (injections.host_galaxy_index >= 0)
            & (injections.host_galaxy_index < n_catalogue)
        )
        self.assertTrue(bool(np.all(valid)))

    def test_reproducible_across_full_reruns(self) -> None:
        """Re-running the whole pipeline with the same seed reproduces it exactly."""
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = _tiny_config(
                    n_ra_tiles=2,
                    n_dec_tiles=1,
                    n_injection_shards=2,
                    n_draw_per_shard=20,
                    master_seed=123,
                    store=str(Path(tmp) / "store.zarr"),
                )
                for tile_id in range(cfg.n_tiles):
                    generate_catalogue_tile(cfg, tile_id)
                consolidate_catalogue(cfg)
                for shard_id in range(cfg.n_injection_shards):
                    generate_injection_shard(cfg, shard_id)
                catalogue = load_consolidated_catalogue(cfg)
                results.append(catalogue.redshifts.copy())

        np.testing.assert_array_equal(results[0], results[1])


class TestConsolidateCatalogueValidation(unittest.TestCase):
    """Tests for consolidate_catalogue's sanity-check gate."""

    def test_missing_tile_fails(self) -> None:
        """Consolidating with a tile missing raises MDCValidationError."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=2, n_dec_tiles=1, store=str(Path(tmp) / "store.zarr")
            )
            generate_catalogue_tile(cfg, 0)
            # tile 1 deliberately not generated
            with self.assertRaises(MDCValidationError) as ctx:
                consolidate_catalogue(cfg)
            self.assertIn("missing tile", ctx.exception.report["issues"][0])

    def test_missing_tile_no_raise_when_disabled(self) -> None:
        """raise_on_failure=False returns a failing report instead of raising."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=2, n_dec_tiles=1, store=str(Path(tmp) / "store.zarr")
            )
            generate_catalogue_tile(cfg, 0)
            report = consolidate_catalogue(cfg, raise_on_failure=False)
        self.assertFalse(report["ok"])


class TestAssembleInjectionsValidation(unittest.TestCase):
    """Tests for assemble_injections's sanity-check gate."""

    def test_missing_shard_fails(self) -> None:
        """Assembling with a shard missing raises MDCValidationError."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=1,
                n_dec_tiles=1,
                n_injection_shards=2,
                n_draw_per_shard=20,
                store=str(Path(tmp) / "store.zarr"),
            )
            generate_catalogue_tile(cfg, 0)
            consolidate_catalogue(cfg)
            generate_injection_shard(cfg, 0)
            # shard 1 deliberately not generated
            with self.assertRaises(MDCValidationError) as ctx:
                assemble_injections(cfg)
            self.assertIn("missing injection shard", ctx.exception.report["issues"][0])
