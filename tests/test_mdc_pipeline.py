"""Tests for bagpuss.mdc.pipeline."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr

from bagpuss.injection import GPS_O3_END, GPS_O3_START, InjectionSet
from bagpuss.mdc.config import MDCConfig
from bagpuss.mdc.pipeline import (
    MDCValidationError,
    assemble_detections,
    assemble_injections,
    build_detectable,
    build_population,
    build_rate,
    build_selection,
    build_tile_universe,
    build_universe,
    consolidate_catalogue,
    detect_injection_shard,
    expected_injection_count,
    export_glade_catalogue,
    generate_catalogue_tile,
    generate_injection_shard,
    load_consolidated_catalogue,
    package_manifest,
    shard_rng,
    shard_time_bounds,
    tile_bounds,
)
from bagpuss.population import SECONDS_PER_YEAR, expected_n_mergers
from bagpuss.universe import SkyPatch

# A tiny, fast-drawing configuration: small z_max and a scaled-down phi_star
# (per the project's existing test convention for build_catalogue) so tile
# draws are O(1000) galaxies rather than millions. rate_density is likewise
# scaled down so a full-window Poisson draw is O(10-100) events, not O(1e5).
_TINY_KWARGS: dict[str, float | int | str | None] = {
    "z_max": 0.1,
    "phi_star": 1.0e-5,
    "m_lim": 25.0,
    "rate_density": 3.0e-7,
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
# shard_time_bounds
# ---------------------------------------------------------------------------


class TestShardTimeBounds(unittest.TestCase):
    """Tests for the shard_id -> (t_start, t_end) time-slice mapping."""

    def test_single_shard_covers_full_window(self) -> None:
        """With one shard, its bounds cover the whole default O3 window."""
        cfg = _tiny_config(n_injection_shards=1)
        t_start, t_end = shard_time_bounds(cfg, 0)
        self.assertEqual(t_start, GPS_O3_START)
        self.assertEqual(t_end, GPS_O3_END)

    def test_custom_window_used_when_set(self) -> None:
        """An explicit t_start/t_end on the config overrides the O3 default."""
        cfg = _tiny_config(n_injection_shards=1, t_start=0.0, t_end=100.0)
        self.assertEqual(shard_time_bounds(cfg, 0), (0.0, 100.0))

    def test_shards_partition_window_with_no_gaps(self) -> None:
        """Consecutive shards' bounds tile [t_start, t_end] with no gaps/overlaps."""
        cfg = _tiny_config(n_injection_shards=4, t_start=0.0, t_end=400.0)
        bounds = [shard_time_bounds(cfg, i) for i in range(cfg.n_injection_shards)]
        self.assertEqual(bounds[0][0], 0.0)
        self.assertEqual(bounds[-1][1], 400.0)
        for (_, end_i), (start_next, _) in zip(bounds, bounds[1:]):
            self.assertEqual(end_i, start_next)

    def test_out_of_range_shard_id_raises(self) -> None:
        """A shard_id outside [0, n_injection_shards) raises ValueError."""
        cfg = _tiny_config(n_injection_shards=2)
        with self.assertRaises(ValueError):
            shard_time_bounds(cfg, 2)

    def test_negative_shard_id_raises(self) -> None:
        """A negative shard_id raises ValueError."""
        cfg = _tiny_config(n_injection_shards=2)
        with self.assertRaises(ValueError):
            shard_time_bounds(cfg, -1)


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

    def test_build_rate_uses_config_rate_density(self) -> None:
        """build_rate's rate_density matches config.rate_density."""
        cfg = _tiny_config(rate_density=5.0e-7)
        rate = build_rate(cfg)
        np.testing.assert_array_equal(rate.rate_density(np.array([0.0, 0.05])), 5.0e-7)

    def test_build_rate_gives_nonzero_expected_count(self) -> None:
        """build_rate wired into expected_n_mergers gives a positive count."""
        cfg = _tiny_config()
        universe = build_universe(cfg)
        rate = build_rate(cfg)
        t_start, t_end = shard_time_bounds(cfg, 0)
        n_expected = expected_n_mergers(
            universe.structure, universe.cosmology, rate, t_start, t_end
        )
        self.assertGreater(n_expected, 0.0)


# ---------------------------------------------------------------------------
# expected_injection_count
# ---------------------------------------------------------------------------


class TestExpectedInjectionCount(unittest.TestCase):
    """Tests for the analytic, no-simulation event-count estimator."""

    def test_no_d_max_gives_none_detected(self) -> None:
        """With no d_max configured or passed, n_detected is None."""
        cfg = _tiny_config(d_max=None)
        result = expected_injection_count(cfg)
        self.assertIsNone(result["d_max"])
        self.assertIsNone(result["n_detected"])
        self.assertGreater(result["n_total"], 0.0)

    def test_config_d_max_used_by_default(self) -> None:
        """config.d_max is used when no override is passed."""
        cfg = _tiny_config(d_max=200.0)
        result = expected_injection_count(cfg)
        self.assertEqual(result["d_max"], 200.0)
        self.assertIsNotNone(result["n_detected"])

    def test_d_max_override_beats_config(self) -> None:
        """An explicit d_max argument overrides config.d_max."""
        cfg = _tiny_config(d_max=200.0)
        result = expected_injection_count(cfg, d_max=300.0)
        self.assertEqual(result["d_max"], 300.0)

    def test_small_d_max_gives_fewer_than_total(self) -> None:
        """A d_max well inside the survey volume gives n_detected < n_total."""
        cfg = _tiny_config()  # z_max=0.1 -> horizon ~= 476 Mpc
        result = expected_injection_count(cfg, d_max=200.0)
        self.assertLess(result["n_detected"], result["n_total"])

    def test_d_max_beyond_z_max_matches_total(self) -> None:
        """A d_max far beyond the survey volume gives n_detected == n_total."""
        cfg = _tiny_config()
        result = expected_injection_count(cfg, d_max=1.0e6)
        self.assertAlmostEqual(result["n_detected"], result["n_total"])

    def test_t_obs_years_matches_window(self) -> None:
        """t_obs_years reports the resolved (t_end - t_start) in years."""
        cfg = _tiny_config()
        result = expected_injection_count(cfg, t_start=0.0, t_end=SECONDS_PER_YEAR)
        self.assertAlmostEqual(result["t_obs_years"], 1.0)

    def test_doubling_window_doubles_total(self) -> None:
        """n_total scales linearly with the observation window length."""
        cfg = _tiny_config()
        one_year = expected_injection_count(cfg, t_start=0.0, t_end=SECONDS_PER_YEAR)
        two_years = expected_injection_count(
            cfg, t_start=0.0, t_end=2.0 * SECONDS_PER_YEAR
        )
        self.assertAlmostEqual(two_years["n_total"], 2.0 * one_year["n_total"])


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
                d_max=5000.0,
                store=str(Path(tmp) / "store.zarr"),
            )
            result = self._run_full_pipeline(cfg)

        self.assertTrue(result["catalogue_report"]["ok"])
        self.assertEqual(result["catalogue_report"]["n_tiles"], 4)
        self.assertGreater(result["catalogue_report"]["total_galaxies"], 0)

        self.assertTrue(result["injection_report"]["ok"])
        self.assertEqual(result["injection_report"]["n_shards"], 3)
        # The count is now a Poisson realisation from rate_density, not a
        # caller-chosen target -- just check it's positive and consistent.
        self.assertGreater(result["injection_report"]["total_injections"], 0)

        self.assertEqual(
            result["manifest"]["total_galaxies"],
            result["catalogue_report"]["total_galaxies"],
        )
        self.assertEqual(
            result["manifest"]["total_injections"],
            result["injection_report"]["total_injections"],
        )

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

    def test_catalogued_hosts_resolve_to_matching_catalogue_rows(self) -> None:
        """A catalogued injection's fields exactly match its catalogue row.

        Stronger than test_injections_have_valid_host_indices (which only
        checks the index is in-range): this confirms the rewritten
        build_catalogue (selection-weighted sampling, not raw-then-filter)
        hasn't broken the property that a "catalogued" injection really
        does correspond to a real row in the same run's catalogue -- the
        same property already verified for the GLADE export
        (TestExportGladeCatalogue.test_exported_rows_match_host_galaxy_index).
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=2,
                n_dec_tiles=1,
                n_injection_shards=1,
                store=str(Path(tmp) / "store.zarr"),
            )
            for tile_id in range(cfg.n_tiles):
                generate_catalogue_tile(cfg, tile_id)
            consolidate_catalogue(cfg)
            injections = generate_injection_shard(cfg, 0)
            catalogue = load_consolidated_catalogue(cfg)

        hosted = injections.host_galaxy_index[injections.host_galaxy_index >= 0]
        self.assertGreater(len(hosted), 0, "test config drew no catalogued hosts")
        for idx in hosted:
            event_mask = injections.host_galaxy_index == idx
            np.testing.assert_allclose(
                injections.redshift[event_mask], catalogue.redshifts[idx]
            )
            np.testing.assert_allclose(injections.ra[event_mask], catalogue.ra[idx])
            np.testing.assert_allclose(injections.dec[event_mask], catalogue.dec[idx])

    def test_reproducible_across_full_reruns(self) -> None:
        """Re-running the whole pipeline with the same seed reproduces it exactly."""
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = _tiny_config(
                    n_ra_tiles=2,
                    n_dec_tiles=1,
                    n_injection_shards=2,
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


class TestExportGladeCatalogue(unittest.TestCase):
    """Tests for export_glade_catalogue."""

    def _read_ids(self, path: Path) -> list[int]:
        with open(path) as f:
            return [int(line.split()[0]) for line in f if line.strip()]

    def test_raises_if_not_consolidated(self) -> None:
        """Exporting before consolidate_catalogue has run raises."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(store=str(Path(tmp) / "store.zarr"))
            generate_catalogue_tile(cfg, 0)
            with self.assertRaises(MDCValidationError):
                export_glade_catalogue(cfg, Path(tmp) / "export")

    def test_row_count_matches_consolidated_total(self) -> None:
        """The exported row count matches consolidate_catalogue's total."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=2, n_dec_tiles=2, store=str(Path(tmp) / "store.zarr")
            )
            for tile_id in range(cfg.n_tiles):
                generate_catalogue_tile(cfg, tile_id)
            report = consolidate_catalogue(cfg)
            out_dir = Path(tmp) / "export"
            result = export_glade_catalogue(cfg, out_dir)
            ids = self._read_ids(out_dir / "catalogue.dat")

        self.assertEqual(result["n_galaxies"], report["total_galaxies"])
        self.assertEqual(len(ids), report["total_galaxies"])

    def test_ids_are_contiguous_across_tile_boundaries(self) -> None:
        """IDs run 0..N-1 across tiles, not reset per tile."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=3, n_dec_tiles=1, store=str(Path(tmp) / "store.zarr")
            )
            for tile_id in range(cfg.n_tiles):
                generate_catalogue_tile(cfg, tile_id)
            consolidate_catalogue(cfg)
            out_dir = Path(tmp) / "export"
            export_glade_catalogue(cfg, out_dir)
            ids = self._read_ids(out_dir / "catalogue.dat")

        self.assertEqual(ids, list(range(len(ids))))

    def test_writes_completeness_and_readme(self) -> None:
        """The completeness table and README are written alongside the catalogue."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(store=str(Path(tmp) / "store.zarr"))
            generate_catalogue_tile(cfg, 0)
            consolidate_catalogue(cfg)
            out_dir = Path(tmp) / "export"
            export_glade_catalogue(cfg, out_dir)

            self.assertTrue((out_dir / "completeness.dat").exists())
            readme_text = (out_dir / "README.txt").read_text()
            self.assertIn("host_galaxy_index", readme_text)

    def test_exported_rows_match_host_galaxy_index(self) -> None:
        """An injection's host_galaxy_index matches the exported row's data.

        The property the user specifically asked about: the exported
        catalogue must be usable as the host-galaxy lookup table.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config(
                n_ra_tiles=2,
                n_dec_tiles=1,
                n_injection_shards=1,
                store=str(Path(tmp) / "store.zarr"),
            )
            for tile_id in range(cfg.n_tiles):
                generate_catalogue_tile(cfg, tile_id)
            consolidate_catalogue(cfg)
            injections = generate_injection_shard(cfg, 0)
            catalogue = load_consolidated_catalogue(cfg)
            out_dir = Path(tmp) / "export"
            export_glade_catalogue(cfg, out_dir)

            with open(out_dir / "catalogue.dat") as f:
                exported = {
                    int(parts[0]): parts
                    for parts in (line.split() for line in f)
                    if parts
                }

        hosted = injections.host_galaxy_index[injections.host_galaxy_index >= 0]
        self.assertGreater(len(hosted), 0, "test config drew no catalogued hosts")
        for idx in hosted:
            row = exported[int(idx)]
            self.assertAlmostEqual(
                float(row[1]), np.degrees(catalogue.ra[idx]), places=4
            )
            self.assertAlmostEqual(
                float(row[2]), np.degrees(catalogue.dec[idx]), places=4
            )
            self.assertAlmostEqual(float(row[3]), catalogue.redshifts[idx], places=5)


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
                store=str(Path(tmp) / "store.zarr"),
            )
            generate_catalogue_tile(cfg, 0)
            consolidate_catalogue(cfg)
            generate_injection_shard(cfg, 0)
            # shard 1 deliberately not generated
            with self.assertRaises(MDCValidationError) as ctx:
                assemble_injections(cfg)
            self.assertIn("missing injection shard", ctx.exception.report["issues"][0])


# ---------------------------------------------------------------------------
# Detection stage (SNRs, frames, blueprints) with a fake waveform backend
# ---------------------------------------------------------------------------


class _FakeSchedule:
    """Detector locked on a fixed window of GPS time."""

    def __init__(self, locked: tuple[float, float]) -> None:
        self.locked = locked

    def is_active(self, t: float) -> bool:
        return self.locked[0] <= t < self.locked[1]


class _FakeBackend:
    """Deterministic stand-in for MinkeBackend: SNR = 5000 / distance.

    The "L1" detector is locked for the first half of the observation
    window only, so events in the second half have no observing detector.
    """

    def __init__(self) -> None:
        self.injected: list[tuple[str, ...]] = []
        self.framefiles: list[str | None] = []
        self.rng_draws: list[float] = []

    def abbreviation(self, detector: str) -> str:
        return detector[-2:]

    def injection_parameters(
        self, injections: InjectionSet, f_ref: float
    ) -> list[dict[str, Any]]:
        return [
            {"gpstime": float(t), "dist": float(d), "index": i}
            for i, (t, d) in enumerate(
                zip(injections.geocent_time, injections.luminosity_distance)
            )
        ]

    def duty_schedules(
        self,
        config: MDCConfig,
        t_start: float,
        t_end: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        self.rng_draws.append(float(rng.random()))
        mid = 0.5 * (t_start + t_end)
        return {name: _FakeSchedule((t_start, mid)) for name in config.detectors}

    def active_detectors(
        self, schedules: dict[str, Any], detectors: dict[str, str], t: float
    ) -> dict[str, str]:
        return {n: p for n, p in detectors.items() if schedules[n].is_active(t)}

    def inject(
        self,
        params: dict[str, Any],
        detectors: dict[str, str],
        config: MDCConfig,
        framefile: str | None,
    ) -> tuple[dict[str, Any], float]:
        self.injected.append(tuple(detectors))
        self.framefiles.append(framefile)
        frames: dict[str, Any] = {}
        if framefile is not None:
            frames = {"H1": {"path": f"{framefile}.gwf", "channel": "x", "snr": 1.0}}
            Path(f"{framefile}.gwf").write_text("frame")
        return frames, 5000.0 / params["dist"]

    def write_blueprints(
        self,
        params: list[dict[str, Any]],
        path: str | Path,
        config: MDCConfig,
        frame_files: list[dict[str, Any]],
        network_snrs: list[float],
        active: list[dict[str, str]],
        skymaps: list[Path | None],
    ) -> None:
        docs = [
            {"kind": "event", "name": f"inj_{p['gpstime']:.3f}", "snr": s}
            | ({"localization file": str(m)} if m is not None else {})
            for p, s, m in zip(params, network_snrs, skymaps)
        ]
        Path(path).write_text(yaml.safe_dump_all(docs, sort_keys=False))


class _FakeLocalizer:
    """Records calls; fails for every other event if ``fail_odd``."""

    def __init__(self, fail_odd: bool = False) -> None:
        self.calls: list[
            tuple[dict[str, Any], dict[str, str], dict[str, str], int]
        ] = []
        self.fail_odd = fail_odd

    def localize(
        self,
        event: dict[str, Any],
        observing: dict[str, str],
        psds: dict[str, str],
        seed: int,
        out_path: Path,
    ) -> Path | None:
        self.calls.append((event, observing, psds, seed))
        if self.fail_odd and event["index"] % 2:
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("fits")
        return out_path


class TestDetection(unittest.TestCase):
    """Tests for detect_injection_shard / assemble_detections."""

    def _run_to_injections(self, tmp: str, **overrides: Any) -> MDCConfig:  # noqa: ANN401
        cfg = _tiny_config(
            n_injection_shards=2,
            store=str(Path(tmp) / "store.zarr"),
            output_dir=str(Path(tmp) / "out"),
            snr_threshold=8.0,
            **overrides,
        )
        generate_catalogue_tile(cfg, 0)
        consolidate_catalogue(cfg)
        for shard_id in range(cfg.n_injection_shards):
            generate_injection_shard(cfg, shard_id)
        assemble_injections(cfg)
        return cfg

    def test_snr_stored_for_every_injection(self) -> None:
        """The SNR table has one row per injection, detectable or not."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            detect_injection_shard(cfg, 0, backend=_FakeBackend())
            root = zarr.open_group(cfg.store, mode="r", use_consolidated=False)
            n = len(InjectionSet.from_zarr(root["injections"]["shard_0000"]))
            group = root["detections"]["shard_0000"]
            snr = np.asarray(group["network_snr"])
            detectable = np.asarray(group["detectable"])
            active = np.asarray(group["active"])
        self.assertGreater(n, 0)
        self.assertEqual(snr.shape, (n,))
        self.assertEqual(detectable.shape, (n,))
        self.assertEqual(active.shape, (n, len(cfg.detectors)))
        self.assertTrue(np.array_equal(detectable, snr >= cfg.snr_threshold))
        # Events with no observing detector get exactly zero SNR.
        self.assertTrue(np.all(snr[~active.any(axis=1)] == 0.0))

    def test_frames_only_for_detectable_events(self) -> None:
        """Frames are written for survivors only; the cut events get none."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            backend = _FakeBackend()
            summary = detect_injection_shard(cfg, 0, backend=backend)
            written = [f for f in backend.framefiles if f is not None]
            frames = sorted((Path(tmp) / "out" / "frames" / "shard_0000").glob("*.gwf"))
        self.assertGreater(summary["n_detectable"], 0)
        self.assertEqual(len(written), summary["n_detectable"])
        self.assertEqual(len(frames), summary["n_detectable"])
        self.assertTrue(all("s0000_event_" in f for f in written))

    def test_no_frames_when_write_frames_false(self) -> None:
        """write_frames=False stores SNRs and blueprints but writes no frames."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp, write_frames=False)
            backend = _FakeBackend()
            summary = detect_injection_shard(cfg, 0, backend=backend)
            frames = list((Path(tmp) / "out" / "frames").rglob("*.gwf"))
            docs = [
                d
                for d in yaml.safe_load_all(
                    (Path(tmp) / "out" / "blueprints" / "shard_0000.yaml").read_text()
                )
                if d
            ]
        self.assertGreater(summary["n_detectable"], 0)
        self.assertEqual(frames, [])
        self.assertTrue(all(f is None for f in backend.framefiles))
        self.assertEqual(len(docs), summary["n_detectable"])

    def test_assemble_writes_analysis_blueprint_without_frames(self) -> None:
        """A frameless run also gets one generic minke analysis blueprint."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp, write_frames=False)
            for shard_id in range(cfg.n_injection_shards):
                detect_injection_shard(cfg, shard_id, _FakeBackend())
            assemble_detections(cfg)
            path = Path(tmp) / "out" / "analysis-minke.yaml"
            analysis = yaml.safe_load(path.read_text())
        self.assertEqual(analysis["kind"], "analysis")
        self.assertEqual(analysis["pipeline"], "minke")

    def test_no_analysis_blueprint_when_frames_written(self) -> None:
        """With frames written there is nothing for an asimov analysis to do."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            for shard_id in range(cfg.n_injection_shards):
                detect_injection_shard(cfg, shard_id, _FakeBackend())
            assemble_detections(cfg)
            self.assertFalse((Path(tmp) / "out" / "analysis-minke.yaml").exists())

    def test_skymaps_written_for_survivors(self) -> None:
        """With write_skymaps every survivor gets a skymap, noted in its blueprint."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp, write_skymaps=True, write_frames=False)
            localizer = _FakeLocalizer()
            summary = detect_injection_shard(
                cfg, 0, backend=_FakeBackend(), localizer=localizer
            )
            docs = [
                d
                for d in yaml.safe_load_all(
                    (Path(tmp) / "out" / "blueprints" / "shard_0000.yaml").read_text()
                )
                if d
            ]
            files = sorted(
                (Path(tmp) / "out" / "skymaps" / "shard_0000").glob("*.fits")
            )
            group = zarr.open_group(cfg.store, mode="r", use_consolidated=False)[
                "detections"
            ]["shard_0000"]
            has_skymap = np.asarray(group["has_skymap"])
        self.assertGreater(summary["n_detectable"], 0)
        self.assertEqual(summary["n_skymaps"], summary["n_detectable"])
        self.assertEqual(len(localizer.calls), summary["n_detectable"])
        self.assertEqual(len(files), summary["n_detectable"])
        self.assertTrue(all("localization file" in d for d in docs))
        self.assertEqual(int(has_skymap.sum()), summary["n_detectable"])
        # detector abbreviations and PSDs are passed through
        _, observing, psds, _ = localizer.calls[0]
        self.assertEqual(set(observing), set(psds))

    def test_skymap_seeds_differ_per_event(self) -> None:
        """Each event's simulated measurement errors use their own seed."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp, write_skymaps=True)
            localizer = _FakeLocalizer()
            detect_injection_shard(cfg, 0, backend=_FakeBackend(), localizer=localizer)
        seeds = [c[3] for c in localizer.calls]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_skymap_failures_are_counted_not_fatal(self) -> None:
        """An event BAYESTAR could not localise has no skymap but is still written."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp, write_skymaps=True, write_frames=False)
            summary = detect_injection_shard(
                cfg, 0, backend=_FakeBackend(), localizer=_FakeLocalizer(fail_odd=True)
            )
            docs = [
                d
                for d in yaml.safe_load_all(
                    (Path(tmp) / "out" / "blueprints" / "shard_0000.yaml").read_text()
                )
                if d
            ]
        self.assertEqual(len(docs), summary["n_detectable"])
        self.assertLess(summary["n_skymaps"], summary["n_detectable"])
        self.assertEqual(
            sum("localization file" in d for d in docs), summary["n_skymaps"]
        )

    def test_no_skymaps_by_default(self) -> None:
        """The localiser is not touched unless write_skymaps is set."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            localizer = _FakeLocalizer()
            summary = detect_injection_shard(
                cfg, 0, backend=_FakeBackend(), localizer=localizer
            )
        self.assertEqual(localizer.calls, [])
        self.assertEqual(summary["n_skymaps"], 0)

    def test_assemble_reports_skymap_totals(self) -> None:
        """assemble_detections totals the skymaps and the events without one."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp, write_skymaps=True, write_frames=False)
            for shard_id in range(cfg.n_injection_shards):
                detect_injection_shard(
                    cfg,
                    shard_id,
                    backend=_FakeBackend(),
                    localizer=_FakeLocalizer(fail_odd=True),
                )
            report = assemble_detections(cfg)
        self.assertEqual(
            report["total_skymaps"] + report["n_without_skymap"],
            report["total_detectable"],
        )
        self.assertGreater(report["n_without_skymap"], 0)

    def test_threshold_controls_selection(self) -> None:
        """A higher SNR threshold selects a subset of the same SNR table."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            low = detect_injection_shard(cfg, 0, backend=_FakeBackend())
            from dataclasses import replace

            strict = replace(cfg, snr_threshold=1e9)
            none = detect_injection_shard(strict, 0, backend=_FakeBackend())
        self.assertGreater(low["n_detectable"], 0)
        self.assertEqual(none["n_detectable"], 0)
        self.assertEqual(low["n_injections"], none["n_injections"])

    def test_duty_schedule_identical_across_shards(self) -> None:
        """Every shard draws the duty-cycle schedule from the same RNG stream."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            b0, b1 = _FakeBackend(), _FakeBackend()
            detect_injection_shard(cfg, 0, backend=b0)
            detect_injection_shard(cfg, 1, backend=b1)
        self.assertEqual(b0.rng_draws, b1.rng_draws)

    def test_out_of_range_shard_raises(self) -> None:
        """A shard_id outside [0, n_injection_shards) raises ValueError."""
        cfg = _tiny_config(n_injection_shards=2)
        with self.assertRaises(ValueError):
            detect_injection_shard(cfg, 2, backend=_FakeBackend())

    def test_assemble_merges_blueprints(self) -> None:
        """assemble_detections writes one merged blueprints.yaml."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            total = 0
            for shard_id in range(cfg.n_injection_shards):
                total += detect_injection_shard(cfg, shard_id, _FakeBackend())[
                    "n_detectable"
                ]
            report = assemble_detections(cfg)
            merged = (Path(tmp) / "out" / "blueprints.yaml").read_text()
            docs = [d for d in yaml.safe_load_all(merged) if d]
            manifest = package_manifest(cfg)
        self.assertTrue(report["ok"])
        self.assertEqual(report["total_detectable"], total)
        self.assertEqual(len(docs), total)
        self.assertEqual(manifest["total_detectable"], total)

    def test_assemble_flags_missing_shard(self) -> None:
        """A shard that never ran is reported as missing."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            detect_injection_shard(cfg, 0, backend=_FakeBackend())
            with self.assertRaises(MDCValidationError):
                assemble_detections(cfg)

    def test_assemble_flags_blueprint_count_mismatch(self) -> None:
        """A truncated blueprint file fails validation."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._run_to_injections(tmp)
            for shard_id in range(cfg.n_injection_shards):
                detect_injection_shard(cfg, shard_id, _FakeBackend())
            n_det = int(
                zarr.open_group(cfg.store, mode="r", use_consolidated=False)[
                    "detections"
                ]["shard_0000"].attrs["n_detectable"]
            )
            self.assertGreater(n_det, 0)
            (Path(tmp) / "out" / "blueprints" / "shard_0000.yaml").write_text("")
            report = assemble_detections(cfg, raise_on_failure=False)
        self.assertFalse(report["ok"])
