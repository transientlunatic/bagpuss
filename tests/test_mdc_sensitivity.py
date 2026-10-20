"""Tests for the sensitivity-injection stages of bagpuss.mdc."""

# zarr indexes into a union of Group | Array; the tests know which.
# pyright: reportIndexIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import zarr

from bagpuss.injection import InjectionSet
from bagpuss.mdc.config import MDCConfig, load_config
from bagpuss.mdc.pipeline import (
    MDCValidationError,
    assemble_sensitivity,
    build_population,
    build_population_log_prob,
    build_sensitivity_distribution,
    generate_sensitivity_shard,
    package_manifest,
    sensitivity_time_bounds,
    shard_rng,
)
from bagpuss.population import ConstantMergerRate, sample_merger_redshifts
from bagpuss.sensitivity import SensitivityInjectionSet, draw_injections
from bagpuss.universe import PointProcess

_T0 = 1_000_000_000.0
_T1 = _T0 + 4.0e6
_N_PER_SHARD = 400


class _FakeBackend:
    """Deterministic stand-in for MinkeBackend: SNR = 12000 / distance.

    Detector "AdvancedLIGOLivingston" is locked in the first half of the
    window only, "AdvancedLIGOHanford" throughout.
    """

    def __init__(self) -> None:
        self.rng_draws: list[float] = []

    def injection_parameters(
        self, injections: InjectionSet, f_ref: float
    ) -> list[dict[str, Any]]:
        assert np.all(injections.host_galaxy_index == -1)
        return [
            {"gpstime": float(t), "dist": float(d)}
            for t, d in zip(injections.geocent_time, injections.luminosity_distance)
        ]

    def duty_schedules(
        self,
        config: MDCConfig,
        t_start: float,
        t_end: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        self.rng_draws.append(float(rng.random()))
        return {"mid": 0.5 * (t_start + t_end)}

    def active_detectors(
        self, schedules: dict[str, Any], detectors: dict[str, str], t: float
    ) -> dict[str, str]:
        observing = dict(detectors)
        if t >= schedules["mid"]:
            observing.pop("AdvancedLIGOLivingston", None)
        return observing

    def inject(
        self,
        params: dict[str, Any],
        detectors: dict[str, str],
        config: MDCConfig,
        framefile: str | None,
    ) -> tuple[dict[str, Any], float]:
        assert framefile is None, "sensitivity injections never write frames"
        return {}, 12_000.0 / params["dist"]


def _config(tmp: str, **overrides: Any) -> MDCConfig:  # noqa: ANN401
    kwargs: dict[str, Any] = {
        "t_start": _T0,
        "t_end": _T1,
        "store": str(Path(tmp) / "store.zarr"),
        "output_dir": str(Path(tmp) / "out"),
        "snr_threshold": 8.0,
        "sensitivity_n_shards": 3,
        "sensitivity_n_per_shard": _N_PER_SHARD,
        "sensitivity_z_max": 2.0,
    }
    kwargs.update(overrides)
    return MDCConfig(**kwargs)


def _run_all(cfg: MDCConfig, backend: _FakeBackend | None = None) -> None:
    for shard_id in range(cfg.sensitivity_n_shards):
        generate_sensitivity_shard(cfg, shard_id, backend or _FakeBackend())


class TestSensitivityConfig(unittest.TestCase):
    """The sensitivity section of the configuration."""

    def test_disabled_by_default(self) -> None:
        """No campaign is configured unless asked for."""
        self.assertFalse(MDCConfig().sensitivity_enabled)

    def test_loaded_from_yaml_section(self) -> None:
        """The nested section maps onto the flat fields."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "sensitivity": {
                            "n_shards": 7,
                            "n_per_shard": 123,
                            "z_max": 2.5,
                            "min_snr": 3.0,
                        }
                    }
                )
            )
            cfg = load_config(path)
        self.assertTrue(cfg.sensitivity_enabled)
        self.assertEqual(cfg.sensitivity_n_shards, 7)
        self.assertEqual(cfg.sensitivity_n_per_shard, 123)
        self.assertEqual(cfg.sensitivity_z_max, 2.5)
        self.assertEqual(cfg.sensitivity_min_snr, 3.0)

    def test_min_snr_may_not_exceed_threshold(self) -> None:
        """Dropping injections above the threshold would bias P_det."""
        with self.assertRaises(ValueError):
            MDCConfig(snr_threshold=8.0, sensitivity_min_snr=9.0)

    def test_rejects_bad_sizes(self) -> None:
        """Shard counts must be sensible."""
        with self.assertRaises(ValueError):
            MDCConfig(sensitivity_n_shards=0)
        with self.assertRaises(ValueError):
            MDCConfig(sensitivity_n_per_shard=-1)


class TestSensitivityTimeBounds(unittest.TestCase):
    """Shards are equal slices of the observation window."""

    def test_slices_tile_the_window(self) -> None:
        """Adjacent slices abut and cover the whole window."""
        cfg = MDCConfig(t_start=_T0, t_end=_T1, sensitivity_n_shards=4)
        bounds = [sensitivity_time_bounds(cfg, i) for i in range(4)]
        self.assertEqual(bounds[0][0], _T0)
        self.assertEqual(bounds[-1][1], _T1)
        for (_, hi), (lo, _) in zip(bounds[:-1], bounds[1:]):
            self.assertEqual(hi, lo)

    def test_rejects_out_of_range_shard(self) -> None:
        """Shard ids are bounds-checked."""
        cfg = MDCConfig(sensitivity_n_shards=2)
        with self.assertRaises(ValueError):
            sensitivity_time_bounds(cfg, 2)


class TestGenerateSensitivityShard(unittest.TestCase):
    """generate_sensitivity_shard."""

    def test_refuses_when_disabled(self) -> None:
        """n_per_shard = 0 means there is nothing to generate."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, sensitivity_n_per_shard=0)
            with self.assertRaises(ValueError):
                generate_sensitivity_shard(cfg, 0, _FakeBackend())

    def test_writes_every_injection_with_its_snr(self) -> None:
        """All draws are stored with an SNR, and no frames are written."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            summary = generate_sensitivity_shard(cfg, 1, _FakeBackend())
            root = zarr.open_group(cfg.store, mode="r", use_consolidated=False)
            group = root["sensitivity"]["shard_0001"]
            injections = SensitivityInjectionSet.from_zarr(group)
            snr = np.asarray(group["network_snr"])
            active = np.asarray(group["active"])
        self.assertEqual(len(injections), _N_PER_SHARD)
        self.assertEqual(injections.n_total, _N_PER_SHARD)
        self.assertEqual(snr.shape, (_N_PER_SHARD,))
        np.testing.assert_allclose(snr, 12_000.0 / injections.luminosity_distance)
        self.assertEqual(active.shape, (_N_PER_SHARD, len(cfg.detectors)))
        self.assertEqual(summary["n_total"], _N_PER_SHARD)
        self.assertEqual(summary["n_detected"], int(np.sum(snr >= 8.0)))

    def test_injections_lie_in_the_shards_time_slice(self) -> None:
        """Each shard draws only from its own slice of the window."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            generate_sensitivity_shard(cfg, 2, _FakeBackend())
            root = zarr.open_group(cfg.store, mode="r", use_consolidated=False)
            t = np.asarray(root["sensitivity"]["shard_0002"]["geocent_time"])
        lo, hi = sensitivity_time_bounds(cfg, 2)
        self.assertTrue(np.all((t >= lo) & (t <= hi)))

    def test_draw_density_is_that_of_the_full_window(self) -> None:
        """The pooled sample's density is uniform in time over the whole window."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            generate_sensitivity_shard(cfg, 0, _FakeBackend())
            root = zarr.open_group(cfg.store, mode="r", use_consolidated=False)
            injections = SensitivityInjectionSet.from_zarr(
                root["sensitivity"]["shard_0000"]
            )
        expected = build_sensitivity_distribution(cfg).log_prob(injections.as_dict())
        np.testing.assert_allclose(injections.log_p_draw, expected)

    def test_uses_the_same_duty_cycle_stream_as_the_event_set(self) -> None:
        """Both campaigns must see the same detector schedules."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            backend = _FakeBackend()
            generate_sensitivity_shard(cfg, 0, backend)
        self.assertEqual(backend.rng_draws, [float(shard_rng(cfg, "duty", 0).random())])

    def test_detectors_follow_the_schedule(self) -> None:
        """Events after the (fake) lock loss are seen by one detector fewer."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            generate_sensitivity_shard(cfg, 2, _FakeBackend())
            root = zarr.open_group(cfg.store, mode="r", use_consolidated=False)
            active = np.asarray(root["sensitivity"]["shard_0002"]["active"])
        names = list(cfg.detectors)
        livingston = names.index("AdvancedLIGOLivingston")
        self.assertFalse(active[:, livingston].any())
        self.assertTrue(active[:, names.index("AdvancedLIGOHanford")].all())

    def test_reproducible(self) -> None:
        """The same shard of the same config draws the same injections."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for tmp in (a, b):
                generate_sensitivity_shard(
                    _config(tmp, master_seed=5), 0, _FakeBackend()
                )
            sets = [
                SensitivityInjectionSet.from_zarr(
                    zarr.open_group(
                        str(Path(tmp) / "store.zarr"), mode="r", use_consolidated=False
                    )["sensitivity"]["shard_0000"]
                )
                for tmp in (a, b)
            ]
        np.testing.assert_array_equal(sets[0].m1_source, sets[1].m1_source)


class TestAssembleSensitivity(unittest.TestCase):
    """assemble_sensitivity."""

    def test_reports_totals(self) -> None:
        """The report and store attributes carry the pooled totals."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            _run_all(cfg)
            report = assemble_sensitivity(cfg)
            attrs = dict(zarr.open_group(cfg.store, mode="r")["sensitivity"].attrs)
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_shards"], 3)
        self.assertEqual(report["n_total"], 3 * _N_PER_SHARD)
        self.assertEqual(attrs["_n_total"], 3 * _N_PER_SHARD)
        self.assertEqual(attrs["_n_detected"], report["n_detected"])

    def test_missing_shard_is_a_failure(self) -> None:
        """Every shard must be present."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            for shard_id in (0, 2):
                generate_sensitivity_shard(cfg, shard_id, _FakeBackend())
            with self.assertRaises(MDCValidationError) as ctx:
                assemble_sensitivity(cfg)
        self.assertIn(
            "missing sensitivity shard: shard_0001", ctx.exception.report["issues"]
        )

    def test_wrong_shard_size_is_a_failure(self) -> None:
        """A shard of the wrong size would break the uniform-mixture density."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            _run_all(cfg)
            other = _config(tmp, sensitivity_n_per_shard=_N_PER_SHARD + 10)
            with self.assertRaises(MDCValidationError):
                assemble_sensitivity(other)

    def test_can_report_without_raising(self) -> None:
        """raise_on_failure=False returns the failing report instead."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            report = assemble_sensitivity(cfg, raise_on_failure=False)
        self.assertFalse(report["ok"])


class TestPooledEstimate(unittest.TestCase):
    """The pooled shards give a correct importance-sampling estimate."""

    def _pooled(self, cfg: MDCConfig) -> tuple[SensitivityInjectionSet, np.ndarray]:
        root = zarr.open_group(cfg.store, mode="r", use_consolidated=False)
        shards = [
            SensitivityInjectionSet.from_zarr(root["sensitivity"][f"shard_{i:04d}"])
            for i in range(cfg.sensitivity_n_shards)
        ]
        snr = np.concatenate(
            [
                np.asarray(root["sensitivity"][f"shard_{i:04d}"]["network_snr"])
                for i in range(cfg.sensitivity_n_shards)
            ]
        )
        return SensitivityInjectionSet.concatenate(shards), snr

    def test_target_equal_to_draw_recovers_the_detected_fraction(self) -> None:
        """With the full-window density, the weights are exactly one."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            _run_all(cfg)
            pooled, snr = self._pooled(cfg)
        estimate = pooled.detection_probability(
            build_sensitivity_distribution(cfg).log_prob,
            snr=snr,
            snr_threshold=cfg.snr_threshold,
        )
        self.assertEqual(pooled.n_total, 3 * _N_PER_SHARD)
        self.assertAlmostEqual(estimate.value, float(np.mean(snr >= 8.0)), places=12)

    def test_pooled_density_is_normalised_over_the_window(self) -> None:
        """E_draw[p_target / p_draw] = 1 only if the pooled density is right."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp, sensitivity_n_per_shard=4000)
            _run_all(cfg)
            pooled, _ = self._pooled(cfg)
        target = build_sensitivity_distribution(cfg)
        weights = np.exp(target.log_prob(pooled.as_dict()) - pooled.log_p_draw)
        sem = weights.std() / np.sqrt(len(weights))
        self.assertLess(abs(weights.mean() - 1.0), 4 * sem + 1e-12)


class TestPopulationClosure(unittest.TestCase):
    """The run's true population, reweighted from the broad injections."""

    @staticmethod
    def _toy_snr(m1: np.ndarray, d_l: np.ndarray) -> np.ndarray:
        """Return a toy SNR that depends on mass as well as distance."""
        return 16_000.0 / d_l * np.sqrt(m1 / 30.0)

    def test_reweighted_estimate_matches_direct_monte_carlo(self) -> None:
        """P_det(true population) from the broad injections equals a direct draw."""
        cfg = MDCConfig(
            t_start=_T0,
            t_end=_T1,
            z_max=1.0,
            sensitivity_z_max=3.0,
            sensitivity_n_per_shard=1,
        )
        rng = np.random.default_rng(42)

        # Direct: draw from the true population and count detections.
        n_direct = 40_000
        bbh = build_population(cfg).sample(n_direct, rng)
        z = sample_merger_redshifts(
            PointProcess(z_max=cfg.z_max),
            build_sensitivity_distribution(cfg).cosmology,
            ConstantMergerRate(cfg.rate_density),
            n_direct,
            rng,
        )
        d_l = build_sensitivity_distribution(cfg).cosmology.luminosity_distance(z).value
        detected = self._toy_snr(bbh.m1_source, d_l) >= 8.0
        direct = detected.mean()
        direct_err = np.sqrt(direct * (1.0 - direct) / n_direct)

        # Reweighted: draw from the broad distribution, reweight to the population.
        broad = build_sensitivity_distribution(cfg)
        inj = draw_injections(broad, 150_000, rng)
        snr = self._toy_snr(inj.m1_source, inj.luminosity_distance)
        estimate = inj.detection_probability(
            build_population_log_prob(cfg), snr, snr_threshold=8.0
        )

        self.assertGreater(direct, 0.02)
        self.assertGreater(estimate.n_found, 300)
        error = np.hypot(direct_err, estimate.std)
        self.assertLess(abs(estimate.value - direct), 4.0 * error)
        self.assertTrue(estimate.is_reliable(n_events=50))


class TestManifest(unittest.TestCase):
    """package_manifest records the sensitivity totals."""

    def test_manifest_has_sensitivity_counts(self) -> None:
        """The manifest names the number of draws and detections."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            _run_all(cfg)
            assemble_sensitivity(cfg)
            # package_manifest also reads the catalogue and injection groups.
            root = zarr.open_group(cfg.store, mode="a")
            root.require_group("catalogue")
            root.require_group("injections")
            manifest = package_manifest(cfg)
        self.assertEqual(manifest["sensitivity_n_total"], 3 * _N_PER_SHARD)
        self.assertIsNotNone(manifest["sensitivity_n_detected"])


if __name__ == "__main__":
    unittest.main()
