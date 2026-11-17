"""Tests for bagpuss.mdc.recalibration (distance recalibration of posteriors)."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from click.testing import CliRunner
from scipy.special import ndtri

from bagpuss.mdc.recalibration import (
    DistanceRecalibration,
    blueprint_truths,
    calibration_report,
    credible_level,
    fit_distance_recalibration,
    recalibrate_catalogue,
    recalibrate_directory,
    recalibrate_distance,
    release_truths,
    split_validation,
)

Events = list[tuple[np.ndarray, np.ndarray, float]]


def _gaussian_log_d(
    rng: np.random.Generator, centre: float, sigma: float = 0.25, n: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-weight distance samples, Gaussian in ln D."""
    return np.exp(centre + sigma * rng.standard_normal(n)), np.ones(n)


def _events(rng: np.random.Generator, offset: float, spread: float, n: int) -> Events:
    """Events whose posterior centre is offset ``offset``, scattered by ``spread``."""
    events = []
    for _ in range(n):
        truth = float(np.exp(rng.uniform(np.log(300.0), np.log(5000.0))))
        centre = np.log(truth) + offset + 0.25 * spread * rng.standard_normal()
        d, w = _gaussian_log_d(rng, centre)
        events.append((d, w, truth))
    return events


class TestRecalibrateDistance(unittest.TestCase):
    """The map applied to one event's samples."""

    def test_identity(self) -> None:
        """Zero shift and unit scale change nothing."""
        d = np.array([100.0, 200.0, 400.0])
        new_d, new_w = recalibrate_distance(d, np.ones(3), DistanceRecalibration())
        np.testing.assert_allclose(new_d, d)
        np.testing.assert_allclose(new_w, np.ones(3))

    def test_median_shifts_and_width_scales(self) -> None:
        """The median moves by the shift and the width in ln D by the scale."""
        d, w = _gaussian_log_d(np.random.default_rng(0), np.log(1000.0), 0.2, 4000)
        cal = DistanceRecalibration(shift=0.1, scale=1.5)
        new_d, _ = recalibrate_distance(d, w, cal)
        x, y = np.log(d), np.log(new_d)
        self.assertAlmostEqual(float(np.median(y)), float(np.median(x)) - 0.1, places=2)
        self.assertAlmostEqual(float(y.std() / x.std()), 1.5, places=2)

    def test_weights_carry_the_prior_ratio(self) -> None:
        """With pi ~ D^2 a sample moved from D to D' is weighted by (D'/D)^2."""
        d = np.array([500.0, 1000.0, 2000.0])
        cal = DistanceRecalibration(shift=0.05, scale=1.3)
        new_d, new_w = recalibrate_distance(d, np.ones(3), cal)
        np.testing.assert_allclose(new_w, (new_d / d) ** 2)

    def test_prior_power_is_used(self) -> None:
        """A flat prior in D leaves the weights alone."""
        d = np.array([500.0, 1000.0, 2000.0])
        cal = DistanceRecalibration(shift=0.05, scale=1.3)
        new_d, new_w = recalibrate_distance(d, np.ones(3), cal, prior_power=0.0)
        np.testing.assert_allclose(new_w, np.ones(3))
        self.assertFalse(np.allclose(new_d, d))

    def test_bad_input(self) -> None:
        """Mismatched arrays and a non-positive scale are errors."""
        with self.assertRaises(ValueError):
            recalibrate_distance(
                np.array([1.0, 2.0]), np.ones(3), DistanceRecalibration()
            )
        with self.assertRaises(ValueError):
            DistanceRecalibration(scale=0.0)


class TestCredibleLevel(unittest.TestCase):
    """The weighted credible level of a value."""

    def test_uniform_weights(self) -> None:
        """Half the samples lie below the middle."""
        x = np.arange(100.0)
        self.assertAlmostEqual(credible_level(x, np.ones(100), 49.5), 0.5)

    def test_weights(self) -> None:
        """Weights count."""
        x = np.array([1.0, 2.0, 3.0])
        level = credible_level(x, np.array([1.0, 1.0, 2.0]), 2.5)
        self.assertAlmostEqual(level, 0.5)


class TestFit(unittest.TestCase):
    """Fitting the shift and width factor."""

    def test_recovers_known_bias_and_overconfidence(self) -> None:
        """Posteriors 30% too narrow, with the median 0.07 above the truth."""
        events = _events(np.random.default_rng(1), offset=0.07, spread=1.3, n=300)
        cal = fit_distance_recalibration(events, prior_power=0.0)
        self.assertAlmostEqual(cal.shift, 0.07, delta=0.04)
        self.assertAlmostEqual(cal.scale, 1.3, delta=0.12)

    def test_calibrated_input_is_left_alone(self) -> None:
        """Truths drawn from the posteriors need no correction."""
        events = _events(np.random.default_rng(2), offset=0.0, spread=1.0, n=400)
        cal = fit_distance_recalibration(events, prior_power=0.0)
        self.assertAlmostEqual(cal.shift, 0.0, delta=0.04)
        self.assertAlmostEqual(cal.scale, 1.0, delta=0.08)

    def test_fit_is_calibrating(self) -> None:
        """Applying the fit to fresh events gives unit z-spread and zero mean."""
        train = _events(np.random.default_rng(3), 0.07, 1.3, 300)
        cal = fit_distance_recalibration(train)
        z = []
        for d, w, truth in _events(np.random.default_rng(4), 0.07, 1.3, 300):
            new_d, new_w = recalibrate_distance(d, w, cal)
            level = credible_level(np.log(new_d), new_w, np.log(truth))
            z.append(ndtri(np.clip(level, 1e-3, 1 - 1e-3)))
        self.assertAlmostEqual(float(np.mean(z)), 0.0, delta=0.15)
        self.assertAlmostEqual(float(np.std(z)), 1.0, delta=0.1)

    def test_needs_events(self) -> None:
        """An empty catalogue cannot be fitted."""
        with self.assertRaises(ValueError):
            fit_distance_recalibration([])


class TestDirectory(unittest.TestCase):
    """Writing recalibrated posterior files."""

    @staticmethod
    def _write(root: Path, name: str, header: str, rng: np.random.Generator) -> None:
        out = root / name / "run" / "output"
        out.mkdir(parents=True)
        d = np.exp(np.log(1000.0) + 0.2 * rng.standard_normal(500))
        columns = [d, np.ones(500), rng.uniform(-1, 1, 500)]
        data = np.column_stack(columns[: len(header.split())])
        np.savetxt(out / "posterior_samples.dat", data, header=header, comments="")

    def test_round_trip_and_record(self) -> None:
        """Only distance and weights change; the parameters are recorded."""
        rng = np.random.default_rng(5)
        header = "luminosity_distance weights cos_theta_jn"
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in", Path(tmp) / "out"
            for name in ("inj_1.0", "inj_2.0"):
                self._write(src, name, header, rng)
            cal = DistanceRecalibration(shift=0.07, scale=1.3)
            written = recalibrate_directory(src, dst, "run", cal)
            self.assertEqual(written, ["inj_1.0", "inj_2.0"])
            path = "inj_1.0/run/output/posterior_samples.dat"
            before = np.genfromtxt(src / path, names=True)
            after = np.genfromtxt(dst / path, names=True)
            self.assertEqual(before.dtype.names, after.dtype.names)
            np.testing.assert_allclose(after["cos_theta_jn"], before["cos_theta_jn"])
            self.assertFalse(
                np.allclose(after["luminosity_distance"], before["luminosity_distance"])
            )
            record = json.loads((dst / "recalibration.json").read_text())
            self.assertEqual(record["shift"], 0.07)
            self.assertEqual(record["scale"], 1.3)
            self.assertEqual(record["events"], 2)

    def test_column_aliases(self) -> None:
        """The per-branch files call the columns distance and weight."""
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in", Path(tmp) / "out"
            self._write(src, "inj_3.0", "distance weight", rng)
            recalibrate_directory(src, dst, "run", DistanceRecalibration(0.05, 1.2))
            path = "inj_3.0/run/output/posterior_samples.dat"
            after = np.genfromtxt(dst / path, names=True)
            self.assertEqual(after.dtype.names, ("distance", "weight"))

    def test_refuses_existing_output(self) -> None:
        """A non-empty output directory is never overwritten."""
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "out"
            dst.mkdir()
            (dst / "x").write_text("")
            with self.assertRaises(FileExistsError):
                recalibrate_directory(
                    Path(tmp) / "in", dst, "run", DistanceRecalibration()
                )


class TestReleaseAndReport(unittest.TestCase):
    """The release truths, the report and the command line."""

    @staticmethod
    def _release(root: Path, names: list[str], distances: list[float]) -> None:
        root.mkdir(parents=True)
        with h5py.File(root / "injections.h5", "w") as fh:
            grp = fh.create_group("injections")
            grp["event_name"] = np.array(names + ["inj_9.0"], dtype="S")
            grp["detectable"] = np.array([True] * len(names) + [False])
            grp["luminosity_distance"] = np.array(distances + [10.0])

    def test_release_truths_only_detectable(self) -> None:
        """Only the detectable events appear, with their distances."""
        with tempfile.TemporaryDirectory() as tmp:
            self._release(Path(tmp) / "rel", ["inj_1.0", "inj_2.0"], [500.0, 900.0])
            truths = release_truths(Path(tmp) / "rel")
        self.assertEqual(truths, {"inj_1.0": 500.0, "inj_2.0": 900.0})

    def test_blueprint_truths(self) -> None:
        """True distances come from the blueprints, skipping events with no skymap."""
        text = (
            "name: inj_1.0\nlocalization file: a.fits\ninjection:\n  parameters:\n"
            "    luminosity_distance: 500.0\n"
            "---\nname: inj_2.0\nlocalization file: b.fits\ninjection:\n"
            "  parameters:\n    luminosity_distance: 900.5\n"
            "---\nname: inj_3.0\ninjection:\n  parameters:\n"
            "    luminosity_distance: 100.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blueprints.yaml"
            path.write_text(text)
            self.assertEqual(
                blueprint_truths(path), {"inj_1.0": 500.0, "inj_2.0": 900.5}
            )

    def test_report_of_a_calibrated_catalogue(self) -> None:
        """Truths drawn from the posteriors have coverage near 90% and unit spread."""
        events = _events(np.random.default_rng(7), 0.0, 1.0, 400)
        report = calibration_report(events, DistanceRecalibration(), prior_power=0.0)
        self.assertEqual(report["events"], 400)
        self.assertAlmostEqual(report["coverage_90"], 0.9, delta=0.05)
        self.assertAlmostEqual(report["std_z"], 1.0, delta=0.1)

    def test_command_line(self) -> None:
        """The command fits, writes the posteriors and records the diagnostics."""
        from bagpuss.cli import mdc_recalibrate_posteriors

        rng = np.random.default_rng(8)
        events = _events(rng, 0.1, 1.3, 40)
        names = [f"inj_{i}.0" for i in range(40)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._release(root / "rel", names, [e[2] for e in events])
            for name, (d, w, _) in zip(names, events, strict=True):
                out = root / "work" / name / "run" / "output"
                out.mkdir(parents=True)
                np.savetxt(
                    out / "posterior_samples.dat",
                    np.column_stack([d, w]),
                    header="luminosity_distance weights",
                    comments="",
                )
            result = CliRunner().invoke(
                mdc_recalibrate_posteriors,
                [
                    str(root / "rel"),
                    str(root / "work"),
                    "--run",
                    "run",
                    "--out-dir",
                    str(root / "out"),
                    "--prior-power",
                    "0",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            record = json.loads((root / "out" / "recalibration.json").read_text())
            group = record["groups"]["run"]
            self.assertEqual(record["events"], 40)
            self.assertEqual(group["events"], 40)
            self.assertLess(group["after"]["std_z"] - 1.0, 0.1)
            self.assertGreater(group["scale"], 1.0)
            self.assertTrue((root / "out" / "missing_events.txt").exists())


class TestGroups(unittest.TestCase):
    """Per-group recalibration and the list of events left out."""

    @staticmethod
    def _put(root: Path, name: str, run: str, d: np.ndarray, w: np.ndarray) -> None:
        out = root / name / run / "output"
        out.mkdir(parents=True)
        np.savetxt(
            out / "posterior_samples.dat",
            np.column_stack([d, w]),
            header="luminosity_distance weights",
            comments="",
        )

    def _project(self, root: Path) -> dict[str, float]:
        """Events 0-79 in run a (1.3 too narrow), 80-159 in run b (1.8)."""
        truths: dict[str, float] = {}
        groups = [("a", 0, 80, 1.3), ("b", 80, 160, 1.8)]
        for run, lo, hi, spread in groups:
            events = _events(np.random.default_rng(lo), 0.07, spread, hi - lo)
            for i, (d, w, truth) in zip(range(lo, hi), events, strict=True):
                name = f"inj_{i}.0"
                truths[name] = truth
                self._put(root, name, run, d, w)
        # event 0 also has a run-b posterior: run a, listed first, takes precedence
        d, w, _ = _events(np.random.default_rng(99), 0.0, 1.0, 1)[0]
        self._put(root, "inj_0.0", "b", d, w)
        # events with no combined posterior: one has a single-branch file
        for name in ("inj_200.0", "inj_201.0"):
            truths[name] = 1000.0
        branch = root / "inj_200.0" / "a" / "_scipy_plus_metric" / "output"
        branch.mkdir(parents=True)
        (branch / "posterior_samples.dat").write_text("distance weight\n1 1\n")
        return truths

    def test_groups_are_fitted_separately(self) -> None:
        """Each run gets its own shift and width factor."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truths = self._project(root / "work")
            record = recalibrate_catalogue(
                root / "work", root / "out", ["a", "b"], truths, 0.0, n_splits=5
            )
            groups = record["groups"]
            self.assertEqual(groups["a"]["events"], 80)
            self.assertEqual(groups["b"]["events"], 80)
            self.assertGreater(groups["b"]["scale"], groups["a"]["scale"] + 0.2)
            self.assertAlmostEqual(groups["a"]["scale"], 1.3, delta=0.2)
            self.assertAlmostEqual(groups["a"]["after"]["std_z"], 1.0, delta=0.15)
            self.assertIn("validation", groups["a"])
            self.assertTrue(
                (root / "out/inj_81.0/b/output/posterior_samples.dat").exists()
            )

    def test_first_run_takes_precedence(self) -> None:
        """An event with posteriors in both runs is written once, from the first."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truths = self._project(root / "work")
            recalibrate_catalogue(
                root / "work", root / "out", ["a", "b"], truths, 0.0, n_splits=2
            )
            self.assertTrue(
                (root / "out/inj_0.0/a/output/posterior_samples.dat").exists()
            )
            self.assertFalse((root / "out/inj_0.0/b").exists())

    def test_missing_events_listed(self) -> None:
        """Events with no combined posterior are listed with their branch files."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truths = self._project(root / "work")
            record = recalibrate_catalogue(
                root / "work", root / "out", ["a", "b"], truths, 0.0, n_splits=2
            )
            lines = (root / "out" / "missing_events.txt").read_text().splitlines()
            rows = dict(line.split("\t") for line in lines if not line.startswith("#"))
            self.assertEqual(sorted(rows), ["inj_200.0", "inj_201.0"])
            self.assertIn("scipy", rows["inj_200.0"])
            self.assertEqual(rows["inj_201.0"], "none")
            self.assertEqual(record["missing"], 2)
            self.assertEqual(record["events"], 160)

    def test_split_validation(self) -> None:
        """The held-out check of a miscalibrated sample comes out calibrated."""
        events = _events(np.random.default_rng(10), 0.07, 1.3, 120)
        result = split_validation(events, prior_power=0.0, n_splits=10)
        self.assertAlmostEqual(result["coverage_90"], 0.9, delta=0.06)
        self.assertAlmostEqual(result["std_z"], 1.0, delta=0.15)
        self.assertGreater(result["scale_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
