"""Tests for bagpuss.population_inference."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import Planck18

from bagpuss.population import PowerLawPlusPeakMassDistribution
from bagpuss.population_inference import (
    event_samples,
    load_release,
    selection_data,
    simplepe_samples,
    summarise,
    true_hyperparameters,
    unsupported_events,
)
from bagpuss.sensitivity import InjectionDistribution, draw_injections

_HAVE_GWPOPULATION = (
    importlib.util.find_spec("gwpopulation") is not None
    and importlib.util.find_spec("bilby") is not None
)
_T0 = 1_000_000_000.0
_T1 = _T0 + 1.0e6
_Z_MAX = 3.0


def _mass() -> PowerLawPlusPeakMassDistribution:
    return PowerLawPlusPeakMassDistribution(
        alpha=3.5,
        beta_q=1.4,
        m_min=5.0,
        m_max=87.0,
        lambda_peak=0.03,
        mu_m=34.0,
        sigma_m=3.6,
        delta_m=4.8,
    )


def _dist() -> InjectionDistribution:
    return InjectionDistribution(
        cosmology=Planck18,
        t_start=_T0,
        t_end=_T1,
        z_max=_Z_MAX,
        m_min=3.0,
        m_max=100.0,
        alpha=2.0,
        beta_q=0.0,
    )


class TestSummarise(unittest.TestCase):
    """Tests for summarise."""

    def test_truth_inside_and_outside(self) -> None:
        """The truth is flagged inside or outside the credible interval."""
        rng = np.random.default_rng(0)
        samples = {"a": rng.normal(0.0, 1.0, 20000), "b": rng.normal(5.0, 1.0, 20000)}
        rows = {r.name: r for r in summarise(samples, {"a": 0.1, "b": 0.0, "c": 1.0})}
        self.assertEqual(set(rows), {"a", "b"})
        self.assertTrue(rows["a"].inside)
        self.assertFalse(rows["b"].inside)
        self.assertAlmostEqual(rows["a"].z_score, 0.1, delta=0.05)
        self.assertLess(rows["b"].quantile, 0.01)

    def test_true_hyperparameters(self) -> None:
        """The hyperparameters use gwpopulation's names."""
        truth = true_hyperparameters(_mass())
        self.assertEqual(truth["beta"], 1.4)
        self.assertEqual(truth["lam"], 0.03)
        self.assertEqual(truth["lamb"], 0.0)


@unittest.skipUnless(_HAVE_GWPOPULATION, "pandas is not installed")
class TestSimplePESamples(unittest.TestCase):
    """The SimplePE prior in (m1, q, z) is the Jacobian of the detector-frame one."""

    def _write(
        self,
        path: Path,
        m1d: np.ndarray,
        m2d: np.ndarray,
        dl: np.ndarray,
        w: np.ndarray,
    ) -> None:
        total = m1d + m2d
        eta = m1d * m2d / total**2
        chirp = total * eta**0.6
        header = "chirp_mass symmetric_mass_ratio distance weight"
        np.savetxt(path, np.c_[chirp, eta, dl, w], header=header, comments="")

    def test_prior_is_flat_in_source_frame_variables(self) -> None:
        """Samples from the PE prior, weighted by 1/prior, are uniform in a box."""
        rng = np.random.default_rng(0)
        n = 400_000
        m = np.sort(rng.uniform(20.0, 100.0, size=(n, 2)), axis=1)[:, ::-1]
        # p(d_L) proportional to d_L**2 on [100, 3000]
        dl = (rng.uniform(size=n) * (3000.0**3 - 100.0**3) + 100.0**3) ** (1 / 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "posterior_samples.dat"
            self._write(path, m[:, 0], m[:, 1], dl, np.ones(n))
            df = simplepe_samples(path, Planck18, n_samples=None)
        box = (
            (df.mass_1 > 30.0)
            & (df.mass_1 < 50.0)
            & (df.mass_ratio > 0.8)
            & (df.redshift > 0.05)
            & (df.redshift < 0.3)
        )
        w = 1.0 / df.prior[box].to_numpy()
        for name, centre in (
            ("mass_1", 40.0),
            ("mass_ratio", 0.9),
            ("redshift", 0.175),
        ):
            mean = float(np.average(df[name][box].to_numpy(), weights=w))  # pyright: ignore[reportAttributeAccessIssue]
            self.assertAlmostEqual(mean / centre, 1.0, delta=0.02, msg=name)

    def test_weights_enter_the_prior(self) -> None:
        """A sample with twice the weight has half the prior."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "posterior_samples.dat"
            self._write(
                path,
                np.array([40.0, 40.0]),
                np.array([30.0, 30.0]),
                np.array([500.0, 500.0]),
                np.array([1.0, 2.0]),
            )
            df = simplepe_samples(path, Planck18, n_samples=None)
        self.assertAlmostEqual(float(df.prior[0] / df.prior[1]), 2.0)
        self.assertAlmostEqual(float(df.mass_ratio[0]), 0.75)


class TestSelectionData(unittest.TestCase):
    """Tests for selection_data."""

    def test_snr_shape_checked(self) -> None:
        """A wrong-length SNR raises."""
        inj = draw_injections(_dist(), 100, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            selection_data(inj, np.ones(5), 8.0, _dist())

    def test_only_found_injections_are_kept(self) -> None:
        """Only injections above threshold are returned, with n_total intact."""
        inj = draw_injections(_dist(), 2000, np.random.default_rng(0))
        snr = 12_000.0 / inj.luminosity_distance
        data = selection_data(inj, snr, 8.0, _dist())
        self.assertEqual(len(data["prior"]), int((snr >= 8.0).sum()))
        self.assertEqual(data["total_generated"], 2000)
        self.assertTrue(np.all(data["mass_ratio"] <= 1.0))
        self.assertTrue(np.all(data["prior"] > 0.0))


@unittest.skipUnless(_HAVE_GWPOPULATION, "gwpopulation and bilby are not installed")
class TestAgreementWithGWPopulation(unittest.TestCase):
    """The conventions of bagpuss and gwpopulation agree."""

    def test_event_samples(self) -> None:
        """One sample per event, at the true values."""
        out = event_samples(np.array([30.0, 40.0]), np.array([15.0, 20.0]), np.ones(2))
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(float(out[0]["mass_ratio"][0]), 0.5)  # pyright: ignore[reportArgumentType]

    def test_mass_density(self) -> None:
        """The gwpopulation mass model is the bagpuss one, in (m1, q)."""
        from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution

        mass = _mass()
        truth = true_hyperparameters(mass)
        rng = np.random.default_rng(1)
        m1, m2 = mass.sample(4000, rng)
        data = {"mass_1": m1, "mass_ratio": m2 / m1}
        model = SinglePeakSmoothedMassDistribution()
        names = ("alpha", "beta", "mmin", "mmax", "lam", "mpp", "sigpp", "delta_m")
        theirs = np.asarray(model(data, **{k: truth[k] for k in names}))
        ours = np.exp(mass.log_prob(m1, m2)) * m1
        # gwpopulation normalises on a coarse grid, which matters only in the
        # steep taper at the lowest masses: 98 per cent of samples agree to 1 per cent.
        rel = np.abs(theirs / ours - 1.0)
        self.assertLess(np.mean(rel > 0.01), 0.02)
        self.assertLess(rel.max(), 0.1)

    def test_redshift_density(self) -> None:
        """PowerLawRedshift with lamb=0 is a constant source-frame rate."""
        from gwpopulation.models.redshift import PowerLawRedshift

        z = np.linspace(0.05, 2.9, 50)
        theirs = np.asarray(
            PowerLawRedshift(z_max=_Z_MAX, cosmo_model="Planck18").probability(
                {"redshift": z}, lamb=0.0
            )
        )
        ours = _dist().redshift_pdf(z)
        np.testing.assert_allclose(theirs, ours, rtol=0.01)

    def test_selection_matches_detection_probability(self) -> None:
        """ResamplingVT on selection_data reproduces detection_probability."""
        from bilby.hyper.model import Model
        from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
        from gwpopulation.models.redshift import PowerLawRedshift
        from gwpopulation.vt import ResamplingVT

        dist = _dist()
        mass = _mass()
        inj = draw_injections(dist, 60_000, np.random.default_rng(2))
        snr = 6_000.0 / inj.luminosity_distance
        estimate = inj.detection_probability(
            lambda p: dist.log_prob(p, mass_log_prob=mass.log_prob), snr, 8.0
        )
        data = selection_data(inj, snr, 8.0, dist)
        vt = ResamplingVT(
            model=Model(
                [
                    SinglePeakSmoothedMassDistribution(),
                    PowerLawRedshift(z_max=_Z_MAX, cosmo_model="Planck18"),
                ]
            ),
            data=data,
            n_events=1,
        )
        mu, _ = vt.detection_efficiency(true_hyperparameters(mass))
        self.assertGreater(estimate.value, 0.0)
        self.assertAlmostEqual(mu / estimate.value, 1.0, delta=0.03)


@unittest.skipUnless(_HAVE_GWPOPULATION, "gwpopulation and bilby are not installed")
class TestUnsupportedEvents(unittest.TestCase):
    """unsupported_events finds posteriors which exclude the true population."""

    def test_flags_only_the_event_outside_the_support(self) -> None:
        """An event with every sample below m_min is flagged, and no other."""
        import pandas as pd

        from bagpuss.population_inference import ReleaseInputs

        def frame(m1: float, q: float) -> pd.DataFrame:
            n = 50
            return pd.DataFrame(
                {
                    "mass_1": np.full(n, m1),
                    "mass_ratio": np.full(n, q),
                    "redshift": np.full(n, 0.1),
                    "prior": np.ones(n),
                }
            )

        inputs = ReleaseInputs(
            posteriors=[frame(30.0, 0.8), frame(9.0, 0.4), frame(40.0, 0.9)],
            selection={},
            truth=true_hyperparameters(_mass()),
            z_max=_Z_MAX,
            cosmology="Planck18",
            event_names=["a", "b", "c"],
        )
        self.assertEqual(unsupported_events(inputs, (200, 100)), ["b"])


class TestLoadRelease(unittest.TestCase):
    """load_release on a tiny release built with the fake waveform engine."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a tiny release with a sensitivity campaign."""
        from bagpuss.mdc.release import build_release
        from tests.test_mdc_release import _SENSITIVITY, _run_pipeline

        cls._tmp = tempfile.TemporaryDirectory()
        cls.cfg = _run_pipeline(cls._tmp.name, **_SENSITIVITY)
        cls.out = Path(cls._tmp.name) / "release"
        build_release(cls.cfg, cls.out)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the scratch directory."""
        cls._tmp.cleanup()

    @unittest.skipUnless(_HAVE_GWPOPULATION, "pandas is not installed")
    def test_events_and_selection(self) -> None:
        """One sample per detectable event; every draw counted in the selection."""
        with h5py.File(self.out / "injections.h5", "r") as fh:
            n_detectable = int(np.sum(fh["injections"]["detectable"]))  # type: ignore[index]
        inputs = load_release(self.out, self.cfg)
        self.assertEqual(len(inputs.posteriors), n_detectable)
        self.assertEqual(
            inputs.selection["total_generated"],
            self.cfg.sensitivity_n_shards * self.cfg.sensitivity_n_per_shard,
        )
        self.assertEqual(inputs.truth["alpha"], self.cfg.mass_alpha)
        self.assertEqual(inputs.z_max, self.cfg.z_max)

    def test_missing_sensitivity_raises(self) -> None:
        """A release without sensitivity.h5 is rejected."""
        (self.out / "sensitivity.h5").rename(self.out / "moved.h5")
        try:
            with self.assertRaises(ValueError):
                load_release(self.out, self.cfg)
        finally:
            (self.out / "moved.h5").rename(self.out / "sensitivity.h5")


if __name__ == "__main__":
    unittest.main()
