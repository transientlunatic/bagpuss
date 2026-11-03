"""Tests for bagpuss.sensitivity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18

from bagpuss.sensitivity import (
    DetectionEstimate,
    InjectionDistribution,
    SensitivityInjectionSet,
    draw_injections,
)

_T0 = 1_000_000_000.0
_T1 = _T0 + 1.0e6


def _dist(**kwargs: float) -> InjectionDistribution:
    return InjectionDistribution(cosmology=Planck18, t_start=_T0, t_end=_T1, **kwargs)


def _toy_snr(
    injections: SensitivityInjectionSet, scale: float = 12_000.0
) -> np.ndarray:
    """Return a mass-independent toy detector's SNR, ``scale / d_L``."""
    return scale / injections.luminosity_distance


class TestInjectionDistribution(unittest.TestCase):
    """Tests for InjectionDistribution."""

    def test_samples_lie_within_bounds(self) -> None:
        """Samples lie within bounds."""
        dist = _dist(z_max=2.0, m_min=3.0, m_max=100.0)
        inj = draw_injections(dist, 5000, np.random.default_rng(0))
        self.assertEqual(len(inj), 5000)
        self.assertEqual(inj.n_total, 5000)
        self.assertTrue(np.all((inj.redshift > 0) & (inj.redshift <= 2.0)))
        self.assertTrue(np.all((inj.m1_source >= 3.0) & (inj.m1_source <= 100.0)))
        self.assertTrue(np.all(inj.m2_source >= 3.0))
        self.assertTrue(np.all(inj.m2_source <= inj.m1_source))
        self.assertTrue(np.all((inj.a1 >= 0) & (inj.a1 <= 1)))
        self.assertTrue(np.all(np.abs(inj.cos_tilt1) <= 1))
        self.assertTrue(np.all((inj.theta_jn >= 0) & (inj.theta_jn <= np.pi)))
        self.assertTrue(np.all((inj.ra >= 0) & (inj.ra < 2 * np.pi)))
        self.assertTrue(np.all(np.abs(inj.dec) <= np.pi / 2))
        self.assertTrue(np.all((inj.geocent_time >= _T0) & (inj.geocent_time <= _T1)))

    def test_luminosity_distance_follows_fiducial_cosmology(self) -> None:
        """Luminosity distance follows fiducial cosmology."""
        inj = draw_injections(_dist(), 200, np.random.default_rng(1))
        expected = Planck18.luminosity_distance(inj.redshift).to("Mpc").value  # pyright: ignore[reportAttributeAccessIssue]
        np.testing.assert_allclose(inj.luminosity_distance, expected)

    def test_reproducible_with_seed(self) -> None:
        """Reproducible with seed."""
        a = draw_injections(_dist(), 100, np.random.default_rng(7))
        b = draw_injections(_dist(), 100, np.random.default_rng(7))
        np.testing.assert_array_equal(a.m1_source, b.m1_source)
        np.testing.assert_array_equal(a.log_p_draw, b.log_p_draw)

    def test_log_p_draw_matches_log_prob(self) -> None:
        """Log p draw matches log prob."""
        dist = _dist()
        inj = draw_injections(dist, 500, np.random.default_rng(2))
        np.testing.assert_allclose(inj.log_p_draw, dist.log_prob(inj.as_dict()))

    def test_densities_are_normalised(self) -> None:
        """Densities are normalised."""
        """E_draw[p_alt / p_draw] = 1 for any normalised alternative density.

        This checks sampling and ``log_prob`` against each other, and the
        normalisation of both distributions, without a high-dimensional integral.
        """
        draw = _dist(z_max=3.0, m_min=3.0, m_max=200.0, alpha=2.0, beta_q=0.0)
        alt = _dist(z_max=1.5, m_min=5.0, m_max=90.0, alpha=3.0, beta_q=1.5, kappa=1.0)
        inj = draw_injections(draw, 400_000, np.random.default_rng(3))
        log_ratio = alt.log_prob(inj.as_dict()) - inj.log_p_draw
        w = np.exp(log_ratio)
        # alt is supported on a sub-volume of draw, so E[w] = 1 exactly.
        mean, sem = w.mean(), w.std() / np.sqrt(len(w))
        self.assertLess(abs(mean - 1.0), 4 * sem)

    def test_log_prob_is_minus_inf_outside_support(self) -> None:
        """Log prob is minus inf outside support."""
        dist = _dist(z_max=1.0)
        inj = draw_injections(_dist(z_max=3.0), 2000, np.random.default_rng(4))
        lp = dist.log_prob(inj.as_dict())
        self.assertTrue(np.all(np.isneginf(lp[inj.redshift > 1.0])))
        self.assertTrue(np.all(np.isfinite(lp[inj.redshift <= 1.0])))

    def test_rejects_invalid_parameters(self) -> None:
        """Rejects invalid parameters."""
        with self.assertRaises(ValueError):
            _dist(z_max=0.0)
        with self.assertRaises(ValueError):
            _dist(m_min=10.0, m_max=5.0)
        with self.assertRaises(ValueError):
            InjectionDistribution(cosmology=Planck18, t_start=2.0, t_end=1.0)


class TestDetectionEstimate(unittest.TestCase):
    """Tests for the importance-sampling estimate of P(D|Lambda)."""

    def test_target_equal_to_draw_recovers_detected_fraction(self) -> None:
        """Target equal to draw recovers detected fraction."""
        dist = _dist()
        inj = draw_injections(dist, 20_000, np.random.default_rng(5))
        est = inj.detection_probability(
            dist.log_prob, snr=_toy_snr(inj), snr_threshold=8.0
        )
        found = _toy_snr(inj) >= 8.0
        self.assertAlmostEqual(est.value, found.mean(), places=12)
        self.assertEqual(est.n_found, int(found.sum()))
        # Farr (2019): N_eff = mu^2 / sigma^2 = N f / (1 - f) for unit weights.
        f = found.mean()
        self.assertAlmostEqual(est.n_eff, len(inj) * f / (1.0 - f), places=6)

    def test_recovers_analytic_detection_probability(self) -> None:
        """Recovers analytic detection probability."""
        draw = _dist(z_max=3.0)
        target = _dist(z_max=1.0)
        inj = draw_injections(draw, 300_000, np.random.default_rng(6))
        est = inj.detection_probability(
            target.log_prob, snr=_toy_snr(inj), snr_threshold=8.0
        )

        # The toy detector finds everything with d_L <= 12000/8 Mpc.
        d_max = 12_000.0 / 8.0
        z_star = float(
            np.interp(
                d_max,
                Planck18.luminosity_distance(np.linspace(1e-4, 1.0, 4000)).value,  # pyright: ignore[reportAttributeAccessIssue]
                np.linspace(1e-4, 1.0, 4000),
            )
        )
        z = np.linspace(1e-6, z_star, 20_000)
        pdf = target.redshift_pdf(z)
        truth = float(np.sum(0.5 * (pdf[:-1] + pdf[1:]) * np.diff(z)))
        self.assertGreater(est.std, 0.0)
        self.assertLess(abs(est.value - truth), 4 * est.std)
        self.assertGreater(truth, 0.01)

    def test_hopeless_cut_still_counts_in_n_total(self) -> None:
        """Hopeless cut still counts in n total."""
        dist = _dist()
        inj = draw_injections(dist, 30_000, np.random.default_rng(8))
        snr = _toy_snr(inj)
        full = inj.detection_probability(dist.log_prob, snr=snr, snr_threshold=8.0)

        keep = snr > 4.0  # discard the hopeless injections, keep the count
        cut = inj.select(keep)
        self.assertEqual(cut.n_total, 30_000)
        self.assertLess(len(cut), 30_000)
        cut_est = cut.detection_probability(
            dist.log_prob, snr=snr[keep], snr_threshold=8.0
        )
        self.assertAlmostEqual(cut_est.value, full.value, places=12)

    def test_n_eff_falls_with_uneven_weights(self) -> None:
        """N eff falls with uneven weights."""
        draw = _dist(z_max=3.0)
        inj = draw_injections(draw, 50_000, np.random.default_rng(9))
        even = inj.detection_probability(
            draw.log_prob, snr=_toy_snr(inj), snr_threshold=8.0
        )
        skewed = inj.detection_probability(
            _dist(z_max=0.5, alpha=3.5).log_prob,
            snr=_toy_snr(inj),
            snr_threshold=8.0,
        )
        self.assertLess(skewed.n_eff, even.n_eff)

    def test_no_detections_gives_zero(self) -> None:
        """No detections gives zero."""
        dist = _dist()
        inj = draw_injections(dist, 100, np.random.default_rng(10))
        est = inj.detection_probability(
            dist.log_prob, snr=np.zeros(len(inj)), snr_threshold=8.0
        )
        self.assertEqual(est.value, 0.0)
        self.assertEqual(est.n_found, 0)
        self.assertEqual(est.n_eff, 0.0)

    def test_is_reliable_uses_the_neff_criterion(self) -> None:
        """Is reliable uses the neff criterion."""
        est = DetectionEstimate(
            value=0.1, std=0.01, n_eff=100.0, n_found=200, n_total=1000
        )
        self.assertTrue(est.is_reliable(n_events=25))
        self.assertFalse(est.is_reliable(n_events=26))

    def test_snr_length_must_match(self) -> None:
        """Snr length must match."""
        dist = _dist()
        inj = draw_injections(dist, 10, np.random.default_rng(11))
        with self.assertRaises(ValueError):
            inj.detection_probability(dist.log_prob, snr=np.zeros(3), snr_threshold=8.0)


class TestSensitivityInjectionSetIO(unittest.TestCase):
    """Tests for SensitivityInjectionSet I/O and joining."""

    def setUp(self) -> None:
        """Create the shared fixture."""
        self.inj = draw_injections(_dist(), 50, np.random.default_rng(12)).select(
            np.arange(50) % 2 == 0
        )

    def test_hdf5_round_trip(self) -> None:
        """Hdf5 round trip."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sens.h5"
            self.inj.to_hdf5(path, network_snr=np.arange(25.0))
            loaded, snr = SensitivityInjectionSet.from_hdf5(path, with_snr=True)
        self.assertEqual(len(loaded), 25)
        self.assertEqual(loaded.n_total, 50)
        np.testing.assert_array_equal(loaded.m1_source, self.inj.m1_source)
        np.testing.assert_array_equal(loaded.log_p_draw, self.inj.log_p_draw)
        np.testing.assert_array_equal(snr, np.arange(25.0))

    def test_zarr_round_trip(self) -> None:
        """Zarr round trip."""
        with tempfile.TemporaryDirectory() as tmp:
            self.inj.to_zarr(Path(tmp) / "s.zarr")
            loaded = SensitivityInjectionSet.from_zarr(Path(tmp) / "s.zarr")
        self.assertEqual(loaded.n_total, 50)
        np.testing.assert_array_equal(loaded.redshift, self.inj.redshift)

    def test_concatenate_sums_n_total(self) -> None:
        """Concatenate sums n total."""
        both = SensitivityInjectionSet.concatenate([self.inj, self.inj])
        self.assertEqual(len(both), 50)
        self.assertEqual(both.n_total, 100)

    def test_concatenate_rejects_empty(self) -> None:
        """Concatenate rejects empty."""
        with self.assertRaises(ValueError):
            SensitivityInjectionSet.concatenate([])


if __name__ == "__main__":
    unittest.main()
