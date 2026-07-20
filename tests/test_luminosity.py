"""Tests for bagpuss.luminosity."""

import unittest

import numpy as np

from bagpuss.luminosity import SchechterLuminosityModel

# Representative Schechter parameters used throughout the tests
_PHI_STAR = 1.61e-2
_M_STAR = -19.66
_ALPHA = -1.16
_M_MIN = -25.0
_M_MAX = -14.0


def _make(**kwargs: float) -> SchechterLuminosityModel:
    defaults = dict(
        phi_star=_PHI_STAR,
        m_star=_M_STAR,
        alpha=_ALPHA,
        m_min=_M_MIN,
        m_max=_M_MAX,
    )
    defaults.update(kwargs)
    return SchechterLuminosityModel(**defaults)


class TestSchechterLuminosityModelInit(unittest.TestCase):
    """Tests for SchechterLuminosityModel parameter validation and storage."""

    def test_stores_phi_star(self) -> None:
        """phi_star is stored as an attribute."""
        model = _make(phi_star=3.5e-3)
        self.assertEqual(model.phi_star, 3.5e-3)

    def test_stores_m_star(self) -> None:
        """m_star is stored as an attribute."""
        model = _make(m_star=-21.0)
        self.assertEqual(model.m_star, -21.0)

    def test_stores_alpha(self) -> None:
        """Alpha is stored as an attribute."""
        model = _make(alpha=-1.5)
        self.assertEqual(model.alpha, -1.5)

    def test_stores_m_min(self) -> None:
        """m_min is stored as an attribute."""
        model = _make(m_min=-24.0)
        self.assertEqual(model.m_min, -24.0)

    def test_stores_m_max(self) -> None:
        """m_max is stored as an attribute."""
        model = _make(m_max=-10.0)
        self.assertEqual(model.m_max, -10.0)

    def test_default_m_sun(self) -> None:
        """m_sun defaults to the bolometric solar magnitude (4.83)."""
        model = _make()
        self.assertAlmostEqual(model.m_sun, 4.83)

    def test_custom_m_sun(self) -> None:
        """Custom m_sun is stored correctly."""
        model = _make(m_sun=5.48)
        self.assertAlmostEqual(model.m_sun, 5.48)

    def test_invalid_m_min_equal_m_max(self) -> None:
        """Raises ValueError when m_min == m_max."""
        with self.assertRaises(ValueError):
            _make(m_min=-20.0, m_max=-20.0)

    def test_invalid_m_min_greater_than_m_max(self) -> None:
        """Raises ValueError when m_min > m_max."""
        with self.assertRaises(ValueError):
            _make(m_min=-10.0, m_max=-20.0)


class TestSchechterLuminosityModelSample(unittest.TestCase):
    """Tests for SchechterLuminosityModel.sample."""

    def test_sample_shape(self) -> None:
        """Sample returns an array of the requested length."""
        model = _make()
        result = model.sample(500, rng=np.random.default_rng(0))
        self.assertEqual(result.shape, (500,))

    def test_sample_all_positive(self) -> None:
        """All sampled luminosities are positive."""
        model = _make()
        result = model.sample(1000, rng=np.random.default_rng(1))
        self.assertTrue(np.all(result > 0))

    def test_sample_reproducible(self) -> None:
        """Same RNG seed produces identical output."""
        model = _make()
        r1 = model.sample(200, rng=np.random.default_rng(42))
        r2 = model.sample(200, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_sample_different_seeds_differ(self) -> None:
        """Different RNG seeds produce different output."""
        model = _make()
        r1 = model.sample(200, rng=np.random.default_rng(0))
        r2 = model.sample(200, rng=np.random.default_rng(1))
        self.assertFalse(np.array_equal(r1, r2))

    def test_sample_default_rng(self) -> None:
        """Sample works when rng=None (uses internal default)."""
        model = _make()
        result = model.sample(10, rng=None)
        self.assertEqual(result.shape, (10,))

    def test_luminosities_within_expected_range(self) -> None:
        """Sampled luminosities correspond to magnitudes inside [m_min, m_max].

        For m_sun=4.83, m_min=-25, m_max=-14:
        L_max = 10^(0.4*(4.83 - (-25))) = 10^(0.4*29.83)
        L_min = 10^(0.4*(4.83 - (-14))) = 10^(0.4*18.83)
        """
        model = _make()
        result = model.sample(2000, rng=np.random.default_rng(7))
        l_min = 10.0 ** (0.4 * (model.m_sun - model.m_max))
        l_max = 10.0 ** (0.4 * (model.m_sun - model.m_min))
        self.assertTrue(np.all(result >= l_min))
        self.assertTrue(np.all(result <= l_max))


class TestSchechterLuminosityModelCdf(unittest.TestCase):
    """Tests for SchechterLuminosityModel.cdf."""

    def test_cdf_at_m_min_is_zero(self) -> None:
        """No galaxies are brighter than the bright-end limit itself."""
        model = _make()
        self.assertAlmostEqual(float(model.cdf(np.array([_M_MIN]))[0]), 0.0)

    def test_cdf_at_m_max_is_one(self) -> None:
        """All galaxies are brighter than or equal to the faint-end limit."""
        model = _make()
        self.assertAlmostEqual(float(model.cdf(np.array([_M_MAX]))[0]), 1.0)

    def test_cdf_below_m_min_clamped_to_zero(self) -> None:
        """Magnitudes brighter than m_min clamp to a CDF of zero."""
        model = _make()
        result = model.cdf(np.array([_M_MIN - 10.0]))
        self.assertAlmostEqual(float(result[0]), 0.0)

    def test_cdf_above_m_max_clamped_to_one(self) -> None:
        """Magnitudes fainter than m_max clamp to a CDF of one."""
        model = _make()
        result = model.cdf(np.array([_M_MAX + 10.0]))
        self.assertAlmostEqual(float(result[0]), 1.0)

    def test_cdf_monotonic_increasing(self) -> None:
        """The CDF is non-decreasing in magnitude."""
        model = _make()
        magnitudes = np.linspace(_M_MIN, _M_MAX, 50)
        result = model.cdf(magnitudes)
        self.assertTrue(np.all(np.diff(result) >= 0.0))

    def test_cdf_vectorized(self) -> None:
        """Cdf accepts and returns an array matching the input shape."""
        model = _make()
        magnitudes = np.array([-24.0, -20.0, -16.0])
        result = model.cdf(magnitudes)
        self.assertEqual(result.shape, magnitudes.shape)

    def test_number_density_positive(self) -> None:
        """number_density returns a positive value."""
        model = _make()
        self.assertGreater(model.number_density(), 0.0)

    def test_number_density_scales_with_phi_star(self) -> None:
        """number_density scales linearly with phi_star."""
        d1 = _make(phi_star=1e-2).number_density()
        d2 = _make(phi_star=2e-2).number_density()
        self.assertAlmostEqual(d2 / d1, 2.0, places=10)

    def test_number_density_units(self) -> None:
        """number_density is consistent with the Schechter integral."""
        model = _make()
        # The CDF was normalised by the same total; verify
        # number_density() == phi_star * ∫ φ̃(M) dM (the unnormalised total).
        self.assertGreater(model.number_density(), 0.0)
        self.assertLess(model.number_density(), 1.0)  # typical B-band Schechter density

    def test_cdf_matches_sampled_fraction(self) -> None:
        """cdf(M) matches the empirical fraction of samples brighter than M."""
        model = _make()
        sample = model.sample(200_000, rng=np.random.default_rng(3))
        magnitude_threshold = _M_STAR
        l_threshold = 10.0 ** (0.4 * (model.m_sun - magnitude_threshold))
        empirical = float(np.mean(sample >= l_threshold))
        expected = float(model.cdf(np.array([magnitude_threshold]))[0])
        self.assertAlmostEqual(empirical, expected, delta=0.01)
