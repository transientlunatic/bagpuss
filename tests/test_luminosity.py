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
