"""Tests for bagpuss.population."""

import unittest

import numpy as np

from bagpuss.population import (
    BBHSet,
    DefaultSpinDistribution,
    IsotropicSpinDistribution,
    MassDistribution,
    PopulationModel,
    PowerLawPlusPeakMassDistribution,
    SpinDistribution,
)

# ---------------------------------------------------------------------------
# Shared parameter sets
# ---------------------------------------------------------------------------

# Simple power-law-only params (lambda_peak=0, delta_m=0) for easy testing
_MASS_PARAMS = dict(
    alpha=2.5,
    beta_q=1.0,
    m_min=5.0,
    m_max=50.0,
    lambda_peak=0.0,
    mu_m=30.0,
    sigma_m=5.0,
    delta_m=0.0,
)

# Full params with a small Gaussian peak and smoothing
_MASS_PARAMS_FULL = dict(
    alpha=3.5,
    beta_q=1.4,
    m_min=5.0,
    m_max=87.0,
    lambda_peak=0.03,
    mu_m=34.0,
    sigma_m=3.6,
    delta_m=4.8,
)


def _make_mass() -> PowerLawPlusPeakMassDistribution:
    return PowerLawPlusPeakMassDistribution(**_MASS_PARAMS)


def _make_isotropic_spin() -> IsotropicSpinDistribution:
    return IsotropicSpinDistribution()


# ---------------------------------------------------------------------------
# BBHSet
# ---------------------------------------------------------------------------


class TestBBHSet(unittest.TestCase):
    """Tests for the BBHSet dataclass."""

    def _make(self, n: int = 4) -> BBHSet:
        return BBHSet(
            m1_source=np.ones(n) * 30.0,
            m2_source=np.ones(n) * 20.0,
            a1=np.ones(n) * 0.5,
            a2=np.ones(n) * 0.3,
            cos_tilt1=np.ones(n) * 0.8,
            cos_tilt2=np.ones(n) * -0.2,
            phi12=np.ones(n) * 1.0,
            phi_jl=np.ones(n) * 2.0,
        )

    def test_stores_m1_source(self) -> None:
        """m1_source is stored correctly."""
        b = self._make(3)
        np.testing.assert_array_equal(b.m1_source, np.ones(3) * 30.0)

    def test_stores_m2_source(self) -> None:
        """m2_source is stored correctly."""
        b = self._make(3)
        np.testing.assert_array_equal(b.m2_source, np.ones(3) * 20.0)

    def test_stores_spins(self) -> None:
        """Spin fields are stored correctly."""
        b = self._make(2)
        np.testing.assert_array_equal(b.a1, [0.5, 0.5])
        np.testing.assert_array_equal(b.a2, [0.3, 0.3])
        np.testing.assert_array_equal(b.cos_tilt1, [0.8, 0.8])
        np.testing.assert_array_equal(b.cos_tilt2, [-0.2, -0.2])
        np.testing.assert_array_equal(b.phi12, [1.0, 1.0])
        np.testing.assert_array_equal(b.phi_jl, [2.0, 2.0])

    def test_len(self) -> None:
        """__len__ returns the number of events."""
        b = self._make(7)
        self.assertEqual(len(b), 7)

    def test_len_empty(self) -> None:
        """__len__ returns 0 for an empty set."""
        empty = np.array([])
        b = BBHSet(empty, empty, empty, empty, empty, empty, empty, empty)
        self.assertEqual(len(b), 0)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class TestAbstractBaseClasses(unittest.TestCase):
    """MassDistribution and SpinDistribution cannot be instantiated directly."""

    def test_mass_distribution_is_abstract(self) -> None:
        """MassDistribution raises TypeError on direct instantiation."""
        with self.assertRaises(TypeError):
            MassDistribution()  # type: ignore[abstract]

    def test_spin_distribution_is_abstract(self) -> None:
        """SpinDistribution raises TypeError on direct instantiation."""
        with self.assertRaises(TypeError):
            SpinDistribution()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# PopulationModel
# ---------------------------------------------------------------------------


class TestPopulationModel(unittest.TestCase):
    """Tests for PopulationModel composition."""

    def _make(self) -> PopulationModel:
        return PopulationModel(mass=_make_mass(), spin=_make_isotropic_spin())

    def test_stores_mass(self) -> None:
        """mass attribute is stored."""
        mass = _make_mass()
        pm = PopulationModel(mass=mass, spin=_make_isotropic_spin())
        self.assertIs(pm.mass, mass)

    def test_stores_spin(self) -> None:
        """spin attribute is stored."""
        spin = _make_isotropic_spin()
        pm = PopulationModel(mass=_make_mass(), spin=spin)
        self.assertIs(pm.spin, spin)

    def test_sample_returns_bbh_set(self) -> None:
        """sample() returns a BBHSet instance."""
        pm = self._make()
        result = pm.sample(10, rng=np.random.default_rng(0))
        self.assertIsInstance(result, BBHSet)

    def test_sample_length(self) -> None:
        """The returned BBHSet has exactly n events."""
        pm = self._make()
        result = pm.sample(37, rng=np.random.default_rng(0))
        self.assertEqual(len(result), 37)

    def test_sample_all_fields_same_length(self) -> None:
        """All fields of the returned BBHSet have shape (n,)."""
        pm = self._make()
        result = pm.sample(20, rng=np.random.default_rng(0))
        for field in (
            result.m1_source,
            result.m2_source,
            result.a1,
            result.a2,
            result.cos_tilt1,
            result.cos_tilt2,
            result.phi12,
            result.phi_jl,
        ):
            self.assertEqual(field.shape, (20,))


# ---------------------------------------------------------------------------
# PowerLawPlusPeakMassDistribution
# ---------------------------------------------------------------------------


class TestPowerLawPlusPeakMassDistribution(unittest.TestCase):
    """Tests for the power-law-plus-Gaussian-peak mass distribution."""

    def test_stores_parameters(self) -> None:
        """All hyperparameters are stored as attributes."""
        m = PowerLawPlusPeakMassDistribution(**_MASS_PARAMS)
        self.assertEqual(m.alpha, _MASS_PARAMS["alpha"])
        self.assertEqual(m.beta_q, _MASS_PARAMS["beta_q"])
        self.assertEqual(m.m_min, _MASS_PARAMS["m_min"])
        self.assertEqual(m.m_max, _MASS_PARAMS["m_max"])
        self.assertEqual(m.lambda_peak, _MASS_PARAMS["lambda_peak"])
        self.assertEqual(m.mu_m, _MASS_PARAMS["mu_m"])
        self.assertEqual(m.sigma_m, _MASS_PARAMS["sigma_m"])
        self.assertEqual(m.delta_m, _MASS_PARAMS["delta_m"])

    def test_invalid_m_min_ge_m_max(self) -> None:
        """m_min >= m_max raises ValueError."""
        with self.assertRaises(ValueError):
            PowerLawPlusPeakMassDistribution(
                **{**_MASS_PARAMS, "m_min": 50.0, "m_max": 50.0}
            )

    def test_invalid_lambda_peak_negative(self) -> None:
        """Negative lambda_peak raises ValueError."""
        with self.assertRaises(ValueError):
            PowerLawPlusPeakMassDistribution(**{**_MASS_PARAMS, "lambda_peak": -0.1})

    def test_invalid_lambda_peak_gt_one(self) -> None:
        """lambda_peak > 1 raises ValueError."""
        with self.assertRaises(ValueError):
            PowerLawPlusPeakMassDistribution(**{**_MASS_PARAMS, "lambda_peak": 1.1})

    def test_sample_shape(self) -> None:
        """sample() returns two arrays of the requested length."""
        m = _make_mass()
        m1, m2 = m.sample(100, rng=np.random.default_rng(0))
        self.assertEqual(m1.shape, (100,))
        self.assertEqual(m2.shape, (100,))

    def test_m1_ge_m2(self) -> None:
        """Primary mass is always at least as large as the secondary."""
        m = _make_mass()
        m1, m2 = m.sample(500, rng=np.random.default_rng(1))
        self.assertTrue(np.all(m1 >= m2))

    def test_m2_ge_m_min(self) -> None:
        """Secondary mass is always at least m_min."""
        m = _make_mass()
        m1, m2 = m.sample(500, rng=np.random.default_rng(2))
        self.assertTrue(np.all(m2 >= _MASS_PARAMS["m_min"] - 1e-10))

    def test_m1_le_m_max(self) -> None:
        """Primary mass never exceeds m_max."""
        m = _make_mass()
        m1, m2 = m.sample(500, rng=np.random.default_rng(3))
        self.assertTrue(np.all(m1 <= _MASS_PARAMS["m_max"] + 1e-10))

    def test_reproducible(self) -> None:
        """Same RNG seed produces identical output."""
        m = _make_mass()
        m1a, m2a = m.sample(50, rng=np.random.default_rng(99))
        m1b, m2b = m.sample(50, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(m1a, m1b)
        np.testing.assert_array_equal(m2a, m2b)

    def test_default_rng(self) -> None:
        """sample() works when rng=None."""
        m = _make_mass()
        m1, m2 = m.sample(10, rng=None)
        self.assertEqual(m1.shape, (10,))

    def test_with_gaussian_peak_and_smoothing(self) -> None:
        """Full model (lambda_peak > 0, delta_m > 0) produces valid samples."""
        m = PowerLawPlusPeakMassDistribution(**_MASS_PARAMS_FULL)
        m1, m2 = m.sample(200, rng=np.random.default_rng(0))
        self.assertTrue(np.all(m1 >= m2))
        self.assertTrue(np.all(m2 >= _MASS_PARAMS_FULL["m_min"] - 1e-10))
        self.assertTrue(np.all(m1 <= _MASS_PARAMS_FULL["m_max"] + 1e-10))

    def test_masses_are_positive(self) -> None:
        """All sampled masses are strictly positive."""
        m = _make_mass()
        m1, m2 = m.sample(200, rng=np.random.default_rng(5))
        self.assertTrue(np.all(m1 > 0))
        self.assertTrue(np.all(m2 > 0))


# ---------------------------------------------------------------------------
# IsotropicSpinDistribution
# ---------------------------------------------------------------------------


class TestIsotropicSpinDistribution(unittest.TestCase):
    """Tests for the isotropic spin distribution."""

    def test_default_a_max(self) -> None:
        """Default a_max is 1.0."""
        s = IsotropicSpinDistribution()
        self.assertEqual(s.a_max, 1.0)

    def test_custom_a_max(self) -> None:
        """Custom a_max is stored."""
        s = IsotropicSpinDistribution(a_max=0.8)
        self.assertEqual(s.a_max, 0.8)

    def test_invalid_a_max(self) -> None:
        """a_max <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            IsotropicSpinDistribution(a_max=0.0)

    def test_sample_shapes(self) -> None:
        """All six returned arrays have the requested shape."""
        s = IsotropicSpinDistribution()
        a1, a2, ct1, ct2, phi12, phi_jl = s.sample(50, rng=np.random.default_rng(0))
        for arr in (a1, a2, ct1, ct2, phi12, phi_jl):
            self.assertEqual(arr.shape, (50,))

    def test_spin_magnitudes_bounds(self) -> None:
        """Spin magnitudes lie in [0, a_max]."""
        a_max = 0.9
        s = IsotropicSpinDistribution(a_max=a_max)
        a1, a2, *_ = s.sample(500, rng=np.random.default_rng(0))
        self.assertTrue(np.all(a1 >= 0.0))
        self.assertTrue(np.all(a1 <= a_max))
        self.assertTrue(np.all(a2 >= 0.0))
        self.assertTrue(np.all(a2 <= a_max))

    def test_cos_tilt_bounds(self) -> None:
        """Cosine spin tilts lie in [-1, 1]."""
        s = IsotropicSpinDistribution()
        _, _, ct1, ct2, _, _ = s.sample(500, rng=np.random.default_rng(0))
        self.assertTrue(np.all(ct1 >= -1.0))
        self.assertTrue(np.all(ct1 <= 1.0))
        self.assertTrue(np.all(ct2 >= -1.0))
        self.assertTrue(np.all(ct2 <= 1.0))

    def test_phi_bounds(self) -> None:
        """Azimuthal angles lie in [0, 2π)."""
        s = IsotropicSpinDistribution()
        *_, phi12, phi_jl = s.sample(500, rng=np.random.default_rng(0))
        self.assertTrue(np.all(phi12 >= 0.0))
        self.assertTrue(np.all(phi12 < 2.0 * np.pi))
        self.assertTrue(np.all(phi_jl >= 0.0))
        self.assertTrue(np.all(phi_jl < 2.0 * np.pi))

    def test_reproducible(self) -> None:
        """Same seed produces identical output."""
        s = IsotropicSpinDistribution()
        out_a = s.sample(30, rng=np.random.default_rng(7))
        out_b = s.sample(30, rng=np.random.default_rng(7))
        for a, b in zip(out_a, out_b):
            np.testing.assert_array_equal(a, b)

    def test_default_rng(self) -> None:
        """sample() works when rng=None."""
        s = IsotropicSpinDistribution()
        result = s.sample(5, rng=None)
        self.assertEqual(len(result), 6)


# ---------------------------------------------------------------------------
# DefaultSpinDistribution
# ---------------------------------------------------------------------------


class TestDefaultSpinDistribution(unittest.TestCase):
    """Tests for the LVK-default spin distribution."""

    _PARAMS = dict(alpha_chi=2.0, beta_chi=4.0, xi_spin=0.5, sigma_spin=1.5)

    def test_stores_parameters(self) -> None:
        """All hyperparameters are stored."""
        s = DefaultSpinDistribution(**self._PARAMS)
        self.assertEqual(s.alpha_chi, 2.0)
        self.assertEqual(s.beta_chi, 4.0)
        self.assertEqual(s.xi_spin, 0.5)
        self.assertEqual(s.sigma_spin, 1.5)

    def test_invalid_alpha_chi(self) -> None:
        """alpha_chi <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            DefaultSpinDistribution(**{**self._PARAMS, "alpha_chi": 0.0})

    def test_invalid_beta_chi(self) -> None:
        """beta_chi <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            DefaultSpinDistribution(**{**self._PARAMS, "beta_chi": -1.0})

    def test_invalid_xi_spin_range(self) -> None:
        """xi_spin outside [0, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            DefaultSpinDistribution(**{**self._PARAMS, "xi_spin": 1.5})

    def test_sample_shapes(self) -> None:
        """All six returned arrays have the requested shape."""
        s = DefaultSpinDistribution(**self._PARAMS)
        a1, a2, ct1, ct2, phi12, phi_jl = s.sample(80, rng=np.random.default_rng(0))
        for arr in (a1, a2, ct1, ct2, phi12, phi_jl):
            self.assertEqual(arr.shape, (80,))

    def test_spin_magnitudes_bounds(self) -> None:
        """Spin magnitudes lie in [0, 1]."""
        s = DefaultSpinDistribution(**self._PARAMS)
        a1, a2, *_ = s.sample(500, rng=np.random.default_rng(0))
        self.assertTrue(np.all(a1 >= 0.0))
        self.assertTrue(np.all(a1 <= 1.0))
        self.assertTrue(np.all(a2 >= 0.0))
        self.assertTrue(np.all(a2 <= 1.0))

    def test_cos_tilt_bounds(self) -> None:
        """Cosine spin tilts lie in [-1, 1]."""
        s = DefaultSpinDistribution(**self._PARAMS)
        _, _, ct1, ct2, _, _ = s.sample(500, rng=np.random.default_rng(0))
        self.assertTrue(np.all(ct1 >= -1.0))
        self.assertTrue(np.all(ct1 <= 1.0))

    def test_reproducible(self) -> None:
        """Same seed produces identical output."""
        s = DefaultSpinDistribution(**self._PARAMS)
        out_a = s.sample(40, rng=np.random.default_rng(13))
        out_b = s.sample(40, rng=np.random.default_rng(13))
        for a, b in zip(out_a, out_b):
            np.testing.assert_array_equal(a, b)
