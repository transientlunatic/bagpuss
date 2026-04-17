"""Tests for bagpuss.universe."""

import unittest

import numpy as np
from astropy.cosmology import Planck18

from bagpuss.galaxies import GalaxySet
from bagpuss.universe import (
    LuminosityModel,
    PointProcess,
    Structure,
    Universe,
)

# ---------------------------------------------------------------------------
# Minimal concrete stubs for use in tests
# ---------------------------------------------------------------------------


class _FlatLuminosity(LuminosityModel):
    """Trivial luminosity model — all galaxies get 1e10 L_sun."""

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return a constant-luminosity array."""
        return np.ones(n) * 1e10


# ---------------------------------------------------------------------------
# Abstract base class instantiation guards
# ---------------------------------------------------------------------------


class TestAbstractBaseClasses(unittest.TestCase):
    """Structure and LuminosityModel cannot be instantiated directly."""

    def test_structure_is_abstract(self) -> None:
        """Structure raises TypeError on direct instantiation."""
        with self.assertRaises(TypeError):
            Structure()  # type: ignore[abstract]

    def test_luminosity_model_is_abstract(self) -> None:
        """LuminosityModel raises TypeError on direct instantiation."""
        with self.assertRaises(TypeError):
            LuminosityModel()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# PointProcess
# ---------------------------------------------------------------------------


class TestPointProcess(unittest.TestCase):
    """Tests for the PointProcess structure model."""

    def test_stores_z_max(self) -> None:
        """z_max is stored as an attribute."""
        pp = PointProcess(z_max=1.0)
        self.assertEqual(pp.z_max, 1.0)

    def test_invalid_z_max_zero(self) -> None:
        """z_max = 0 raises ValueError."""
        with self.assertRaises(ValueError):
            PointProcess(z_max=0.0)

    def test_invalid_z_max_negative(self) -> None:
        """Negative z_max raises ValueError."""
        with self.assertRaises(ValueError):
            PointProcess(z_max=-0.5)

    def test_sample_redshifts_shape(self) -> None:
        """sample_redshifts returns an array of the requested length."""
        pp = PointProcess(z_max=1.0)
        rng = np.random.default_rng(0)
        z = pp.sample_redshifts(200, Planck18, rng)
        self.assertEqual(z.shape, (200,))

    def test_sample_redshifts_bounds(self) -> None:
        """All sampled redshifts lie in [0, z_max]."""
        pp = PointProcess(z_max=0.5)
        rng = np.random.default_rng(42)
        z = pp.sample_redshifts(1000, Planck18, rng)
        self.assertGreaterEqual(float(z.min()), 0.0)
        self.assertLessEqual(float(z.max()), 0.5)

    def test_sample_redshifts_reproducible(self) -> None:
        """Same RNG seed produces identical output."""
        pp = PointProcess(z_max=1.0)
        z1 = pp.sample_redshifts(50, Planck18, np.random.default_rng(7))
        z2 = pp.sample_redshifts(50, Planck18, np.random.default_rng(7))
        np.testing.assert_array_equal(z1, z2)

    def test_sample_redshifts_default_rng(self) -> None:
        """sample_redshifts works when rng=None (uses internal default)."""
        pp = PointProcess(z_max=1.0)
        z = pp.sample_redshifts(10, Planck18, rng=None)
        self.assertEqual(z.shape, (10,))


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


class TestUniverse(unittest.TestCase):
    """Tests for the Universe sampling wrapper."""

    def _make_universe(self, z_max: float = 1.0) -> Universe:
        return Universe(
            cosmology=Planck18,
            structure=PointProcess(z_max=z_max),
            luminosity=_FlatLuminosity(),
        )

    def test_sample_returns_galaxy_set(self) -> None:
        """Universe.sample returns a GalaxySet instance."""
        u = self._make_universe()
        result = u.sample(10, rng=np.random.default_rng(0))
        self.assertIsInstance(result, GalaxySet)

    def test_sample_length(self) -> None:
        """The returned GalaxySet has exactly n galaxies."""
        u = self._make_universe()
        result = u.sample(123, rng=np.random.default_rng(0))
        self.assertEqual(len(result), 123)

    def test_sample_redshift_shape(self) -> None:
        """The redshifts array has shape (n,)."""
        u = self._make_universe()
        result = u.sample(50, rng=np.random.default_rng(0))
        self.assertEqual(result.redshifts.shape, (50,))

    def test_sample_luminosity_shape(self) -> None:
        """The luminosities array has shape (n,)."""
        u = self._make_universe()
        result = u.sample(50, rng=np.random.default_rng(0))
        self.assertEqual(result.luminosities.shape, (50,))

    def test_sample_luminosities_from_model(self) -> None:
        """Luminosities reflect the luminosity model output."""
        u = self._make_universe()
        result = u.sample(20, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result.luminosities, np.ones(20) * 1e10)

    def test_attributes_stored(self) -> None:
        """Cosmology, structure, and luminosity are stored on the instance."""
        pp = PointProcess(z_max=2.0)
        lum = _FlatLuminosity()
        u = Universe(cosmology=Planck18, structure=pp, luminosity=lum)
        self.assertIs(u.cosmology, Planck18)
        self.assertIs(u.structure, pp)
        self.assertIs(u.luminosity, lum)
