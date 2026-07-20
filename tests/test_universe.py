"""Tests for bagpuss.universe."""

import unittest

import numpy as np
from astropy.cosmology import Planck18

from bagpuss.galaxies import GalaxySet
from bagpuss.universe import (
    LuminosityModel,
    PointProcess,
    SkyPatch,
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

    def cdf(self, magnitude: np.ndarray) -> np.ndarray:
        """Return 1.0 everywhere; every galaxy has the same luminosity."""
        return np.ones_like(np.asarray(magnitude, dtype=float))

    def number_density(self) -> float:
        """Return a fixed placeholder density of 1.0 Mpc⁻³."""
        return 1.0


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

    def test_sample_positions_shape(self) -> None:
        """sample_positions returns two arrays of the requested length."""
        pp = PointProcess(z_max=1.0)
        rng = np.random.default_rng(0)
        ra, dec = pp.sample_positions(200, rng)
        self.assertEqual(ra.shape, (200,))
        self.assertEqual(dec.shape, (200,))

    def test_sample_positions_ra_bounds(self) -> None:
        """All RA values lie in [0, 2π)."""
        pp = PointProcess(z_max=1.0)
        ra, _ = pp.sample_positions(2000, np.random.default_rng(1))
        self.assertGreaterEqual(float(ra.min()), 0.0)
        self.assertLess(float(ra.max()), 2.0 * np.pi)

    def test_sample_positions_dec_bounds(self) -> None:
        """All Dec values lie in [-π/2, π/2]."""
        pp = PointProcess(z_max=1.0)
        _, dec = pp.sample_positions(2000, np.random.default_rng(2))
        self.assertGreaterEqual(float(dec.min()), -np.pi / 2.0)
        self.assertLessEqual(float(dec.max()), np.pi / 2.0)

    def test_sample_positions_reproducible(self) -> None:
        """Same RNG seed produces identical positions."""
        pp = PointProcess(z_max=1.0)
        ra1, dec1 = pp.sample_positions(50, np.random.default_rng(9))
        ra2, dec2 = pp.sample_positions(50, np.random.default_rng(9))
        np.testing.assert_array_equal(ra1, ra2)
        np.testing.assert_array_equal(dec1, dec2)

    def test_sample_positions_default_rng(self) -> None:
        """sample_positions works when rng=None."""
        pp = PointProcess(z_max=1.0)
        ra, dec = pp.sample_positions(10, rng=None)
        self.assertEqual(ra.shape, (10,))
        self.assertEqual(dec.shape, (10,))


# ---------------------------------------------------------------------------
# SkyPatch
# ---------------------------------------------------------------------------


class TestSkyPatch(unittest.TestCase):
    """Tests for the SkyPatch structure decorator."""

    def test_full_sky_fraction_is_one(self) -> None:
        """A tile covering the whole sky has sky_fraction == 1."""
        tile = SkyPatch(
            PointProcess(z_max=1.0),
            ra_range=(0.0, 2.0 * np.pi),
            sin_dec_range=(-1.0, 1.0),
        )
        self.assertAlmostEqual(tile.sky_fraction, 1.0)

    def test_quarter_sky_fraction(self) -> None:
        """A tile covering a quarter of ra and half of sin(dec) is 1/8 sky."""
        tile = SkyPatch(
            PointProcess(z_max=1.0),
            ra_range=(0.0, np.pi / 2.0),
            sin_dec_range=(0.0, 1.0),
        )
        self.assertAlmostEqual(tile.sky_fraction, 0.125)

    def test_survey_volume_scaled_by_sky_fraction(self) -> None:
        """survey_volume is the base model's volume times sky_fraction."""
        base = PointProcess(z_max=3.0)
        tile = SkyPatch(base, ra_range=(0.0, np.pi), sin_dec_range=(0.0, 1.0))
        self.assertAlmostEqual(
            tile.survey_volume(Planck18), base.survey_volume(Planck18) * 0.25
        )

    def test_sample_redshifts_delegates_to_base(self) -> None:
        """sample_redshifts matches the base model given the same RNG state."""
        base = PointProcess(z_max=2.0)
        tile = SkyPatch(base, ra_range=(0.0, np.pi), sin_dec_range=(0.0, 1.0))
        z_base = base.sample_redshifts(100, Planck18, np.random.default_rng(3))
        z_tile = tile.sample_redshifts(100, Planck18, np.random.default_rng(3))
        np.testing.assert_array_equal(z_base, z_tile)

    def test_sample_positions_within_tile(self) -> None:
        """Sampled ra/dec always lie within the declared tile bounds."""
        tile = SkyPatch(
            PointProcess(z_max=1.0),
            ra_range=(1.0, 2.0),
            sin_dec_range=(-0.5, 0.5),
        )
        ra, dec = tile.sample_positions(2000, np.random.default_rng(11))
        self.assertGreaterEqual(float(ra.min()), 1.0)
        self.assertLess(float(ra.max()), 2.0)
        self.assertGreaterEqual(float(np.sin(dec).min()), -0.5)
        self.assertLess(float(np.sin(dec).max()), 0.5)

    def test_sample_positions_shape(self) -> None:
        """sample_positions returns arrays of the requested length."""
        tile = SkyPatch(
            PointProcess(z_max=1.0), ra_range=(0.0, 1.0), sin_dec_range=(-1.0, 0.0)
        )
        ra, dec = tile.sample_positions(50, np.random.default_rng(0))
        self.assertEqual(ra.shape, (50,))
        self.assertEqual(dec.shape, (50,))

    def test_sample_positions_default_rng(self) -> None:
        """sample_positions works when rng=None."""
        tile = SkyPatch(
            PointProcess(z_max=1.0), ra_range=(0.0, 1.0), sin_dec_range=(-1.0, 0.0)
        )
        ra, dec = tile.sample_positions(10, rng=None)
        self.assertEqual(ra.shape, (10,))
        self.assertEqual(dec.shape, (10,))

    def test_invalid_ra_range_out_of_bounds(self) -> None:
        """ra_range outside [0, 2*pi] raises ValueError."""
        with self.assertRaises(ValueError):
            SkyPatch(
                PointProcess(z_max=1.0), ra_range=(-0.1, 1.0), sin_dec_range=(-1.0, 1.0)
            )

    def test_invalid_ra_range_not_increasing(self) -> None:
        """ra_range with min >= max raises ValueError."""
        with self.assertRaises(ValueError):
            SkyPatch(
                PointProcess(z_max=1.0), ra_range=(2.0, 1.0), sin_dec_range=(-1.0, 1.0)
            )

    def test_invalid_sin_dec_range_out_of_bounds(self) -> None:
        """sin_dec_range outside [-1, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            SkyPatch(
                PointProcess(z_max=1.0), ra_range=(0.0, 1.0), sin_dec_range=(-1.5, 1.0)
            )

    def test_invalid_sin_dec_range_not_increasing(self) -> None:
        """sin_dec_range with min >= max raises ValueError."""
        with self.assertRaises(ValueError):
            SkyPatch(
                PointProcess(z_max=1.0), ra_range=(0.0, 1.0), sin_dec_range=(0.5, 0.5)
            )

    def test_tiles_partition_expected_volume(self) -> None:
        """Four equal tiles' volumes sum to the base model's full volume."""
        base = PointProcess(z_max=1.5)
        tiles = [
            SkyPatch(base, ra_range=(0.0, np.pi), sin_dec_range=(0.0, 1.0)),
            SkyPatch(base, ra_range=(np.pi, 2.0 * np.pi), sin_dec_range=(0.0, 1.0)),
            SkyPatch(base, ra_range=(0.0, np.pi), sin_dec_range=(-1.0, 0.0)),
            SkyPatch(base, ra_range=(np.pi, 2.0 * np.pi), sin_dec_range=(-1.0, 0.0)),
        ]
        total = sum(tile.survey_volume(Planck18) for tile in tiles)
        self.assertAlmostEqual(total, base.survey_volume(Planck18))


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

    def test_sample_ra_shape(self) -> None:
        """The ra array has shape (n,)."""
        u = self._make_universe()
        result = u.sample(50, rng=np.random.default_rng(0))
        self.assertEqual(result.ra.shape, (50,))

    def test_sample_dec_shape(self) -> None:
        """The dec array has shape (n,)."""
        u = self._make_universe()
        result = u.sample(50, rng=np.random.default_rng(0))
        self.assertEqual(result.dec.shape, (50,))

    def test_sample_ra_bounds(self) -> None:
        """All RA values lie in [0, 2π)."""
        u = self._make_universe()
        result = u.sample(500, rng=np.random.default_rng(0))
        self.assertGreaterEqual(float(result.ra.min()), 0.0)
        self.assertLess(float(result.ra.max()), 2.0 * np.pi)

    def test_sample_dec_bounds(self) -> None:
        """All Dec values lie in [-π/2, π/2]."""
        u = self._make_universe()
        result = u.sample(500, rng=np.random.default_rng(0))
        self.assertGreaterEqual(float(result.dec.min()), -np.pi / 2.0)
        self.assertLessEqual(float(result.dec.max()), np.pi / 2.0)

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
