"""Tests for bagpuss.galaxies."""

import unittest

import numpy as np

from bagpuss.galaxies import GalaxySet


class TestGalaxySet(unittest.TestCase):
    """Tests for the GalaxySet data container."""

    def _make(self, n: int = 5) -> GalaxySet:
        return GalaxySet(
            redshifts=np.linspace(0.1, 0.5, n),
            luminosities=np.ones(n) * 1e10,
            ra=np.linspace(0.0, 1.0, n),
            dec=np.linspace(-0.5, 0.5, n),
        )

    def test_len(self) -> None:
        """len() returns the number of galaxies."""
        gs = self._make(7)
        self.assertEqual(len(gs), 7)

    def test_len_single(self) -> None:
        """len() works for a single-galaxy set."""
        gs = self._make(1)
        self.assertEqual(len(gs), 1)

    def test_redshifts_stored(self) -> None:
        """Redshifts attribute is preserved exactly."""
        z = np.array([0.1, 0.2, 0.3])
        gs = GalaxySet(
            redshifts=z,
            luminosities=np.ones(3),
            ra=np.zeros(3),
            dec=np.zeros(3),
        )
        np.testing.assert_array_equal(gs.redshifts, z)

    def test_luminosities_stored(self) -> None:
        """Luminosities attribute is preserved exactly."""
        lum = np.array([1e10, 2e10, 3e10])
        gs = GalaxySet(
            redshifts=np.zeros(3),
            luminosities=lum,
            ra=np.zeros(3),
            dec=np.zeros(3),
        )
        np.testing.assert_array_equal(gs.luminosities, lum)

    def test_ra_stored(self) -> None:
        """RA attribute is preserved exactly."""
        ra = np.array([0.1, 1.5, 3.0])
        gs = GalaxySet(
            redshifts=np.zeros(3),
            luminosities=np.ones(3),
            ra=ra,
            dec=np.zeros(3),
        )
        np.testing.assert_array_equal(gs.ra, ra)

    def test_dec_stored(self) -> None:
        """Dec attribute is preserved exactly."""
        dec = np.array([-0.5, 0.0, 0.5])
        gs = GalaxySet(
            redshifts=np.zeros(3),
            luminosities=np.ones(3),
            ra=np.zeros(3),
            dec=dec,
        )
        np.testing.assert_array_equal(gs.dec, dec)
