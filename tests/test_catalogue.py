"""Tests for bagpuss.catalogue."""

import unittest

import numpy as np
from astropy.cosmology import Planck18

from bagpuss.catalogue import (
    GalaxyCatalogue,
    MagnitudeLimitedSurvey,
    SelectionFunction,
)
from bagpuss.galaxies import GalaxySet


def _apparent_magnitude(
    z: float,
    luminosity: float,
    m_sun: float = 4.83,
) -> float:
    """Compute the expected apparent magnitude for a galaxy."""
    d_L_pc = float(Planck18.luminosity_distance(z).to("pc").value)
    mu = 5.0 * np.log10(d_L_pc / 10.0)
    M_abs = m_sun - 2.5 * np.log10(luminosity)
    return float(M_abs + mu)


def _make_galaxy_set(
    redshifts: list[float],
    luminosities: list[float],
) -> GalaxySet:
    return GalaxySet(
        redshifts=np.array(redshifts),
        luminosities=np.array(luminosities),
    )


# ---------------------------------------------------------------------------
# GalaxyCatalogue
# ---------------------------------------------------------------------------


class TestGalaxyCatalogue(unittest.TestCase):
    """Tests for the GalaxyCatalogue dataclass."""

    def _make_catalogue(self) -> GalaxyCatalogue:
        return GalaxyCatalogue(
            redshifts=np.array([0.1, 0.2, 0.3]),
            luminosities=np.array([1e10, 2e10, 5e9]),
            apparent_magnitudes=np.array([18.0, 19.5, 20.1]),
        )

    def test_stores_redshifts(self) -> None:
        """Redshifts is stored on the instance."""
        cat = self._make_catalogue()
        np.testing.assert_array_equal(cat.redshifts, [0.1, 0.2, 0.3])

    def test_stores_luminosities(self) -> None:
        """Luminosities is stored on the instance."""
        cat = self._make_catalogue()
        np.testing.assert_array_equal(cat.luminosities, [1e10, 2e10, 5e9])

    def test_stores_apparent_magnitudes(self) -> None:
        """apparent_magnitudes is stored on the instance."""
        cat = self._make_catalogue()
        np.testing.assert_array_equal(cat.apparent_magnitudes, [18.0, 19.5, 20.1])

    def test_len(self) -> None:
        """__len__ returns the number of galaxies."""
        cat = self._make_catalogue()
        self.assertEqual(len(cat), 3)

    def test_len_empty(self) -> None:
        """__len__ returns 0 for an empty catalogue."""
        cat = GalaxyCatalogue(
            redshifts=np.array([]),
            luminosities=np.array([]),
            apparent_magnitudes=np.array([]),
        )
        self.assertEqual(len(cat), 0)


# ---------------------------------------------------------------------------
# SelectionFunction
# ---------------------------------------------------------------------------


class TestSelectionFunction(unittest.TestCase):
    """SelectionFunction cannot be instantiated directly."""

    def test_is_abstract(self) -> None:
        """Direct instantiation raises TypeError."""
        with self.assertRaises(TypeError):
            SelectionFunction()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MagnitudeLimitedSurvey
# ---------------------------------------------------------------------------


class TestMagnitudeLimitedSurvey(unittest.TestCase):
    """Tests for MagnitudeLimitedSurvey."""

    def test_stores_m_lim(self) -> None:
        """m_lim is stored on the instance."""
        survey = MagnitudeLimitedSurvey(m_lim=20.0)
        self.assertEqual(survey.m_lim, 20.0)

    def test_stores_m_sun(self) -> None:
        """m_sun is stored on the instance when provided."""
        survey = MagnitudeLimitedSurvey(m_lim=20.0, m_sun=5.0)
        self.assertEqual(survey.m_sun, 5.0)

    def test_default_m_sun(self) -> None:
        """Default m_sun is 4.83 (bolometric)."""
        survey = MagnitudeLimitedSurvey(m_lim=20.0)
        self.assertAlmostEqual(survey.m_sun, 4.83)

    def test_apply_returns_catalogue(self) -> None:
        """Apply returns a GalaxyCatalogue instance."""
        survey = MagnitudeLimitedSurvey(m_lim=25.0)
        galaxies = _make_galaxy_set([0.1], [1e10])
        result = survey.apply(galaxies, Planck18)
        self.assertIsInstance(result, GalaxyCatalogue)

    def test_bright_galaxy_selected(self) -> None:
        """A very luminous galaxy is selected for a typical m_lim."""
        # An extremely luminous galaxy (1e15 L_sun) at z=0.1 will have a
        # very small apparent magnitude and be selected for any sane m_lim.
        survey = MagnitudeLimitedSurvey(m_lim=25.0)
        galaxies = _make_galaxy_set([0.1], [1e15])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 1)

    def test_faint_galaxy_rejected(self) -> None:
        """A very faint galaxy is rejected when m_lim is bright."""
        # An extremely faint galaxy (1e-5 L_sun) at z=0.5 will be very faint.
        survey = MagnitudeLimitedSurvey(m_lim=10.0)
        galaxies = _make_galaxy_set([0.5], [1e-5])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 0)

    def test_galaxy_at_limit_selected(self) -> None:
        """A galaxy with m exactly at m_lim is selected (m <= m_lim)."""
        z, lum = 0.1, 1e10
        m_exact = _apparent_magnitude(z, lum)
        survey = MagnitudeLimitedSurvey(m_lim=m_exact)
        galaxies = _make_galaxy_set([z], [lum])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 1)

    def test_mixed_selection(self) -> None:
        """Only galaxies brighter than m_lim are returned."""
        # z=0.1, L=1e12 → bright; z=0.5, L=1e6 → faint
        m_bright = _apparent_magnitude(0.1, 1e12)
        m_faint = _apparent_magnitude(0.5, 1e6)
        # Set m_lim between the two
        m_lim = 0.5 * (m_bright + m_faint)
        survey = MagnitudeLimitedSurvey(m_lim=m_lim)
        galaxies = _make_galaxy_set([0.1, 0.5], [1e12, 1e6])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result.redshifts, [0.1])

    def test_empty_input(self) -> None:
        """An empty GalaxySet produces an empty GalaxyCatalogue."""
        survey = MagnitudeLimitedSurvey(m_lim=20.0)
        galaxies = _make_galaxy_set([], [])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 0)

    def test_all_selected(self) -> None:
        """An extremely permissive m_lim selects all galaxies."""
        survey = MagnitudeLimitedSurvey(m_lim=100.0)
        galaxies = _make_galaxy_set([0.1, 0.2, 0.3], [1e10, 1e10, 1e10])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 3)

    def test_none_selected(self) -> None:
        """An extremely restrictive m_lim selects no galaxies."""
        survey = MagnitudeLimitedSurvey(m_lim=-100.0)
        galaxies = _make_galaxy_set([0.1, 0.2], [1e10, 1e10])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 0)

    def test_apparent_magnitudes_correct(self) -> None:
        """The apparent magnitudes stored in the catalogue are correct."""
        z, lum = 0.1, 1e10
        expected_m = _apparent_magnitude(z, lum)
        survey = MagnitudeLimitedSurvey(m_lim=100.0)
        galaxies = _make_galaxy_set([z], [lum])
        result = survey.apply(galaxies, Planck18)
        np.testing.assert_allclose(result.apparent_magnitudes, [expected_m], rtol=1e-6)

    def test_output_redshifts_match_input(self) -> None:
        """Redshifts in the catalogue match the corresponding input values."""
        survey = MagnitudeLimitedSurvey(m_lim=100.0)
        zs = [0.05, 0.1, 0.2]
        lums = [1e10, 1e10, 1e10]
        galaxies = _make_galaxy_set(zs, lums)
        result = survey.apply(galaxies, Planck18)
        np.testing.assert_array_equal(result.redshifts, zs)

    def test_output_luminosities_match_input(self) -> None:
        """Luminosities in the catalogue match the corresponding input values."""
        survey = MagnitudeLimitedSurvey(m_lim=100.0)
        zs = [0.05, 0.1, 0.2]
        lums = [1e9, 2e10, 5e11]
        galaxies = _make_galaxy_set(zs, lums)
        result = survey.apply(galaxies, Planck18)
        np.testing.assert_array_equal(result.luminosities, lums)

    def test_catalogue_fields_consistent_length(self) -> None:
        """All fields in the output catalogue have the same length."""
        survey = MagnitudeLimitedSurvey(m_lim=20.0)
        galaxies = _make_galaxy_set([0.1, 0.2, 0.5], [1e12, 1e10, 1e6])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(
            result.redshifts.shape,
            result.luminosities.shape,
        )
        self.assertEqual(
            result.redshifts.shape,
            result.apparent_magnitudes.shape,
        )
