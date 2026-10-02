"""Tests for bagpuss.glade_export."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18

from bagpuss.catalogue import GalaxyCatalogue, MagnitudeLimitedSurvey
from bagpuss.glade_export import (
    GLADE_COLUMNS,
    compute_absolute_magnitudes,
    write_completeness_curve,
    write_glade_catalogue,
    write_readme,
)
from bagpuss.luminosity import SchechterLuminosityModel

_LUMINOSITY_PARAMS = dict(
    phi_star=1.61e-2,
    m_star=-19.66,
    alpha=-1.16,
    m_min=-25.0,
    m_max=-14.0,
)


def _make_luminosity_model(**kwargs: float) -> SchechterLuminosityModel:
    params = dict(_LUMINOSITY_PARAMS)
    params.update(kwargs)
    return SchechterLuminosityModel(**params)


def _make_catalogue() -> GalaxyCatalogue:
    return GalaxyCatalogue(
        redshifts=np.array([0.1, 0.2, 0.3]),
        luminosities=np.array([1e10, 2e10, 5e9]),
        apparent_magnitudes=np.array([17.5, 18.2, 19.0]),
        ra=np.array([0.0, np.pi / 2, np.pi]),
        dec=np.array([0.0, np.pi / 4, -np.pi / 4]),
    )


class TestComputeAbsoluteMagnitudes(unittest.TestCase):
    """Tests for compute_absolute_magnitudes."""

    def test_matches_hand_computed_formula(self) -> None:
        """Matches the standard M = m_sun - 2.5 log10(L) formula."""
        luminosities = np.array([1e10, 2e10])
        m_sun = 4.83
        expected = m_sun - 2.5 * np.log10(luminosities)
        np.testing.assert_allclose(
            compute_absolute_magnitudes(luminosities, m_sun), expected
        )


class TestWriteGladeCatalogue(unittest.TestCase):
    """Tests for write_glade_catalogue."""

    def setUp(self) -> None:
        """Build a small catalogue and a scratch output path."""
        self.catalogue = _make_catalogue()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "catalogue.dat"

    def tearDown(self) -> None:
        """Clean up the scratch directory."""
        self.tmpdir.cleanup()

    def _read_rows(self) -> list[list[float]]:
        with open(self.path) as f:
            return [[float(x) for x in line.split()] for line in f if line.strip()]

    def test_row_count_matches_catalogue(self) -> None:
        """One row is written per galaxy in the catalogue."""
        write_glade_catalogue(self.catalogue, Planck18, self.path, m_sun=4.83)
        self.assertEqual(len(self._read_rows()), len(self.catalogue))

    def test_ids_are_zero_based_sequential(self) -> None:
        """IDs start at 0 and increment by 1, matching host_galaxy_index."""
        write_glade_catalogue(self.catalogue, Planck18, self.path, m_sun=4.83)
        ids = [int(row[0]) for row in self._read_rows()]
        self.assertEqual(ids, [0, 1, 2])

    def test_id_offset_applied(self) -> None:
        """id_offset shifts every row's ID by the same amount."""
        write_glade_catalogue(
            self.catalogue, Planck18, self.path, m_sun=4.83, id_offset=100
        )
        ids = [int(row[0]) for row in self._read_rows()]
        self.assertEqual(ids, [100, 101, 102])

    def test_ra_dec_converted_to_degrees(self) -> None:
        """RA/Dec are converted from bagpuss's internal radians to degrees."""
        write_glade_catalogue(self.catalogue, Planck18, self.path, m_sun=4.83)
        rows = self._read_rows()
        expected_ra_deg = np.degrees(self.catalogue.ra)
        expected_dec_deg = np.degrees(self.catalogue.dec)
        for row, ra, dec in zip(rows, expected_ra_deg, expected_dec_deg):
            self.assertAlmostEqual(row[1], ra, places=5)
            self.assertAlmostEqual(row[2], dec, places=5)

    def test_luminosity_distance_matches_cosmology(self) -> None:
        """The D_L column matches cosmology.luminosity_distance directly."""
        write_glade_catalogue(self.catalogue, Planck18, self.path, m_sun=4.83)
        rows = self._read_rows()
        expected = (
            Planck18.luminosity_distance(self.catalogue.redshifts).to("Mpc").value  # pyright: ignore[reportAttributeAccessIssue]
        )
        for row, d_l in zip(rows, expected):
            self.assertAlmostEqual(row[4], d_l, places=2)

    def test_absolute_magnitude_matches_formula(self) -> None:
        """The M_abs column matches compute_absolute_magnitudes directly."""
        write_glade_catalogue(self.catalogue, Planck18, self.path, m_sun=4.83)
        rows = self._read_rows()
        expected = compute_absolute_magnitudes(self.catalogue.luminosities, 4.83)
        for row, m_abs in zip(rows, expected):
            self.assertAlmostEqual(row[6], m_abs, places=3)

    def test_apparent_magnitude_passthrough(self) -> None:
        """The m_app column is the catalogue's apparent_magnitudes unchanged."""
        write_glade_catalogue(self.catalogue, Planck18, self.path, m_sun=4.83)
        rows = self._read_rows()
        for row, m_app in zip(rows, self.catalogue.apparent_magnitudes):
            self.assertAlmostEqual(row[5], m_app, places=3)

    def test_empty_catalogue_writes_empty_file(self) -> None:
        """An empty catalogue writes an empty (not missing/erroring) file."""
        empty = GalaxyCatalogue(
            redshifts=np.array([]),
            luminosities=np.array([]),
            apparent_magnitudes=np.array([]),
            ra=np.array([]),
            dec=np.array([]),
        )
        write_glade_catalogue(empty, Planck18, self.path, m_sun=4.83)
        self.assertEqual(self._read_rows(), [])


class TestWriteCompletenessCurve(unittest.TestCase):
    """Tests for write_completeness_curve."""

    def setUp(self) -> None:
        """Build a selection function, luminosity model, and scratch path."""
        self.selection = MagnitudeLimitedSurvey(m_lim=19.5)
        self.luminosity = _make_luminosity_model()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "completeness.dat"

    def tearDown(self) -> None:
        """Clean up the scratch directory."""
        self.tmpdir.cleanup()

    def test_matches_selection_completeness_directly(self) -> None:
        """Written values match MagnitudeLimitedSurvey.completeness() exactly."""
        write_completeness_curve(
            self.selection,
            self.luminosity,
            Planck18,
            z_max=0.5,
            path=self.path,
            n_points=50,
        )
        redshifts = []
        completeness = []
        with open(self.path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                z, c = line.split()
                redshifts.append(float(z))
                completeness.append(float(c))
        redshifts_arr = np.array(redshifts)
        expected = self.selection.completeness(
            redshifts=redshifts_arr,
            ra=np.zeros_like(redshifts_arr),
            dec=np.zeros_like(redshifts_arr),
            cosmology=Planck18,
            luminosity=self.luminosity,
        )
        # atol accounts for the file format's 6-decimal-place rounding.
        np.testing.assert_allclose(completeness, expected, atol=5e-6)

    def test_values_bounded_in_unit_interval(self) -> None:
        """Every written completeness value is a valid probability."""
        write_completeness_curve(
            self.selection,
            self.luminosity,
            Planck18,
            z_max=0.5,
            path=self.path,
            n_points=50,
        )
        with open(self.path) as f:
            values = [
                float(line.split()[1])
                for line in f
                if not line.startswith("#") and line.strip()
            ]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))


class TestWriteReadme(unittest.TestCase):
    """Tests for write_readme."""

    def setUp(self) -> None:
        """Set up a scratch output path."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "README.txt"

    def tearDown(self) -> None:
        """Clean up the scratch directory."""
        self.tmpdir.cleanup()

    def test_contains_disclaimer_and_provenance(self) -> None:
        """The README states it isn't real GLADE+ and lists run provenance."""
        write_readme(
            self.path,
            m_sun=4.83,
            cosmology_name="Planck18",
            bagpuss_version="0.1.0",
            n_galaxies=42,
            z_max=0.3,
        )
        text = self.path.read_text()
        self.assertIn("NOT THE REAL GLADE+", text)
        self.assertIn("Planck18", text)
        self.assertIn("0.1.0", text)
        self.assertIn("42", text)
        for name, _ in GLADE_COLUMNS:
            self.assertIn(name, text)

    def test_documents_host_galaxy_index_correspondence(self) -> None:
        """The README explains the ID <-> host_galaxy_index correspondence."""
        write_readme(
            self.path,
            m_sun=4.83,
            cosmology_name="Planck18",
            bagpuss_version="0.1.0",
            n_galaxies=42,
            z_max=0.3,
        )
        text = self.path.read_text()
        self.assertIn("host_galaxy_index", text)


if __name__ == "__main__":
    unittest.main()
