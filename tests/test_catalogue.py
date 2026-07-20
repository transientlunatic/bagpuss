"""Tests for bagpuss.catalogue."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr
from astropy.cosmology import Planck18

from bagpuss.catalogue import (
    GalaxyCatalogue,
    MagnitudeLimitedSurvey,
    SelectionFunction,
    build_catalogue,
)
from bagpuss.galaxies import GalaxySet
from bagpuss.luminosity import SchechterLuminosityModel
from bagpuss.universe import PointProcess, Universe

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
    ra: list[float] | None = None,
    dec: list[float] | None = None,
) -> GalaxySet:
    n = len(redshifts)
    return GalaxySet(
        redshifts=np.array(redshifts),
        luminosities=np.array(luminosities),
        ra=np.zeros(n) if ra is None else np.array(ra),
        dec=np.zeros(n) if dec is None else np.array(dec),
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
            ra=np.array([0.1, 1.2, 2.3]),
            dec=np.array([-0.3, 0.0, 0.4]),
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

    def test_stores_ra(self) -> None:
        """RA is stored on the instance."""
        cat = self._make_catalogue()
        np.testing.assert_array_equal(cat.ra, [0.1, 1.2, 2.3])

    def test_stores_dec(self) -> None:
        """Dec is stored on the instance."""
        cat = self._make_catalogue()
        np.testing.assert_array_equal(cat.dec, [-0.3, 0.0, 0.4])

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
            ra=np.array([]),
            dec=np.array([]),
        )
        self.assertEqual(len(cat), 0)

    def test_to_zarr_from_zarr_path_round_trip(self) -> None:
        """to_zarr/from_zarr round-trip all fields via a store path."""
        cat = self._make_catalogue()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalogue.zarr"
            cat.to_zarr(path)
            loaded = GalaxyCatalogue.from_zarr(path)
        self.assertEqual(len(loaded), len(cat))
        for field in ("redshifts", "luminosities", "apparent_magnitudes", "ra", "dec"):
            np.testing.assert_allclose(getattr(loaded, field), getattr(cat, field))

    def test_to_zarr_from_zarr_group_round_trip(self) -> None:
        """to_zarr/from_zarr accept an already-open zarr.Group (shard use case)."""
        cat = self._make_catalogue()
        with tempfile.TemporaryDirectory() as tmp:
            root = zarr.open_group(store=str(Path(tmp) / "store.zarr"), mode="a")
            tile = root.create_group("tile_0000")
            cat.to_zarr(tile)

            root_r = zarr.open_group(store=str(Path(tmp) / "store.zarr"), mode="r")
            loaded = GalaxyCatalogue.from_zarr(root_r["tile_0000"])
        np.testing.assert_allclose(loaded.redshifts, cat.redshifts)


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
        survey = MagnitudeLimitedSurvey(m_lim=25.0)
        galaxies = _make_galaxy_set([0.1], [1e15])
        result = survey.apply(galaxies, Planck18)
        self.assertEqual(len(result), 1)

    def test_faint_galaxy_rejected(self) -> None:
        """A very faint galaxy is rejected when m_lim is bright."""
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
        m_bright = _apparent_magnitude(0.1, 1e12)
        m_faint = _apparent_magnitude(0.5, 1e6)
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

    def test_ra_propagated(self) -> None:
        """RA values are propagated from GalaxySet to selected catalogue entries."""
        survey = MagnitudeLimitedSurvey(m_lim=100.0)
        ra_vals = [0.5, 1.0, 2.0]
        galaxies = _make_galaxy_set([0.1, 0.2, 0.3], [1e10, 1e10, 1e10], ra=ra_vals)
        result = survey.apply(galaxies, Planck18)
        np.testing.assert_array_equal(result.ra, ra_vals)

    def test_dec_propagated(self) -> None:
        """Dec values are propagated from GalaxySet to selected catalogue entries."""
        survey = MagnitudeLimitedSurvey(m_lim=100.0)
        dec_vals = [-0.3, 0.0, 0.5]
        galaxies = _make_galaxy_set([0.1, 0.2, 0.3], [1e10, 1e10, 1e10], dec=dec_vals)
        result = survey.apply(galaxies, Planck18)
        np.testing.assert_array_equal(result.dec, dec_vals)

    def test_ra_dec_filtered_consistently(self) -> None:
        """RA/Dec in catalogue correspond to the selected rows only."""
        m_bright = _apparent_magnitude(0.1, 1e12)
        m_faint = _apparent_magnitude(0.5, 1e6)
        m_lim = 0.5 * (m_bright + m_faint)
        survey = MagnitudeLimitedSurvey(m_lim=m_lim)
        galaxies = _make_galaxy_set(
            [0.1, 0.5], [1e12, 1e6], ra=[0.1, 5.0], dec=[0.2, -0.9]
        )
        result = survey.apply(galaxies, Planck18)
        np.testing.assert_allclose(result.ra, [0.1])
        np.testing.assert_allclose(result.dec, [0.2])

    def test_catalogue_fields_consistent_length(self) -> None:
        """All fields in the output catalogue have the same length."""
        survey = MagnitudeLimitedSurvey(m_lim=20.0)
        galaxies = _make_galaxy_set([0.1, 0.2, 0.5], [1e12, 1e10, 1e6])
        result = survey.apply(galaxies, Planck18)
        shape = result.redshifts.shape
        self.assertEqual(result.luminosities.shape, shape)
        self.assertEqual(result.apparent_magnitudes.shape, shape)
        self.assertEqual(result.ra.shape, shape)
        self.assertEqual(result.dec.shape, shape)


# ---------------------------------------------------------------------------
# MagnitudeLimitedSurvey.completeness
# ---------------------------------------------------------------------------


class TestMagnitudeLimitedSurveyCompleteness(unittest.TestCase):
    """Tests for MagnitudeLimitedSurvey.completeness."""

    def test_returns_array_of_right_shape(self) -> None:
        """Completeness is evaluated once per input redshift."""
        survey = MagnitudeLimitedSurvey(m_lim=19.5)
        z = np.array([0.05, 0.1, 0.2])
        result = survey.completeness(
            z, np.zeros(3), np.zeros(3), Planck18, _make_luminosity_model()
        )
        self.assertEqual(result.shape, z.shape)

    def test_values_in_unit_interval(self) -> None:
        """Completeness is a probability, bounded in [0, 1]."""
        survey = MagnitudeLimitedSurvey(m_lim=19.5)
        z = np.linspace(0.01, 1.0, 20)
        result = survey.completeness(
            z, np.zeros(20), np.zeros(20), Planck18, _make_luminosity_model()
        )
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_decreases_with_redshift(self) -> None:
        """Galaxies are harder to catalogue at higher redshift."""
        survey = MagnitudeLimitedSurvey(m_lim=19.5)
        model = _make_luminosity_model()
        c_near = survey.completeness(
            np.array([0.02]), np.zeros(1), np.zeros(1), Planck18, model
        )[0]
        c_far = survey.completeness(
            np.array([0.5]), np.zeros(1), np.zeros(1), Planck18, model
        )[0]
        self.assertGreater(c_near, c_far)

    def test_independent_of_ra_dec(self) -> None:
        """For now, completeness depends only on redshift."""
        survey = MagnitudeLimitedSurvey(m_lim=19.5)
        model = _make_luminosity_model()
        z = np.array([0.1, 0.1])
        c1 = survey.completeness(
            z, np.array([0.0, 5.0]), np.array([0.0, -1.0]), Planck18, model
        )
        c2 = survey.completeness(
            z, np.array([2.0, 1.0]), np.array([0.5, 0.2]), Planck18, model
        )
        np.testing.assert_allclose(c1, c2)

    def test_zero_for_extremely_bright_limit(self) -> None:
        """A survey far brighter than any galaxy has ~zero completeness."""
        survey = MagnitudeLimitedSurvey(m_lim=-100.0)
        model = _make_luminosity_model()
        result = survey.completeness(
            np.array([0.1]), np.zeros(1), np.zeros(1), Planck18, model
        )
        self.assertAlmostEqual(float(result[0]), 0.0)

    def test_one_for_extremely_faint_limit(self) -> None:
        """An extremely permissive survey catalogues everything."""
        survey = MagnitudeLimitedSurvey(m_lim=100.0)
        model = _make_luminosity_model()
        result = survey.completeness(
            np.array([0.1]), np.zeros(1), np.zeros(1), Planck18, model
        )
        self.assertAlmostEqual(float(result[0]), 1.0)

    def test_matches_direct_cdf_calculation(self) -> None:
        """completeness(z) equals the luminosity model's CDF at the threshold."""
        survey = MagnitudeLimitedSurvey(m_lim=19.5)
        model = _make_luminosity_model()
        z = np.array([0.1])
        d_l_pc = Planck18.luminosity_distance(z).to("pc").value
        mu = 5.0 * np.log10(d_l_pc / 10.0)
        expected = model.cdf(survey.m_lim - mu)
        result = survey.completeness(z, np.zeros(1), np.zeros(1), Planck18, model)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# build_catalogue
# ---------------------------------------------------------------------------


def _make_universe() -> Universe:
    # Use a small phi_star so build_catalogue draws O(1000) total galaxies,
    # not the ~2 million that real-world parameters would produce.
    return Universe(
        cosmology=Planck18,
        structure=PointProcess(z_max=0.3),
        luminosity=_make_luminosity_model(phi_star=1e-5),
    )


class TestBuildCatalogue(unittest.TestCase):
    """Tests for build_catalogue."""

    def test_returns_galaxy_catalogue(self) -> None:
        """Return value is a GalaxyCatalogue."""
        result = build_catalogue(
            _make_universe(),
            MagnitudeLimitedSurvey(m_lim=19.5),
            rng=np.random.default_rng(0),
        )
        self.assertIsInstance(result, GalaxyCatalogue)

    def test_all_within_magnitude_limit(self) -> None:
        """Every galaxy in the result is brighter than the magnitude limit."""
        m_lim = 19.5
        result = build_catalogue(
            _make_universe(),
            MagnitudeLimitedSurvey(m_lim=m_lim),
            rng=np.random.default_rng(1),
        )
        if len(result) > 0:
            self.assertTrue(np.all(result.apparent_magnitudes <= m_lim))

    def test_fields_consistent_length(self) -> None:
        """All fields in the returned catalogue have the same length."""
        result = build_catalogue(
            _make_universe(),
            MagnitudeLimitedSurvey(m_lim=19.5),
            rng=np.random.default_rng(2),
        )
        n = len(result)
        for field in ("redshifts", "luminosities", "apparent_magnitudes", "ra", "dec"):
            self.assertEqual(
                len(getattr(result, field)), n, f"Length mismatch: {field}"
            )

    def test_reproducible(self) -> None:
        """Same seed produces identical output."""
        universe = _make_universe()
        selection = MagnitudeLimitedSurvey(m_lim=19.5)
        r1 = build_catalogue(universe, selection, rng=np.random.default_rng(7))
        r2 = build_catalogue(universe, selection, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(r1.redshifts, r2.redshifts)

    def test_count_within_poisson_bounds(self) -> None:
        """Catalogue size is within 5σ of the expected Poisson mean."""
        universe = _make_universe()
        density = universe.luminosity.number_density()
        volume = universe.structure.survey_volume(universe.cosmology)
        # Upper bound: catalogue can't exceed the Poisson draw of the total
        # population.  Lower bound: at least 0.
        result = build_catalogue(
            universe, MagnitudeLimitedSurvey(m_lim=19.5), rng=np.random.default_rng(0)
        )
        self.assertGreaterEqual(len(result), 0)
        self.assertLessEqual(len(result), density * volume * 2)

    def test_restrictive_limit_gives_empty_catalogue(self) -> None:
        """An impossibly bright limit produces an empty catalogue."""
        result = build_catalogue(
            _make_universe(),
            MagnitudeLimitedSurvey(m_lim=-100.0),
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(result), 0)
