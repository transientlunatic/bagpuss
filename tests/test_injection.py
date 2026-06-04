"""Tests for bagpuss.injection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18

from bagpuss.catalogue import GalaxyCatalogue
from bagpuss.injection import (
    Detectable,
    DistanceThreshold,
    InjectionSet,
    create_injection_set,
)
from bagpuss.population import (
    IsotropicSpinDistribution,
    PopulationModel,
    PowerLawPlusPeakMassDistribution,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N_GALAXIES = 20
_N_EVENTS = 30

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


def _make_catalogue(n: int = _N_GALAXIES) -> GalaxyCatalogue:
    rng = np.random.default_rng(0)
    return GalaxyCatalogue(
        redshifts=rng.uniform(0.01, 0.3, n),
        luminosities=np.ones(n) * 1e10,
        apparent_magnitudes=np.ones(n) * 18.0,
        ra=rng.uniform(0.0, 2 * np.pi, n),
        dec=np.arcsin(rng.uniform(-1.0, 1.0, n)),
    )


def _make_population() -> PopulationModel:
    return PopulationModel(
        mass=PowerLawPlusPeakMassDistribution(**_MASS_PARAMS),
        spin=IsotropicSpinDistribution(),
    )


def _make_injection_set(n: int = _N_EVENTS) -> InjectionSet:
    return create_injection_set(
        catalogue=_make_catalogue(),
        population=_make_population(),
        cosmology=Planck18,
        n_draw=n,
        rng=np.random.default_rng(42),
    )


# ---------------------------------------------------------------------------
# InjectionSet
# ---------------------------------------------------------------------------


class TestInjectionSet(unittest.TestCase):
    """Tests for the InjectionSet dataclass."""

    _FIELDS = [
        "m1_source",
        "m2_source",
        "a1",
        "a2",
        "cos_tilt1",
        "cos_tilt2",
        "phi12",
        "phi_jl",
        "theta_jn",
        "ra",
        "dec",
        "psi",
        "geocent_time",
        "redshift",
        "luminosity_distance",
        "host_galaxy_index",
    ]

    def _make(self, n: int = 5) -> InjectionSet:
        rng = np.random.default_rng(0)
        return InjectionSet(
            m1_source=np.ones(n) * 30.0,
            m2_source=np.ones(n) * 20.0,
            a1=np.ones(n) * 0.5,
            a2=np.ones(n) * 0.3,
            cos_tilt1=np.ones(n) * 0.8,
            cos_tilt2=np.ones(n) * -0.2,
            phi12=np.ones(n) * 1.0,
            phi_jl=np.ones(n) * 2.0,
            theta_jn=np.ones(n) * 0.5,
            ra=rng.uniform(0, 2 * np.pi, n),
            dec=rng.uniform(-np.pi / 2, np.pi / 2, n),
            psi=rng.uniform(0, np.pi, n),
            geocent_time=np.ones(n) * 1.2e9,
            redshift=np.ones(n) * 0.1,
            luminosity_distance=np.ones(n) * 500.0,
            host_galaxy_index=np.arange(n),
        )

    def test_len(self) -> None:
        """__len__ returns the number of injections."""
        inj = self._make(7)
        self.assertEqual(len(inj), 7)

    def test_all_fields_stored(self) -> None:
        """All expected fields are present."""
        inj = self._make(3)
        for field in self._fields:
            self.assertTrue(hasattr(inj, field), f"Missing field: {field}")

    def test_to_hdf5_round_trip(self) -> None:
        """to_hdf5 and from_hdf5 round-trip all fields."""
        inj = self._make(10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "injections.h5"
            inj.to_hdf5(path)
            loaded = InjectionSet.from_hdf5(path)
        self.assertEqual(len(loaded), len(inj))
        for field in self._fields:
            np.testing.assert_allclose(
                getattr(loaded, field),
                getattr(inj, field),
                err_msg=f"Round-trip mismatch for field: {field}",
            )

    @property
    def _fields(self) -> list[str]:
        return self._FIELDS


# ---------------------------------------------------------------------------
# Detectable
# ---------------------------------------------------------------------------


class TestDetectable(unittest.TestCase):
    """Detectable cannot be instantiated directly."""

    def test_is_abstract(self) -> None:
        """Direct instantiation raises TypeError."""
        with self.assertRaises(TypeError):
            Detectable()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DistanceThreshold
# ---------------------------------------------------------------------------


class TestDistanceThreshold(unittest.TestCase):
    """Tests for the distance-threshold detectability model."""

    def test_stores_d_max(self) -> None:
        """d_max is stored on the instance."""
        dt = DistanceThreshold(d_max=1000.0)
        self.assertEqual(dt.d_max, 1000.0)

    def test_invalid_d_max(self) -> None:
        """d_max <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            DistanceThreshold(d_max=0.0)

    def test_near_event_detected(self) -> None:
        """Events closer than d_max have p_det = 1."""
        dt = DistanceThreshold(d_max=500.0)
        params = {"luminosity_distance": np.array([100.0, 200.0, 499.0])}
        p = dt(params)
        np.testing.assert_array_equal(p, [1.0, 1.0, 1.0])

    def test_far_event_not_detected(self) -> None:
        """Events farther than d_max have p_det = 0."""
        dt = DistanceThreshold(d_max=500.0)
        params = {"luminosity_distance": np.array([501.0, 1000.0])}
        p = dt(params)
        np.testing.assert_array_equal(p, [0.0, 0.0])

    def test_event_at_limit_detected(self) -> None:
        """An event exactly at d_max is detected (d <= d_max)."""
        dt = DistanceThreshold(d_max=500.0)
        params = {"luminosity_distance": np.array([500.0])}
        p = dt(params)
        np.testing.assert_array_equal(p, [1.0])

    def test_returns_float_array(self) -> None:
        """Output is a float array."""
        dt = DistanceThreshold(d_max=500.0)
        params = {"luminosity_distance": np.array([100.0, 600.0])}
        p = dt(params)
        self.assertEqual(p.dtype.kind, "f")


# ---------------------------------------------------------------------------
# create_injection_set
# ---------------------------------------------------------------------------


class TestCreateInjectionSet(unittest.TestCase):
    """Tests for the create_injection_set function."""

    def test_returns_injection_set(self) -> None:
        """create_injection_set returns an InjectionSet."""
        result = _make_injection_set(10)
        self.assertIsInstance(result, InjectionSet)

    def test_length_without_detectable(self) -> None:
        """Without a detectability filter all n_draw events are returned."""
        result = _make_injection_set(15)
        self.assertEqual(len(result), 15)

    def test_length_with_detectable_all_pass(self) -> None:
        """A very permissive threshold keeps all events."""
        result = create_injection_set(
            catalogue=_make_catalogue(),
            population=_make_population(),
            cosmology=Planck18,
            n_draw=20,
            detectable=DistanceThreshold(d_max=1e6),
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(result), 20)

    def test_length_with_detectable_none_pass(self) -> None:
        """A zero-distance threshold keeps no events."""
        result = create_injection_set(
            catalogue=_make_catalogue(),
            population=_make_population(),
            cosmology=Planck18,
            n_draw=20,
            detectable=DistanceThreshold(d_max=1e-6),
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(result), 0)

    def test_host_galaxy_indices_valid(self) -> None:
        """Host galaxy indices are valid indices into the catalogue."""
        cat = _make_catalogue(_N_GALAXIES)
        result = create_injection_set(
            catalogue=cat,
            population=_make_population(),
            cosmology=Planck18,
            n_draw=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.host_galaxy_index >= 0))
        self.assertTrue(np.all(result.host_galaxy_index < _N_GALAXIES))

    def test_ra_dec_match_host(self) -> None:
        """RA/Dec of injections match the assigned host galaxies."""
        cat = _make_catalogue(_N_GALAXIES)
        result = create_injection_set(
            catalogue=cat,
            population=_make_population(),
            cosmology=Planck18,
            n_draw=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        idx = result.host_galaxy_index
        np.testing.assert_array_equal(result.ra, cat.ra[idx])
        np.testing.assert_array_equal(result.dec, cat.dec[idx])

    def test_redshift_matches_host(self) -> None:
        """Redshifts of injections match the assigned host galaxies."""
        cat = _make_catalogue(_N_GALAXIES)
        result = create_injection_set(
            catalogue=cat,
            population=_make_population(),
            cosmology=Planck18,
            n_draw=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        idx = result.host_galaxy_index
        np.testing.assert_array_equal(result.redshift, cat.redshifts[idx])

    def test_luminosity_distance_positive(self) -> None:
        """All luminosity distances are positive."""
        result = _make_injection_set(_N_EVENTS)
        self.assertTrue(np.all(result.luminosity_distance > 0))

    def test_theta_jn_bounds(self) -> None:
        """Inclination angle lies in [0, π]."""
        result = _make_injection_set(_N_EVENTS)
        self.assertTrue(np.all(result.theta_jn >= 0.0))
        self.assertTrue(np.all(result.theta_jn <= np.pi))

    def test_psi_bounds(self) -> None:
        """Polarisation angle lies in [0, π)."""
        result = _make_injection_set(_N_EVENTS)
        self.assertTrue(np.all(result.psi >= 0.0))
        self.assertTrue(np.all(result.psi < np.pi))

    def test_geocent_time_in_range(self) -> None:
        """Geocentric times lie within [t_start, t_end]."""
        t_start = 1.187e9
        t_end = 1.270e9
        result = create_injection_set(
            catalogue=_make_catalogue(),
            population=_make_population(),
            cosmology=Planck18,
            n_draw=30,
            t_start=t_start,
            t_end=t_end,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.geocent_time >= t_start))
        self.assertTrue(np.all(result.geocent_time <= t_end))

    def test_reproducible(self) -> None:
        """Same seed produces identical output."""
        kwargs = dict(
            catalogue=_make_catalogue(),
            population=_make_population(),
            cosmology=Planck18,
            n_draw=10,
        )
        r1 = create_injection_set(**kwargs, rng=np.random.default_rng(5))
        r2 = create_injection_set(**kwargs, rng=np.random.default_rng(5))
        np.testing.assert_array_equal(r1.m1_source, r2.m1_source)
        np.testing.assert_array_equal(r1.host_galaxy_index, r2.host_galaxy_index)

    def test_all_fields_same_length(self) -> None:
        """All fields of the returned InjectionSet have the same length."""
        result = _make_injection_set(12)
        n = len(result)
        fields = [
            result.m1_source,
            result.m2_source,
            result.a1,
            result.a2,
            result.cos_tilt1,
            result.cos_tilt2,
            result.phi12,
            result.phi_jl,
            result.theta_jn,
            result.ra,
            result.dec,
            result.psi,
            result.geocent_time,
            result.redshift,
            result.luminosity_distance,
            result.host_galaxy_index,
        ]
        for f in fields:
            self.assertEqual(len(f), n)
