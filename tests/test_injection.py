"""Tests for bagpuss.injection."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

import numpy as np
import zarr
from astropy.cosmology import Planck18

from bagpuss.catalogue import GalaxyCatalogue, MagnitudeLimitedSurvey
from bagpuss.injection import (
    Detectable,
    DistanceThreshold,
    HostAssignment,
    InjectionSet,
    assemble_injection_set,
    build_injection_set,
    create_injection_set,
    sample_host_galaxies,
)
from bagpuss.luminosity import SchechterLuminosityModel
from bagpuss.population import (
    BBHSet,
    ConstantMergerRate,
    IsotropicSpinDistribution,
    PopulationModel,
    PowerLawPlusPeakMassDistribution,
    expected_n_mergers,
)
from bagpuss.universe import PointProcess, Universe

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_N_GALAXIES = 20
_N_EVENTS = 30
_Z_MAX = 0.3
_M_LIM = 19.5

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

_LUMINOSITY_PARAMS = dict(
    phi_star=1.61e-2,
    m_star=-19.66,
    alpha=-1.16,
    m_min=-25.0,
    m_max=-14.0,
)


def _make_catalogue(n: int = _N_GALAXIES) -> GalaxyCatalogue:
    """Build a hand-crafted catalogue, independent of any Universe/selection."""
    rng = np.random.default_rng(0)
    return GalaxyCatalogue(
        redshifts=rng.uniform(0.01, 0.3, n),
        luminosities=np.ones(n) * 1e10,
        apparent_magnitudes=np.ones(n) * 18.0,
        ra=rng.uniform(0.0, 2 * np.pi, n),
        dec=np.arcsin(rng.uniform(-1.0, 1.0, n)),
    )


def _make_universe(z_max: float = _Z_MAX) -> Universe:
    return Universe(
        cosmology=Planck18,
        structure=PointProcess(z_max=z_max),
        luminosity=SchechterLuminosityModel(**_LUMINOSITY_PARAMS),
    )


def _make_selection(m_lim: float = _M_LIM) -> MagnitudeLimitedSurvey:
    return MagnitudeLimitedSurvey(m_lim=m_lim)


def _make_consistent_catalogue(
    universe: Universe,
    selection: MagnitudeLimitedSurvey,
    n: int = 5_000,
    seed: int = 0,
) -> GalaxyCatalogue:
    """Build a catalogue actually drawn from ``universe`` and filtered by ``selection``.

    Used wherever a test needs the catalogue's host distribution to be
    consistent with the universe/selection passed to ``create_injection_set``.
    Uses a fixed draw size rather than a Poisson realisation so tests stay fast.
    """
    galaxies = universe.sample(n, rng=np.random.default_rng(seed))
    return selection.apply(galaxies, universe.cosmology)


def _make_population() -> PopulationModel:
    return PopulationModel(
        mass=PowerLawPlusPeakMassDistribution(**_MASS_PARAMS),
        spin=IsotropicSpinDistribution(),
    )


def _make_rate(rate_density_value: float = 1e-4) -> ConstantMergerRate:
    return ConstantMergerRate(rate_density_value=rate_density_value)


def _make_host_assignment(n: int = _N_EVENTS) -> HostAssignment:
    """Build a simple HostAssignment with all uncatalogued hosts."""
    rng = np.random.default_rng(0)
    return HostAssignment(
        redshift=rng.uniform(0.01, 0.3, n),
        ra=rng.uniform(0.0, 2 * np.pi, n),
        dec=np.arcsin(rng.uniform(-1.0, 1.0, n)),
        host_galaxy_index=np.full(n, -1, dtype=int),
    )


def _make_bbh_set(n: int = _N_EVENTS) -> BBHSet:
    """Build a BBHSet by sampling from the default population model."""
    return _make_population().sample(n, rng=np.random.default_rng(1))


def _make_injection_set(n: int = _N_EVENTS) -> InjectionSet:
    return create_injection_set(
        universe=_make_universe(),
        catalogue=_make_catalogue(),
        selection=_make_selection(),
        population=_make_population(),
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

    def test_to_zarr_from_zarr_path_round_trip(self) -> None:
        """to_zarr/from_zarr round-trip all fields via a store path."""
        inj = self._make(10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "injections.zarr"
            inj.to_zarr(path)
            loaded = InjectionSet.from_zarr(path)
        self.assertEqual(len(loaded), len(inj))
        for field in self._fields:
            np.testing.assert_allclose(
                getattr(loaded, field),
                getattr(inj, field),
                err_msg=f"Round-trip mismatch for field: {field}",
            )

    def test_to_zarr_from_zarr_group_round_trip(self) -> None:
        """to_zarr/from_zarr accept an already-open zarr.Group (shard use case)."""
        inj = self._make(5)
        with tempfile.TemporaryDirectory() as tmp:
            root = zarr.open_group(store=str(Path(tmp) / "store.zarr"), mode="a")
            shard = root.create_group("shard_0000")
            inj.to_zarr(shard)

            root_r = zarr.open_group(store=str(Path(tmp) / "store.zarr"), mode="r")
            loaded = InjectionSet.from_zarr(cast(zarr.Group, root_r["shard_0000"]))
        np.testing.assert_allclose(loaded.m1_source, inj.m1_source)

    @property
    def _fields(self) -> list[str]:
        return self._FIELDS


# ---------------------------------------------------------------------------
# HostAssignment
# ---------------------------------------------------------------------------


class TestHostAssignment(unittest.TestCase):
    """Tests for the HostAssignment dataclass."""

    def test_len(self) -> None:
        """__len__ returns the number of assignments."""
        ha = _make_host_assignment(9)
        self.assertEqual(len(ha), 9)

    def test_all_fields_stored(self) -> None:
        """All expected fields are present."""
        ha = _make_host_assignment(3)
        for field in ("redshift", "ra", "dec", "host_galaxy_index"):
            self.assertTrue(hasattr(ha, field), f"Missing field: {field}")


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
# sample_host_galaxies
# ---------------------------------------------------------------------------


class TestSampleHostGalaxies(unittest.TestCase):
    """Tests for the sample_host_galaxies function."""

    def test_returns_host_assignment(self) -> None:
        """Return value is a HostAssignment."""
        result = sample_host_galaxies(
            universe=_make_universe(),
            catalogue=_make_catalogue(),
            selection=_make_selection(),
            n=10,
            rng=np.random.default_rng(0),
        )
        self.assertIsInstance(result, HostAssignment)

    def test_length(self) -> None:
        """Returned HostAssignment has the requested length."""
        result = sample_host_galaxies(
            universe=_make_universe(),
            catalogue=_make_catalogue(),
            selection=_make_selection(),
            n=15,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(result), 15)

    def test_empty_catalogue_leaves_all_hosts_uncatalogued(self) -> None:
        """An empty catalogue cannot host anything, even if completeness says so."""
        empty = GalaxyCatalogue(
            redshifts=np.array([]),
            luminosities=np.array([]),
            apparent_magnitudes=np.array([]),
            ra=np.array([]),
            dec=np.array([]),
        )
        selection = _make_selection()
        with mock.patch.object(
            selection, "completeness", side_effect=lambda z, *a, **k: np.ones_like(z)
        ):
            result = sample_host_galaxies(
                universe=_make_universe(),
                catalogue=empty,
                selection=selection,
                n=20,
                rng=np.random.default_rng(0),
            )
        self.assertEqual(len(result), 20)
        self.assertTrue(np.all(result.host_galaxy_index == -1))

    def test_host_galaxy_index_valid_when_observed(self) -> None:
        """Observed-host indices are valid indices into the catalogue."""
        universe = _make_universe()
        selection = _make_selection(m_lim=23.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        result = sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        observed = result.host_galaxy_index >= 0
        self.assertTrue(np.any(observed), "expected at least one observed host")
        self.assertTrue(np.all(result.host_galaxy_index[observed] < len(catalogue)))

    def test_host_galaxy_index_minus_one_when_not_observed(self) -> None:
        """Hosts too faint to be catalogued are flagged with index -1."""
        universe = _make_universe()
        selection = _make_selection(m_lim=23.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        result = sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.any(result.host_galaxy_index == -1))

    def test_all_hosts_observed_for_permissive_limit(self) -> None:
        """An extremely faint limiting magnitude catalogues every host."""
        universe = _make_universe()
        selection = _make_selection(m_lim=100.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        result = sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.host_galaxy_index >= 0))

    def test_no_hosts_observed_for_restrictive_limit(self) -> None:
        """An impossibly bright limiting magnitude catalogues no host."""
        universe = _make_universe()
        selection = _make_selection(m_lim=-100.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        result = sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.host_galaxy_index == -1))

    def test_ra_dec_match_catalogue_when_observed(self) -> None:
        """RA/Dec of an observed-host assignment match that catalogue entry."""
        universe = _make_universe()
        selection = _make_selection(m_lim=23.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        result = sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        observed = result.host_galaxy_index >= 0
        idx = result.host_galaxy_index[observed]
        np.testing.assert_array_equal(result.ra[observed], catalogue.ra[idx])
        np.testing.assert_array_equal(result.dec[observed], catalogue.dec[idx])

    def test_redshift_matches_catalogue_when_observed(self) -> None:
        """Redshift of an observed-host assignment matches that catalogue entry."""
        universe = _make_universe()
        selection = _make_selection(m_lim=23.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        result = sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=_N_EVENTS,
            rng=np.random.default_rng(0),
        )
        observed = result.host_galaxy_index >= 0
        idx = result.host_galaxy_index[observed]
        np.testing.assert_array_equal(
            result.redshift[observed], catalogue.redshifts[idx]
        )

    def test_redshift_override_used_verbatim_for_uncatalogued_hosts(self) -> None:
        """A precomputed redshift array is used in place of structure sampling."""
        universe = _make_universe()
        selection = _make_selection(m_lim=-100.0)  # nothing gets catalogued
        catalogue = _make_consistent_catalogue(universe, selection)
        preset_redshift = np.array([0.05, 0.1, 0.15, 0.2])
        result = sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=999,  # deliberately wrong; redshift's length should win
            rng=np.random.default_rng(0),
            redshift=preset_redshift,
        )
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result.redshift, preset_redshift)

    def test_redshift_override_not_mutated(self) -> None:
        """The caller's redshift array is not mutated in place."""
        universe = _make_universe()
        selection = _make_selection(m_lim=23.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        preset_redshift = np.full(200, 0.2)
        original = preset_redshift.copy()
        sample_host_galaxies(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            n=200,
            rng=np.random.default_rng(0),
            redshift=preset_redshift,
        )
        np.testing.assert_array_equal(preset_redshift, original)


# ---------------------------------------------------------------------------
# assemble_injection_set
# ---------------------------------------------------------------------------


class TestAssembleInjectionSet(unittest.TestCase):
    """Tests for the assemble_injection_set function."""

    def test_returns_injection_set(self) -> None:
        """Return value is an InjectionSet."""
        result = assemble_injection_set(
            hosts=_make_host_assignment(),
            bbh=_make_bbh_set(),
            cosmology=Planck18,
            rng=np.random.default_rng(0),
        )
        self.assertIsInstance(result, InjectionSet)

    def test_length_without_detectable(self) -> None:
        """Without a detectability filter all events are returned."""
        result = assemble_injection_set(
            hosts=_make_host_assignment(15),
            bbh=_make_bbh_set(15),
            cosmology=Planck18,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(result), 15)

    def test_length_with_detectable_all_pass(self) -> None:
        """A very permissive threshold keeps all events."""
        result = assemble_injection_set(
            hosts=_make_host_assignment(20),
            bbh=_make_bbh_set(20),
            cosmology=Planck18,
            detectable=DistanceThreshold(d_max=1e6),
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(result), 20)

    def test_length_with_detectable_none_pass(self) -> None:
        """A zero-distance threshold keeps no events."""
        result = assemble_injection_set(
            hosts=_make_host_assignment(20),
            bbh=_make_bbh_set(20),
            cosmology=Planck18,
            detectable=DistanceThreshold(d_max=1e-6),
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(result), 0)

    def test_luminosity_distance_positive(self) -> None:
        """All luminosity distances are positive."""
        result = assemble_injection_set(
            hosts=_make_host_assignment(),
            bbh=_make_bbh_set(),
            cosmology=Planck18,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.luminosity_distance > 0))

    def test_theta_jn_bounds(self) -> None:
        """Inclination angle lies in [0, π]."""
        result = assemble_injection_set(
            hosts=_make_host_assignment(),
            bbh=_make_bbh_set(),
            cosmology=Planck18,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.theta_jn >= 0.0))
        self.assertTrue(np.all(result.theta_jn <= np.pi))

    def test_psi_bounds(self) -> None:
        """Polarisation angle lies in [0, π)."""
        result = assemble_injection_set(
            hosts=_make_host_assignment(),
            bbh=_make_bbh_set(),
            cosmology=Planck18,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.psi >= 0.0))
        self.assertTrue(np.all(result.psi < np.pi))

    def test_geocent_time_in_range(self) -> None:
        """Geocentric times lie within [t_start, t_end]."""
        t_start = 1.187e9
        t_end = 1.270e9
        result = assemble_injection_set(
            hosts=_make_host_assignment(),
            bbh=_make_bbh_set(),
            cosmology=Planck18,
            t_start=t_start,
            t_end=t_end,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(np.all(result.geocent_time >= t_start))
        self.assertTrue(np.all(result.geocent_time <= t_end))

    def test_host_galaxy_index_preserved(self) -> None:
        """host_galaxy_index values from HostAssignment survive into InjectionSet."""
        hosts = _make_host_assignment(10)
        hosts.host_galaxy_index[0] = 3
        result = assemble_injection_set(
            hosts=hosts,
            bbh=_make_bbh_set(10),
            cosmology=Planck18,
            rng=np.random.default_rng(0),
        )
        self.assertIn(3, result.host_galaxy_index)


# ---------------------------------------------------------------------------
# create_injection_set
# ---------------------------------------------------------------------------


class TestCreateInjectionSet(unittest.TestCase):
    """Integration tests for the create_injection_set convenience wrapper."""

    def test_returns_injection_set(self) -> None:
        """create_injection_set returns an InjectionSet."""
        result = _make_injection_set(10)
        self.assertIsInstance(result, InjectionSet)

    def test_length_without_detectable(self) -> None:
        """Without a detectability filter all n_draw events are returned."""
        result = _make_injection_set(15)
        self.assertEqual(len(result), 15)

    def test_reproducible(self) -> None:
        """Same seed produces identical output."""
        kwargs: dict[str, Any] = dict(
            universe=_make_universe(),
            catalogue=_make_catalogue(),
            selection=_make_selection(),
            population=_make_population(),
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


# ---------------------------------------------------------------------------
# build_injection_set
# ---------------------------------------------------------------------------

#: Small enough to keep the Poisson-realised event count in the low hundreds
#: for the _Z_MAX = 0.3, one-year test window, while still large enough that
#: n > 0 is essentially certain.
_TEST_RATE_DENSITY = 2e-8
_ONE_YEAR_SECONDS = 365.25 * 86400.0


def _make_physical_injection_set(
    rate_density_value: float = _TEST_RATE_DENSITY,
    rng: np.random.Generator | None = None,
) -> InjectionSet:
    universe = _make_universe()
    selection = _make_selection(m_lim=23.0)
    catalogue = _make_consistent_catalogue(universe, selection)
    return build_injection_set(
        universe=universe,
        catalogue=catalogue,
        selection=selection,
        population=_make_population(),
        rate=_make_rate(rate_density_value),
        t_start=0.0,
        t_end=_ONE_YEAR_SECONDS,
        rng=rng if rng is not None else np.random.default_rng(42),
    )


class TestBuildInjectionSet(unittest.TestCase):
    """Tests for the rate-driven build_injection_set convenience wrapper."""

    def test_returns_injection_set(self) -> None:
        """build_injection_set returns an InjectionSet."""
        result = _make_physical_injection_set()
        self.assertIsInstance(result, InjectionSet)

    def test_nonzero_events_drawn(self) -> None:
        """A reasonable rate over a one-year window yields some events."""
        result = _make_physical_injection_set()
        self.assertGreater(len(result), 0)

    def test_reproducible(self) -> None:
        """Same seed produces an identical event count and parameters."""
        r1 = _make_physical_injection_set(rng=np.random.default_rng(5))
        r2 = _make_physical_injection_set(rng=np.random.default_rng(5))
        self.assertEqual(len(r1), len(r2))
        np.testing.assert_array_equal(r1.m1_source, r2.m1_source)
        np.testing.assert_array_equal(r1.redshift, r2.redshift)

    def test_event_count_scales_with_rate(self) -> None:
        """A much higher rate density yields more events (statistically)."""
        low = _make_physical_injection_set(
            rate_density_value=_TEST_RATE_DENSITY, rng=np.random.default_rng(1)
        )
        high = _make_physical_injection_set(
            rate_density_value=_TEST_RATE_DENSITY * 20.0,
            rng=np.random.default_rng(1),
        )
        self.assertGreater(len(high), len(low))

    def test_redshifts_within_z_max(self) -> None:
        """All event redshifts lie within the structure model's z_max."""
        result = _make_physical_injection_set()
        self.assertTrue(np.all(result.redshift >= 0.0))
        self.assertTrue(np.all(result.redshift <= _Z_MAX))

    def test_all_fields_same_length(self) -> None:
        """All fields of the returned InjectionSet have the same length."""
        result = _make_physical_injection_set()
        n = len(result)
        for field in TestInjectionSet._FIELDS:
            self.assertEqual(len(getattr(result, field)), n)

    def test_no_detectable_keeps_all_drawn_events(self) -> None:
        """Without a detectability filter, every Poisson-drawn event survives."""
        universe = _make_universe()
        selection = _make_selection(m_lim=23.0)
        catalogue = _make_consistent_catalogue(universe, selection)
        rng = np.random.default_rng(9)
        n_expected = expected_n_mergers(
            universe.structure,
            universe.cosmology,
            _make_rate(_TEST_RATE_DENSITY),
            t_start=0.0,
            t_end=_ONE_YEAR_SECONDS,
        )
        result = build_injection_set(
            universe=universe,
            catalogue=catalogue,
            selection=selection,
            population=_make_population(),
            rate=_make_rate(_TEST_RATE_DENSITY),
            t_start=0.0,
            t_end=_ONE_YEAR_SECONDS,
            rng=rng,
        )
        # n_expected is a mean, not the exact draw; just sanity-check it's
        # in the right ballpark (>3-sigma would be suspicious).
        tolerance = 5.0 * np.sqrt(n_expected)
        self.assertLess(abs(len(result) - n_expected), tolerance)
