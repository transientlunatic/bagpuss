"""Tests for bagpuss.mdc.detection (the minke-backed detection backend)."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from bagpuss.injection import InjectionSet
from bagpuss.mdc.config import MDCConfig
from bagpuss.mdc.detection import MinkeBackend, _cap_eta


def _injections(n: int = 2) -> InjectionSet:
    ones = np.ones(n)
    return InjectionSet(
        m1_source=30 * ones,
        m2_source=20 * ones,
        a1=0.5 * ones,
        a2=0.5 * ones,
        cos_tilt1=0.1 * ones,
        cos_tilt2=0.2 * ones,
        phi12=ones,
        phi_jl=ones,
        theta_jn=ones,
        ra=ones,
        dec=0.1 * ones,
        psi=ones,
        geocent_time=1.2e9 * ones,
        redshift=0.1 * ones,
        luminosity_distance=500 * ones,
        host_galaxy_index=-np.ones(n, dtype=int),
    )


def _fake_minke(returned_key: str) -> dict[str, types.ModuleType]:
    """Return a stub ``minke.bagpuss`` whose reader emits *returned_key*."""

    def read_injection_parameters(path: str, f_ref: float = 20.0) -> list[dict]:
        return [{returned_key: 1.23, "gpstime": 1.2e9} for _ in range(2)]

    bagpuss_mod = types.ModuleType("minke.bagpuss")
    bagpuss_mod.read_injection_parameters = read_injection_parameters  # type: ignore[attr-defined]
    minke_mod = types.ModuleType("minke")
    minke_mod.bagpuss = bagpuss_mod  # type: ignore[attr-defined]
    return {"minke": minke_mod, "minke.bagpuss": bagpuss_mod}


class TestMinkeBackendInjectionParameters(unittest.TestCase):
    """minke names the inclination ``iota`` but its waveforms read ``inclination``."""

    def test_iota_renamed_to_inclination(self) -> None:
        """Without the rename minke silently injects every event face-on."""
        with mock.patch.dict(sys.modules, _fake_minke("iota")):
            params = MinkeBackend().injection_parameters(_injections(), f_ref=20.0)
        for event in params:
            self.assertNotIn("iota", event)
            self.assertAlmostEqual(event["inclination"], 1.23)

    def test_already_inclination_left_alone(self) -> None:
        """A minke that emits ``inclination`` directly is passed through."""
        with mock.patch.dict(sys.modules, _fake_minke("inclination")):
            params = MinkeBackend().injection_parameters(_injections(), f_ref=20.0)
        for event in params:
            self.assertAlmostEqual(event["inclination"], 1.23)


class _Quantity:
    """Stand-in for an astropy Quantity (just a ``.value``)."""

    def __init__(self, value: float) -> None:
        self.value = value


def _fake_blueprint_minke() -> dict[str, types.ModuleType]:
    def make_blueprint(
        param: dict,
        chirp_mass_margin: float = 0.5,
        frame_files: dict | None = None,
        network_snr: float | None = None,
    ) -> dict:
        bp: dict = {"kind": "event", "name": f"inj_{param['gpstime']:.3f}"}
        if frame_files:
            bp["interferometers"] = sorted(frame_files)
        return bp

    class _Det:
        def __init__(self, abbreviation: str) -> None:
            self.abbreviation = abbreviation

    known = {
        "AdvancedLIGOHanford": lambda: _Det("H1"),
        "AdvancedVirgo": lambda: _Det("V1"),
    }
    bagpuss_mod = types.ModuleType("minke.bagpuss")
    bagpuss_mod.make_blueprint = make_blueprint  # type: ignore[attr-defined]
    detector_mod = types.ModuleType("minke.detector")
    detector_mod.KNOWN_IFOS = known  # type: ignore[attr-defined]
    minke_mod = types.ModuleType("minke")
    return {
        "minke": minke_mod,
        "minke.bagpuss": bagpuss_mod,
        "minke.detector": detector_mod,
    }


class TestMissingDutyCycle(unittest.TestCase):
    """minke 2.2.1 has no duty_cycle module; the error should say what to do."""

    def test_clear_error(self) -> None:
        """A minke without minke.duty_cycle gives an actionable ImportError."""
        with mock.patch.dict(sys.modules, {"minke": types.ModuleType("minke")}):
            sys.modules.pop("minke.duty_cycle", None)
            with self.assertRaises(ImportError) as ctx:
                MinkeBackend().duty_schedules(
                    MDCConfig(), 0.0, 1.0, np.random.default_rng(0)
                )
        self.assertIn("after 2.2.1", str(ctx.exception))


class TestCapEta(unittest.TestCase):
    """Capping the trigger's symmetric mass ratio, holding the chirp mass fixed."""

    @staticmethod
    def _eta(m1: float, m2: float) -> float:
        return m1 * m2 / (m1 + m2) ** 2

    @staticmethod
    def _mc(m1: float, m2: float) -> float:
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

    def test_none_leaves_masses_alone(self) -> None:
        """No cap means no change."""
        self.assertEqual(_cap_eta(30.0, 29.9, None), (30.0, 29.9))

    def test_below_cap_unchanged(self) -> None:
        """Masses already below the cap are unchanged."""
        self.assertEqual(_cap_eta(30.0, 10.0, 0.249), (30.0, 10.0))

    def test_equal_mass_capped_with_chirp_mass_fixed(self) -> None:
        """Equal masses are moved to the cap at fixed chirp mass."""
        m1, m2 = _cap_eta(30.0, 30.0, 0.249)
        self.assertAlmostEqual(self._eta(m1, m2), 0.249)
        self.assertAlmostEqual(self._mc(m1, m2), self._mc(30.0, 30.0))
        self.assertGreater(m1, m2)

    def test_near_equal_mass_capped(self) -> None:
        """Near-equal masses above the cap are capped."""
        m1, m2 = _cap_eta(31.0, 30.5, 0.249)
        self.assertLessEqual(self._eta(m1, m2), 0.249 + 1e-12)


class TestMinkeBackendBlueprints(unittest.TestCase):
    """Frameless blueprints carry what minke's asimov pipeline needs."""

    def _write(
        self, frames: list[dict], skymap: Path | None = None, **config: float | None
    ) -> list[dict]:
        cfg = MDCConfig(**config)  # type: ignore[arg-type]
        event = {
            "m1": _Quantity(30.0),
            "m2": _Quantity(25.0),
            "gpstime": 1.2e9,
            "inclination": 1.0,
            "redshift": 0.4,
            "S1z": 0.3,
            "S2z": -0.1,
            "ra": 2.0,
            "dec": 0.5,
            "psi": 0.7,
            "luminosity_distance": _Quantity(900.0),
        }
        observing = {
            "AdvancedLIGOHanford": "AdvancedLIGO_O4",
            "AdvancedVirgo": "AdvancedVirgo_O4",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(sys.modules, _fake_blueprint_minke()),
        ):
            path = Path(tmp) / "bp.yaml"
            MinkeBackend().write_blueprints(
                [event], path, cfg, frames, [12.5], [observing], [skymap]
            )
            return [d for d in yaml.safe_load_all(path.read_text()) if d]

    def test_injection_block_when_no_frames(self) -> None:
        """No frames -> data channels and an injection block are written."""
        (bp,) = self._write([{}])
        self.assertEqual(bp["interferometers"], ["H1", "V1"])
        self.assertEqual(
            bp["data"]["channels"], {"H1": "H1:Injection", "V1": "V1:Injection"}
        )
        inj = bp["injection"]
        self.assertEqual(
            inj["interferometers"], {"H1": "AdvancedLIGOHanford", "V1": "AdvancedVirgo"}
        )
        self.assertEqual(
            inj["psds"], {"H1": "AdvancedLIGO_O4", "V1": "AdvancedVirgo_O4"}
        )
        self.assertAlmostEqual(
            inj["epoch"], 1.2e9 - MDCConfig().frame_pre_signal_padding
        )
        self.assertEqual(inj["parameters"]["m1"], 30.0)
        self.assertEqual(inj["parameters"]["inclination"], 1.0)
        self.assertNotIn("redshift", inj["parameters"])
        self.assertAlmostEqual(bp["injected"]["network snr"], 12.5)
        self.assertEqual(
            bp["trigger"],
            {
                "mass1": 30.0,
                "mass2": 25.0,
                "spin1z": 0.3,
                "spin2z": -0.1,
                "ra": 2.0,
                "dec": 0.5,
                "psi": 0.7,
                "phase": 0.0,
                "distance": 900.0,
                "inclination": 1.0,
            },
        )

    def test_trigger_capped_and_truth_kept(self) -> None:
        """With ``trigger_eta_max`` the trigger moves off eta=0.25 and truth is kept."""
        (bp,) = self._write([{}], trigger_eta_max=0.2)
        t = bp["trigger"]
        self.assertLessEqual(
            t["mass1"] * t["mass2"] / (t["mass1"] + t["mass2"]) ** 2, 0.2 + 1e-12
        )
        self.assertEqual(bp["trigger truth"], {"mass1": 30.0, "mass2": 25.0})

    def test_no_trigger_truth_when_not_capped(self) -> None:
        """Without a cap (or below it) no truth block is written."""
        (bp,) = self._write([{}])
        self.assertNotIn("trigger truth", bp)

    def test_localization_file_recorded(self) -> None:
        """A skymap path is recorded as ``localization file``."""
        (bp,) = self._write([{}], skymap=Path("/sky/inj.fits"))
        self.assertEqual(bp["localization file"], "/sky/inj.fits")

    def test_no_localization_file_without_skymap(self) -> None:
        """No skymap, no key."""
        (bp,) = self._write([{}])
        self.assertNotIn("localization file", bp)

    def test_no_injection_block_when_frames_exist(self) -> None:
        """With frames the blueprint points at them instead."""
        (bp,) = self._write([{"H1": {"path": "x", "channel": "c", "snr": 1.0}}])
        self.assertNotIn("injection", bp)
        self.assertNotIn("trigger", bp)


if __name__ == "__main__":
    unittest.main()
