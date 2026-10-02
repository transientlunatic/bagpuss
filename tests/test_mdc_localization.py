"""Tests for bagpuss.mdc.localization (BAYESTAR skymaps for injections)."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from bagpuss.mdc.localization import PSD_MODELS, BayestarLocalizer

_EVENT: dict[str, Any] = {
    "m1": 30.0,
    "m2": 25.0,
    "S1z": 0.1,
    "S2z": -0.2,
    "luminosity_distance": 900.0,
    "ra": 2.0,
    "dec": 0.5,
    "inclination": 1.0,
    "psi": 0.7,
    "gpstime": 1.2e9,
}
_OBSERVING = {"H1": "AdvancedLIGOHanford", "V1": "AdvancedVirgo"}
_PSDS = {"H1": "AdvancedLIGO_O4", "V1": "AdvancedVirgo_O4"}


class _Recorder:
    """Stand-in for subprocess.run that records calls and fakes the FITS output."""

    def __init__(self, fail_on: str | None = None, n_fits: int = 1) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.fail_on = fail_on
        self.n_fits = n_fits

    def __call__(self, cmd: list[str], **kwargs: Any) -> None:  # noqa: ANN401
        self.calls.append(cmd)
        self.envs.append(kwargs["env"])
        tool = Path(cmd[0]).name
        if tool == self.fail_on:
            raise subprocess.CalledProcessError(1, cmd)
        if tool == "bayestar-localize-coincs":
            outdir = Path(cmd[cmd.index("--output") + 1])
            for k in range(self.n_fits):
                (outdir / f"{k}.fits").write_text("fits")


class TestBayestarLocalizer(unittest.TestCase):
    """The three BAYESTAR tools are run in order, single-threaded, per event."""

    def _localize(
        self, recorder: _Recorder, observing: dict[str, str] | None = None
    ) -> tuple[Path | None, str | None]:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sky" / "event.fits"
            localizer = BayestarLocalizer(bin_dir=Path("/opt/bin"))
            with (
                mock.patch.object(BayestarLocalizer, "_write_injection_table"),
                mock.patch("subprocess.run", recorder),
            ):
                result = localizer.localize(
                    _EVENT,
                    _OBSERVING if observing is None else observing,
                    _PSDS,
                    7,
                    out,
                )
            self.assertEqual(result is not None, out.exists())
            self.assertEqual(result, out if result is not None else None)
            return result, out.read_text() if out.exists() else None

    def test_runs_tools_in_order(self) -> None:
        """PSD, then simulated triggers, then localisation."""
        rec = _Recorder()
        self._localize(rec)
        self.assertEqual(
            [Path(c[0]).name for c in rec.calls],
            [
                "bayestar-sample-model-psd",
                "bayestar-realize-coincs",
                "bayestar-localize-coincs",
            ],
        )
        self.assertTrue(all(c[0].startswith("/opt/bin/") for c in rec.calls))

    def test_psd_models_match_detectors(self) -> None:
        """Each observing detector gets the PSD model its frames used."""
        rec = _Recorder()
        self._localize(rec)
        psd = rec.calls[0]
        self.assertEqual(psd[psd.index("--H1") + 1], PSD_MODELS["AdvancedLIGO_O4"])
        self.assertEqual(psd[psd.index("--V1") + 1], PSD_MODELS["AdvancedVirgo_O4"])

    def test_only_observing_detectors_and_seed(self) -> None:
        """The simulated triggers use the observing detectors and the given seed."""
        rec = _Recorder()
        self._localize(rec)
        realize = rec.calls[1]
        i = realize.index("--detector")
        self.assertEqual(realize[i + 1 : i + 3], ["H1", "V1"])
        self.assertEqual(realize[realize.index("--seed") + 1], "7")

    def test_single_threaded(self) -> None:
        """BAYESTAR would otherwise use every core of a batch node."""
        rec = _Recorder()
        self._localize(rec)
        self.assertTrue(all(e["OMP_NUM_THREADS"] == "1" for e in rec.envs))

    def test_analytic_template_used(self) -> None:
        """The default template needs waveform data files, so one is chosen."""
        rec = _Recorder()
        self._localize(rec)
        localize = rec.calls[2]
        self.assertEqual(localize[localize.index("--waveform") + 1], "IMRPhenomD")

    def test_f_low_reaches_injection_table_and_tools(self) -> None:
        """The injection table and every BAYESTAR step use the configured cutoff."""
        rec = _Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            localizer = BayestarLocalizer(f_low=30.0, bin_dir=Path("/opt/bin"))
            with (
                mock.patch.object(BayestarLocalizer, "_write_injection_table") as table,
                mock.patch("subprocess.run", rec),
            ):
                localizer.localize(
                    _EVENT, _OBSERVING, _PSDS, 7, Path(tmp) / "sky" / "e.fits"
                )
        self.assertEqual(table.call_args.args[2], 30.0)
        realize, localize = rec.calls[1], rec.calls[2]
        self.assertEqual(realize[realize.index("--f-low") + 1], "30.0")
        self.assertEqual(localize[localize.index("--f-low") + 1], "30.0")

    def test_skymap_moved_to_output_path(self) -> None:
        """The FITS file ends up at the requested path."""
        result, text = self._localize(_Recorder())
        self.assertIsNotNone(result)
        self.assertEqual(text, "fits")

    def test_tool_failure_returns_none(self) -> None:
        """A failing BAYESTAR step gives no skymap rather than an exception."""
        result, _ = self._localize(_Recorder(fail_on="bayestar-localize-coincs"))
        self.assertIsNone(result)

    def test_no_coincidence_returns_none(self) -> None:
        """If the simulated triggers produced no coincidence there is no skymap."""
        result, _ = self._localize(_Recorder(n_fits=0))
        self.assertIsNone(result)

    def test_no_observing_detectors_returns_none(self) -> None:
        """Nothing to localise with."""
        rec = _Recorder()
        result, _ = self._localize(rec, observing={})
        self.assertIsNone(result)
        self.assertEqual(rec.calls, [])


if __name__ == "__main__":
    unittest.main()
