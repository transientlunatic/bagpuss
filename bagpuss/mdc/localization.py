"""BAYESTAR skymaps for injections, simulated the way a search would see them.

Downstream PE codes such as simple-pe take their sky localisation from the
search pipeline's BAYESTAR skymap. An MDC has no search, so this module
produces the equivalent: for one injection it writes the ``sim_inspiral``
table ``lalapps_inspinj`` would, has ``bayestar-realize-coincs`` simulate the
single-detector triggers a matched-filter search would record (Gaussian
measurement errors on SNR, phase and arrival time, for the detectors that were
observing), and ``bayestar-localize-coincs`` turn those into a FITS skymap.

The skymap is therefore derived from the injected parameters and the detector
PSDs, not from the frames' noise realisation. Only the three BAYESTAR
command-line tools (``ligo.skymap``) and ``igwn-ligolw`` are needed, and both
are imported lazily so the rest of :mod:`bagpuss.mdc` does not depend on them.
"""

# igwn-ligolw is an optional, lazily-imported dependency (see the module docstring).
# pyright: reportMissingImports=false

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np

__all__: list[str] = ["BayestarLocalizer", "Localizer", "PSD_MODELS"]

#: minke PSD name -> the ``bayestar-sample-model-psd`` curve with the same
#: lalsimulation noise model (``SimNoisePSD<name>``), so the skymap is
#: simulated with the sensitivity the frames were made with.
PSD_MODELS: dict[str, str] = {
    "AdvancedLIGO": "aLIGODesignSensitivityT1800044",
    "AdvancedLIGO_O4": "aLIGODesignSensitivityT1800044",
    "AdvancedVirgo_O4": "AdVO4T1800545",
}


class Localizer(Protocol):
    """Produces a skymap for one event; ``None`` if it could not."""

    def localize(
        self,
        event: dict[str, Any],
        observing: dict[str, str],
        psds: dict[str, str],
        seed: int,
        out_path: Path,
    ) -> Path | None:
        """Write a skymap for *event* to *out_path* and return it.

        Parameters
        ----------
        event : dict
            Waveform parameters (``m1``, ``m2``, ``S1z``, ``S2z``,
            ``luminosity_distance``, ``ra``, ``dec``, ``inclination``,
            ``psi``, ``gpstime``), as returned by the detection backend.
        observing : dict[str, str]
            Observing detectors as ``{abbreviation: detector name}``.
        psds : dict[str, str]
            ``{abbreviation: minke PSD name}`` for the same detectors.
        seed : int
            Seed for the simulated measurement errors.
        out_path : pathlib.Path
            Where to write the FITS file.
        """
        ...


def _value(x: Any) -> float:  # noqa: ANN401
    """Return *x* as a plain float, unwrapping an astropy Quantity."""
    return float(getattr(x, "value", x))


class BayestarLocalizer:
    """:class:`Localizer` using the ``ligo.skymap`` BAYESTAR tools.

    Parameters
    ----------
    waveform : str, optional
        Template used by BAYESTAR. An analytic approximant, so no waveform
        data files are needed (the default ``o2-uberbank`` selects
        ``SEOBNRv4_ROM``, whose data is not always installed).
    f_low : float, optional
        Low-frequency cutoff in Hz.
    snr_threshold : float, optional
        Single-detector SNR below which a detector's trigger is dropped.
    threads : int, optional
        OpenMP threads for BAYESTAR. It uses every core by default, which
        oversubscribes a batch job that was given one or two CPUs.
    bin_dir : pathlib.Path or None, optional
        Directory holding the ``bayestar-*`` executables; defaults to the
        running interpreter's ``bin`` directory.
    """

    def __init__(
        self,
        waveform: str = "IMRPhenomD",
        f_low: float = 20.0,
        snr_threshold: float = 4.0,
        threads: int = 1,
        bin_dir: Path | None = None,
    ) -> None:
        self.waveform = waveform
        self.f_low = f_low
        self.snr_threshold = snr_threshold
        self.threads = threads
        self.bin_dir = bin_dir if bin_dir is not None else Path(sys.executable).parent

    # -- helpers ---------------------------------------------------------

    def _run(self, tool: str, *args: str | Path) -> None:
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads))
        subprocess.run(
            [str(self.bin_dir / tool), *map(str, args)],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _write_injection_table(event: dict[str, Any], path: Path, f_low: float) -> None:
        """Write a one-row ``sim_inspiral`` table, as ``lalapps_inspinj`` would.

        *f_low* is the low-frequency cutoff recorded for the injection, so that
        every BAYESTAR step uses the same value.
        """
        from igwn_ligolw import ligolw, lsctables
        from igwn_ligolw import utils as ligolw_utils
        from igwn_ligolw.utils import process as ligolw_process

        xmldoc = ligolw.Document()
        root = xmldoc.appendChild(ligolw.LIGO_LW())
        process = ligolw_process.register_to_xmldoc(xmldoc, "bagpuss-mdc", {})
        table = lsctables.New(lsctables.SimInspiralTable)
        root.appendChild(table)
        row = table.RowType()
        for column, kind in lsctables.SimInspiralTable.validcolumns.items():
            default = (
                "" if kind == "lstring" else (0 if kind.startswith("int") else 0.0)
            )
            setattr(row, column.split(":")[-1], default)

        gps = _value(event["gpstime"])
        row.mass1, row.mass2 = _value(event["m1"]), _value(event["m2"])
        # A search's templates are aligned-spin, and BAYESTAR rejects
        # transverse spins for them, so only the aligned components are kept.
        row.spin1z, row.spin2z = _value(event["S1z"]), _value(event["S2z"])
        row.distance = _value(event["luminosity_distance"])
        row.longitude, row.latitude = _value(event["ra"]), _value(event["dec"])
        row.inclination = _value(event["inclination"])
        row.polarization = _value(event["psi"])
        row.coa_phase = 0.0
        row.geocent_end_time = int(np.floor(gps))
        row.geocent_end_time_ns = int(round((gps - np.floor(gps)) * 1e9))
        row.f_lower = f_low
        row.waveform = "IMRPhenomPv2"
        row.simulation_id = 0
        row.process_id = process.process_id
        table.append(row)
        ligolw_utils.write_filename(xmldoc, str(path))

    # -- public ----------------------------------------------------------

    def localize(
        self,
        event: dict[str, Any],
        observing: dict[str, str],
        psds: dict[str, str],
        seed: int,
        out_path: Path,
    ) -> Path | None:
        """Write a BAYESTAR skymap for *event* to *out_path*; ``None`` on failure."""
        ifos = sorted(observing)
        if not ifos:
            return None
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                self._write_injection_table(event, work / "sim.xml", self.f_low)

                psd_args: list[str] = []
                for ifo in ifos:
                    psd_args += [f"--{ifo}", PSD_MODELS[psds[ifo]]]
                self._run(
                    "bayestar-sample-model-psd",
                    "-o",
                    work / "psd.xml",
                    "--df",
                    "0.125",
                    "--f-max",
                    "2048",
                    *psd_args,
                )
                self._run(
                    "bayestar-realize-coincs",
                    work / "sim.xml",
                    "-o",
                    work / "coinc.xml",
                    "--detector",
                    *ifos,
                    "--reference-psd",
                    work / "psd.xml",
                    "--measurement-error",
                    "gaussian-noise",
                    "--snr-threshold",
                    str(self.snr_threshold),
                    "--net-snr-threshold",
                    "0",
                    "--min-triggers",
                    "1",
                    "--f-low",
                    str(self.f_low),
                    "--seed",
                    str(seed),
                    "--keep-subthreshold",
                )
                self._run(
                    "bayestar-localize-coincs",
                    work / "coinc.xml",
                    "--output",
                    work,
                    "--f-low",
                    str(self.f_low),
                    "--waveform",
                    self.waveform,
                    "--cosmology",
                )
                fits = sorted(work.glob("*.fits"))
                if len(fits) != 1:
                    return None
                shutil.move(str(fits[0]), str(out_path))
        except (subprocess.CalledProcessError, KeyError, OSError):
            return None
        return out_path
