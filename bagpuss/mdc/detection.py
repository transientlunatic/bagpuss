"""minke-backed frame generation and SNR calculation for the MDC pipeline.

Everything in :mod:`bagpuss.mdc.pipeline` that touches gravitational-wave
data -- reading an injection set as waveform parameters, deciding which
detectors are observing, injecting signals and computing matched-filter
SNRs, writing frame files and asimov blueprints -- goes through a
:class:`DetectionBackend`. The production backend, :class:`MinkeBackend`,
delegates to `minke <https://git.ligo.org/>`_ and imports it lazily, so
bagpuss itself neither depends on minke nor needs it installed to
generate catalogues and injections; only the detection stage does. Tests
(and anyone wanting a different waveform engine) can pass their own
backend to :func:`bagpuss.mdc.pipeline.detect_injection_shard`.
"""

# minke is an optional, lazily-imported dependency (see module docstring).
# pyright: reportMissingImports=false

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from bagpuss.injection import InjectionSet

if TYPE_CHECKING:
    from bagpuss.mdc.config import MDCConfig

__all__: list[str] = ["DetectionBackend", "MinkeBackend"]


def _cap_eta(m1: float, m2: float, eta_max: float | None) -> tuple[float, float]:
    """Cap the symmetric mass ratio of ``(m1, m2)`` at *eta_max*.

    The chirp mass is held fixed.

    Returns the masses unchanged if *eta_max* is ``None`` or the pair is
    already below it.
    """
    eta = m1 * m2 / (m1 + m2) ** 2
    if eta_max is None or eta <= eta_max:
        return m1, m2
    chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    total = chirp_mass * eta_max**-0.6
    half_gap = (1.0 - 4.0 * eta_max) ** 0.5
    return total / 2.0 * (1.0 + half_gap), total / 2.0 * (1.0 - half_gap)


def _duty_cycle() -> Any:  # noqa: ANN401
    """Import ``minke.duty_cycle``, explaining what is needed if it is missing."""
    try:
        from minke import duty_cycle
    except ImportError as exc:
        raise ImportError(
            "the detection stage needs minke.duty_cycle, which is in minke 2.3.0 and "
            "later (not in 2.2.1); install a newer minke"
        ) from exc
    return duty_cycle


class DetectionBackend(Protocol):
    """What the detection stage needs from a waveform/frame engine."""

    def injection_parameters(
        self, injections: InjectionSet, f_ref: float
    ) -> list[dict[str, Any]]:
        """Convert *injections* to one engine-ready parameter dict per event."""
        ...

    def abbreviation(self, detector: str) -> str:
        """Return the two-letter abbreviation (e.g. ``"H1"``) of *detector*."""
        ...

    def duty_schedules(
        self,
        config: MDCConfig,
        t_start: float,
        t_end: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Return a locked/unlocked schedule per detector over the window."""
        ...

    def active_detectors(
        self, schedules: dict[str, Any], detectors: dict[str, str], t: float
    ) -> dict[str, str]:
        """Return the subset of *detectors* observing at GPS time *t*."""
        ...

    def inject(
        self,
        params: dict[str, Any],
        detectors: dict[str, str],
        config: MDCConfig,
        framefile: str | None,
    ) -> tuple[dict[str, Any], float]:
        """Inject one event; return ``(frame_files, network_snr)``.

        When *framefile* is ``None`` nothing is written and ``frame_files``
        is empty; the SNR is the same either way.
        """
        ...

    def write_blueprints(
        self,
        params: list[dict[str, Any]],
        path: str | Path,
        config: MDCConfig,
        frame_files: list[dict[str, Any]],
        network_snrs: list[float],
        active: list[dict[str, str]],
        skymaps: list[Path | None],
    ) -> None:
        """Write one asimov event blueprint per entry of *params* to *path*.

        *skymaps* gives each event's skymap file (or ``None``), recorded as
        ``localization file``.

        *active* is each event's observing detectors (``{name: psd}``). Where
        an event has no *frame_files* (frames are made later, by an asimov
        analysis) its blueprint carries the injection settings needed to make
        them.
        """
        ...


class MinkeBackend:
    """:class:`DetectionBackend` implemented with minke (imported lazily)."""

    def injection_parameters(
        self, injections: InjectionSet, f_ref: float
    ) -> list[dict[str, Any]]:
        """Convert *injections* via minke's bagpuss HDF5 reader."""
        from minke.bagpuss import (
            read_injection_parameters,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "injections.h5"
            injections.to_hdf5(path)
            return list(read_injection_parameters(str(path), f_ref=f_ref))

    def abbreviation(self, detector: str) -> str:
        """Return *detector*'s abbreviation via minke."""
        from minke.detector import KNOWN_IFOS

        return str(KNOWN_IFOS[detector]().abbreviation)

    def duty_schedules(
        self,
        config: MDCConfig,
        t_start: float,
        t_end: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Draw one duty-cycle schedule per configured detector."""
        generate_duty_cycle_schedule = _duty_cycle().generate_duty_cycle_schedule

        return {
            name: generate_duty_cycle_schedule(
                duty_cycle=config.duty_cycles[name],
                mean_lock_duration=config.duty_cycle_mean_lock_duration,
                t_start=t_start,
                t_end=t_end,
                rng=rng,
            )
            for name in config.detectors
        }

    def active_detectors(
        self, schedules: dict[str, Any], detectors: dict[str, str], t: float
    ) -> dict[str, str]:
        """Restrict *detectors* to those locked at GPS time *t*."""
        return dict(_duty_cycle().active_detectors(schedules, detectors, t))

    def inject(
        self,
        params: dict[str, Any],
        detectors: dict[str, str],
        config: MDCConfig,
        framefile: str | None,
    ) -> tuple[dict[str, Any], float]:
        """Inject one event with minke and return its frame metadata and SNR."""
        from minke.injection import (
            make_injection,
        )

        _, frame_files, network_snr = make_injection(
            injection_parameters=params,
            detectors=detectors,
            duration=config.frame_duration,
            sample_rate=config.frame_sample_rate,
            epoch=params["gpstime"] - config.frame_pre_signal_padding,
            channel=config.injection_channel,
            framefile=framefile,
        )
        return frame_files, network_snr

    def write_blueprints(
        self,
        params: list[dict[str, Any]],
        path: str | Path,
        config: MDCConfig,
        frame_files: list[dict[str, Any]],
        network_snrs: list[float],
        active: list[dict[str, str]],
        skymaps: list[Path | None],
    ) -> None:
        """Write asimov event blueprints (GPS-time-derived event names).

        With frames, the blueprint points at them. Without, it carries
        ``data.channels`` and an ``injection`` block holding everything
        minke's asimov pipeline needs to make the frames later.
        """
        import yaml
        from minke.bagpuss import make_blueprint
        from minke.detector import KNOWN_IFOS

        blueprints = []
        for event, frames, snr, observing, skymap in zip(
            params, frame_files, network_snrs, active, skymaps, strict=True
        ):
            blueprint = make_blueprint(
                event,
                chirp_mass_margin=config.chirp_mass_margin,
                frame_files=frames or None,
                network_snr=snr if frames else None,
            )
            if not frames:
                abbr = {name: KNOWN_IFOS[name]().abbreviation for name in observing}
                gps = float(event["gpstime"])
                blueprint["interferometers"] = sorted(abbr.values())
                blueprint["data"] = {
                    "channels": {
                        a: f"{a}:{config.injection_channel}" for a in abbr.values()
                    }
                }
                blueprint["injection"] = {
                    "channel": config.injection_channel,
                    "duration": int(config.frame_duration),
                    "sample rate": int(config.frame_sample_rate),
                    "epoch": gps - config.frame_pre_signal_padding,
                    "parameters": {
                        key: float(getattr(value, "value", value))
                        for key, value in event.items()
                        if key != "redshift"
                    },
                    "interferometers": {a: name for name, a in abbr.items()},
                    "psds": {abbr[name]: psd for name, psd in observing.items()},
                }
                # The "trigger" PE codes such as simple-pe start from. There is
                # no search in this MDC, so it is the injected truth (as in the
                # earlier fake-transients dress rehearsal).
                values = {
                    key: float(getattr(value, "value", value))
                    for key, value in event.items()
                }
                mass1, mass2 = _cap_eta(
                    values["m1"], values["m2"], config.trigger_eta_max
                )
                blueprint["trigger"] = {
                    "mass1": mass1,
                    "mass2": mass2,
                    "spin1z": values["S1z"],
                    "spin2z": values["S2z"],
                    "ra": values["ra"],
                    "dec": values["dec"],
                    "psi": values["psi"],
                    "phase": 0.0,
                    "distance": values["luminosity_distance"],
                    "inclination": values["inclination"],
                }
                if (mass1, mass2) != (values["m1"], values["m2"]):
                    blueprint["trigger truth"] = {
                        "mass1": values["m1"],
                        "mass2": values["m2"],
                    }
                blueprint["injected"] = {"network snr": float(snr)}
            if skymap is not None:
                blueprint["localization file"] = str(skymap)
            blueprints.append(blueprint)

        with open(path, "w") as fh:
            yaml.safe_dump_all(
                blueprints, fh, default_flow_style=False, sort_keys=False
            )
