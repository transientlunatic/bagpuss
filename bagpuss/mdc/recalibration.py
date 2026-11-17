r"""Empirical recalibration of the distance posteriors of the ``SimplePE`` analyses.

``SimplePE`` is an approximation to full parameter estimation, and on the
simulated events its distance posteriors are too narrow and, for this
population, centred slightly too far. Comparing the posteriors with the
injected values lets us measure both: the credible level of the true distance
should be uniformly distributed, with its probit having zero mean and unit
standard deviation.

The correction is an affine map of :math:`\ln D_L` about each event's weighted
median :math:`m`,

.. math:: \ln D' = m - s + k\,(\ln D - m),

with one shift :math:`s` and one width factor :math:`k` for the whole
catalogue. It is applied as a change to the *likelihood*, so that the
posterior remains :math:`L'(D')\,\pi(D')`, where :math:`\pi(D)\propto D^{p}`
is the prior of the analysis (:math:`p=2` for ``--distance_prior
UniformVolume``). A sample moved from :math:`D` to :math:`D'` therefore has
its weight multiplied by :math:`\pi(D')/\pi(D)`. This keeps the prior of the
released samples what the population inference assumes
(:func:`bagpuss.population_inference.simplepe_samples`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from scipy.special import ndtri

POSTERIOR_FILE = "posterior_samples.dat"
_DISTANCE_COLUMNS = ("luminosity_distance", "distance")
_WEIGHT_COLUMNS = ("weights", "weight")


@dataclass(frozen=True)
class DistanceRecalibration:
    r"""The shift and width factor of the distance recalibration.

    Attributes
    ----------
    shift : float
        Subtracted from :math:`\ln D_L`: positive moves the posterior closer.
    scale : float
        Factor by which the width in :math:`\ln D_L` is multiplied.
    """

    shift: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.scale > 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")


def credible_level(x: np.ndarray, weights: np.ndarray, value: float) -> float:
    """Return the weighted fraction of samples ``x`` below ``value``."""
    x, weights = np.asarray(x, dtype=float), np.asarray(weights, dtype=float)
    return float(weights[x < value].sum() / weights.sum())


def _weighted_median(x: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(x)
    cumulative = np.cumsum(weights[order])
    return float(x[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def recalibrate_distance(
    distance: np.ndarray,
    weights: np.ndarray,
    calibration: DistanceRecalibration,
    prior_power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Apply a recalibration to one event's weighted distance samples.

    Parameters
    ----------
    distance : numpy.ndarray
        Luminosity distances of the samples, in Mpc.
    weights : numpy.ndarray
        Importance weights of the samples.
    calibration : DistanceRecalibration
        The shift and width factor.
    prior_power : float, optional
        The analysis prior is :math:`\pi(D)\propto D^p`: 2 for a uniform
        volume prior, the default of ``SimplePE``.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The new distances and weights (not normalised).
    """
    distance = np.asarray(distance, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if distance.shape != weights.shape or distance.ndim != 1:
        raise ValueError(
            "distance and weights must be one-dimensional and the same length"
        )
    x = np.log(distance)
    median = _weighted_median(x, weights)
    new_x = median - calibration.shift + calibration.scale * (x - median)
    new_distance = np.exp(new_x)
    return new_distance, weights * np.exp(prior_power * (new_x - x))


def fit_distance_recalibration(
    events: Sequence[tuple[np.ndarray, np.ndarray, float]],
    prior_power: float = 2.0,
    shifts: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> DistanceRecalibration:
    r"""Fit the shift and width factor which calibrate the distance posteriors.

    The credible level of each injected distance is computed in the
    recalibrated, reweighted posterior; the fit minimises
    :math:`\bar z^2+(\sigma_z-1)^2`, where :math:`z=\Phi^{-1}` of the
    credible level. This is the property of a calibrated posterior which does
    not depend on how the population is distributed, and is stable for a few
    hundred events (unlike the central-90 per cent coverage).

    Parameters
    ----------
    events : sequence of (numpy.ndarray, numpy.ndarray, float)
        For each event, the distance samples, their weights, and the true
        distance.
    prior_power : float, optional
        See :func:`recalibrate_distance`.
    shifts, scales : numpy.ndarray, optional
        Grids searched (default: shifts -0.2 to 0.3 and scales 0.7 to 3.0).

    Returns
    -------
    DistanceRecalibration
        The best grid point.
    """
    if not len(events):
        raise ValueError("at least one event is needed to fit a recalibration")
    shifts = np.arange(-0.2, 0.3001, 0.005) if shifts is None else np.asarray(shifts)
    scales = np.arange(0.7, 3.0001, 0.0125) if scales is None else np.asarray(scales)
    # level(s, k) = sum_{x < m + (t + s - m)/k} w r / sum w r, r = e^{p(k-1)(x-m)}
    # (the weight is the prior ratio, up to a constant):
    # one cumulative sum per k and event, and a search for each s
    levels = np.empty((len(events), len(scales), len(shifts)))
    for i, (distance, weights, truth) in enumerate(events):
        x = np.log(np.asarray(distance, dtype=float))
        w = np.asarray(weights, dtype=float)
        order = np.argsort(x)
        xs, ws = x[order], w[order]
        median = float(xs[np.searchsorted(np.cumsum(ws), 0.5 * ws.sum())])
        t = np.log(truth)
        for j, k in enumerate(scales):
            cumulative = np.cumsum(ws * np.exp(prior_power * (k - 1.0) * (xs - median)))
            cut = median + (t + shifts - median) / k
            below = np.searchsorted(xs, cut, side="left")
            levels[i, j] = (
                np.where(below > 0, cumulative[np.clip(below - 1, 0, None)], 0.0)
                / cumulative[-1]
            )
    z = ndtri(np.clip(levels, 1.0e-3, 1.0 - 1.0e-3))
    loss = z.mean(axis=0) ** 2 + (z.std(axis=0) - 1.0) ** 2
    best_k, best_s = np.unravel_index(int(np.argmin(loss)), loss.shape)
    return DistanceRecalibration(
        shift=float(shifts[best_s]), scale=float(scales[best_k])
    )


def _read(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    raw = np.genfromtxt(path, names=True)
    return raw, tuple(raw.dtype.names or ())


def _column(names: Sequence[str], candidates: Sequence[str], path: Path) -> str | None:
    for name in candidates:
        if name in names:
            return name
    return None


def load_events(
    working_dir: str | Path,
    run: str,
    truths: dict[str, float],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Read the distance samples of the events in ``truths`` which have a posterior.

    Parameters
    ----------
    working_dir : str or pathlib.Path
        An asimov project's ``working`` directory.
    run : str
        The analysis name, e.g. ``simplepe-IMRPhenomXPHM``.
    truths : dict
        True luminosity distance in Mpc by event name.
    """
    events = []
    for name in sorted(truths):
        path = Path(working_dir) / name / run / "output" / POSTERIOR_FILE
        if not path.exists() or path.stat().st_size == 0:
            continue
        raw, names = _read(path)
        d_col = _column(names, _DISTANCE_COLUMNS, path)
        if d_col is None:
            continue
        w_col = _column(names, _WEIGHT_COLUMNS, path)
        weights = np.asarray(raw[w_col], dtype=float) if w_col else np.ones(len(raw))
        events.append(
            (np.asarray(raw[d_col], dtype=float), weights, float(truths[name]))
        )
    return events


def _write_recalibrated(
    path: Path,
    target: Path,
    calibration: DistanceRecalibration,
    prior_power: float,
) -> bool:
    """Write one recalibrated posterior file; False if it has no distance column."""
    raw, names = _read(path)
    d_col = _column(names, _DISTANCE_COLUMNS, path)
    if d_col is None:
        return False
    w_col = _column(names, _WEIGHT_COLUMNS, path)
    weights = np.asarray(raw[w_col], dtype=float) if w_col else np.ones(len(raw))
    new_d, new_w = recalibrate_distance(raw[d_col], weights, calibration, prior_power)
    columns = {n: np.asarray(raw[n]) for n in names}
    columns[d_col] = new_d
    columns[w_col or "weights"] = new_w
    target.mkdir(parents=True)
    np.savetxt(
        target / POSTERIOR_FILE,
        np.column_stack([columns[n] for n in columns]),
        header="\t".join(columns),
        delimiter="\t",
        comments="",
        fmt="%.10g",
    )
    return True


def recalibrate_directory(
    working_dir: str | Path,
    out_dir: str | Path,
    run: str,
    calibration: DistanceRecalibration,
    prior_power: float = 2.0,
    events: Sequence[str] | None = None,
) -> list[str]:
    """Write recalibrated copies of the posterior files, in the same layout.

    The output is ``<out_dir>/<event>/<run>/output/posterior_samples.dat``
    with the columns of the input; only the distance and the weights change.
    A ``recalibration.json`` records the parameters. The input is not
    modified, and ``out_dir`` must be empty or absent.

    Returns
    -------
    list of str
        The names of the events written.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    written = []
    for path in sorted(Path(working_dir).glob(f"*/{run}/output/{POSTERIOR_FILE}")):
        name = path.parts[-4]
        if (events is not None and name not in events) or path.stat().st_size == 0:
            continue
        if _write_recalibrated(
            path, out_dir / name / run / "output", calibration, prior_power
        ):
            written.append(name)
    record = {
        **asdict(calibration),
        "prior_power": prior_power,
        "run": run,
        "events": len(written),
        "parameter": (
            "luminosity_distance (ln D about the weighted median; "
            "weights multiplied by the prior ratio)"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recalibration.json").write_text(json.dumps(record, indent=2) + "\n")
    return sorted(written)


def release_truths(release_dir: str | Path) -> dict[str, float]:
    """Return the true distance (Mpc) of each detectable event of a release."""
    with h5py.File(Path(release_dir) / "injections.h5", "r") as fh:
        grp = fh["injections"]
        detectable = np.asarray(grp["detectable"], dtype=bool)  # pyright: ignore[reportIndexIssue]
        names = np.asarray(grp["event_name"])[detectable]  # pyright: ignore[reportIndexIssue]
        distance = np.asarray(grp["luminosity_distance"], dtype=float)[detectable]  # pyright: ignore[reportIndexIssue]
    return {
        (n.decode() if isinstance(n, bytes) else str(n)): float(d)
        for n, d in zip(names, distance, strict=True)
    }


def blueprint_truths(path: str | Path) -> dict[str, float]:
    """Return the true distance (Mpc) of each event of a blueprints file.

    The blueprints of a run's detectable events are the ``events.yaml`` of its
    release, or ``blueprints.yaml`` in the detection output directory. Events
    with no skymap (no ``localization file``) are skipped: SimplePE is not run
    on them, so they are not events a posterior is awaited for.
    """
    truths = {}
    for doc in yaml.safe_load_all(Path(path).read_text()):
        if doc and "localization file" in doc:
            params = doc["injection"]["parameters"]
            truths[str(doc["name"])] = float(params["luminosity_distance"])
    return truths


def calibration_report(
    events: Sequence[tuple[np.ndarray, np.ndarray, float]],
    calibration: DistanceRecalibration,
    prior_power: float = 2.0,
) -> dict[str, float]:
    """Summarise the calibration of the distance posteriors after a recalibration.

    Returns the number of events, the fraction whose true distance lies in the
    central 90 per cent interval, and the mean and standard deviation of the
    probit of its credible level (0 and 1 for a calibrated posterior).
    """
    z = []
    for distance, weights, truth in events:
        new_d, new_w = recalibrate_distance(distance, weights, calibration, prior_power)
        level = credible_level(np.log(new_d), new_w, float(np.log(truth)))
        z.append(float(ndtri(np.clip(level, 1.0e-3, 1.0 - 1.0e-3))))
    z_arr = np.asarray(z)
    return {
        "events": len(z),
        "coverage_90": float(np.mean(np.abs(z_arr) < ndtri(0.95))),
        "mean_z": float(z_arr.mean()),
        "std_z": float(z_arr.std()),
    }


#: Directories of the two peak-finder branches of a SimplePE analysis, and the
#: short names under which a branch posterior is reported.
_BRANCHES = {"_particle_swarm_plus_metric": "pso", "_scipy_plus_metric": "scipy"}


def split_validation(
    events: Sequence[tuple[np.ndarray, np.ndarray, float]],
    prior_power: float = 2.0,
    n_splits: int = 40,
    seed: int = 0,
    **grid: np.ndarray,
) -> dict[str, float]:
    """Fit on a random half of the events and test on the other, repeatedly.

    Returns the mean over the splits of the central-90 per cent coverage and of
    the mean and standard deviation of the probit of the credible level on the
    events not used in the fit, and the mean and scatter of the fitted shift
    and width factor.
    """
    rng = np.random.default_rng(seed)
    n = len(events)
    fits, tests = [], []
    for _ in range(n_splits):
        order = rng.permutation(n)
        train = [events[i] for i in order[: n // 2]]
        test = [events[i] for i in order[n // 2 :]]
        cal = fit_distance_recalibration(train, prior_power, **grid)
        report = calibration_report(test, cal, prior_power)
        fits.append((cal.shift, cal.scale))
        tests.append((report["coverage_90"], report["mean_z"], report["std_z"]))
    f, t = np.asarray(fits), np.asarray(tests)
    return {
        "n_splits": n_splits,
        "shift_mean": float(f[:, 0].mean()),
        "shift_std": float(f[:, 0].std()),
        "scale_mean": float(f[:, 1].mean()),
        "scale_std": float(f[:, 1].std()),
        "coverage_90": float(t[:, 0].mean()),
        "mean_z": float(t[:, 1].mean()),
        "std_z": float(t[:, 2].mean()),
    }


def _usable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    _, names = _read(path)
    return _column(names, _DISTANCE_COLUMNS, path) is not None


def recalibrate_catalogue(
    working_dir: str | Path,
    out_dir: str | Path,
    runs: Sequence[str],
    truths: dict[str, float],
    prior_power: float = 2.0,
    n_splits: int = 40,
) -> dict[str, Any]:
    """Recalibrate a catalogue's posteriors, with a separate fit for each run.

    Each event is assigned to the first run in ``runs`` which has a combined
    posterior for it (so the order is a priority), and the shift and width
    factor are fitted separately for the events of each run, because the runs
    differ in how well calibrated they are. The corrected files are written in
    the layout of :func:`recalibrate_directory`, together with
    ``recalibration.json`` (the fits, their calibration before and after, and a
    split validation) and ``missing_events.txt``, which lists the events of
    ``truths`` with no combined posterior in any run and which single-branch
    posteriors they have.

    Returns
    -------
    dict
        The contents of ``recalibration.json``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    working_dir = Path(working_dir)
    assigned: dict[str, list[str]] = {run: [] for run in runs}
    missing: dict[str, str] = {}
    for name in sorted(truths):
        for run in runs:
            if _usable(working_dir / name / run / "output" / POSTERIOR_FILE):
                assigned[run].append(name)
                break
        else:
            found = [
                f"{short}({run})"
                for run in runs
                for branch, short in _BRANCHES.items()
                if _usable(
                    working_dir / name / run / branch / "output" / POSTERIOR_FILE
                )
            ]
            missing[name] = ",".join(found) if found else "none"
    groups: dict[str, Any] = {}
    for run in runs:
        names = assigned[run]
        if not names:
            continue
        events = load_events(working_dir, run, {n: truths[n] for n in names})
        calibration = fit_distance_recalibration(events, prior_power)
        group: dict[str, Any] = {
            **asdict(calibration),
            "events": len(events),
            "before": calibration_report(events, DistanceRecalibration(), prior_power),
            "after": calibration_report(events, calibration, prior_power),
        }
        if len(events) >= 8:
            group["validation"] = split_validation(
                events, prior_power, n_splits=n_splits
            )
        groups[run] = group
        for name in names:
            _write_recalibrated(
                working_dir / name / run / "output" / POSTERIOR_FILE,
                out_dir / name / run / "output",
                calibration,
                prior_power,
            )
    record = {
        "prior_power": prior_power,
        "runs": list(runs),
        "events": sum(len(v) for v in assigned.values()),
        "missing": len(missing),
        "groups": groups,
        "parameter": (
            "luminosity_distance (ln D about the weighted median; "
            "weights multiplied by the prior ratio)"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recalibration.json").write_text(json.dumps(record, indent=2) + "\n")
    lines = ["# event\tsingle-branch posteriors (none if no posterior at all)"]
    lines += [f"{name}\t{found}" for name, found in missing.items()]
    (out_dir / "missing_events.txt").write_text("\n".join(lines) + "\n")
    return record
