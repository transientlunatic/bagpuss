r"""Population inference on a bagpuss catalogue with ``gwpopulation``.

This is the check that a catalogue can be used for what it is for: fit the
mass and redshift distributions of the detectable events, correct for the
selection effect with the sensitivity injections of :mod:`bagpuss.sensitivity`,
and see whether the input population is recovered.

The fit is in the variables :math:`(m_1, q, z)`, so the spins, orientation, sky
position and merger time are marginalised. Those are independent of the masses
and redshift in both the injection distribution and the true population, so
the marginalisation is exact: :func:`selection_data` divides them out of the
stored draw density (see
:meth:`~bagpuss.sensitivity.InjectionDistribution.nuisance_log_prob`) and
changes the measure from :math:`dm_1\,dm_2` to :math:`dm_1\,dq`.

The models are ``gwpopulation``'s ``SinglePeakSmoothedMassDistribution`` and
``PowerLawRedshift``, whose hyperparameter names bagpuss follows, so any
difference between the documented and implemented population shows up as a
bias here. ``gwpopulation`` and ``bilby`` are imported lazily; install the
``population`` extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
from astropy.cosmology import FLRW

from bagpuss.population import PowerLawPlusPeakMassDistribution
from bagpuss.sensitivity import InjectionDistribution, SensitivityInjectionSet

if TYPE_CHECKING:
    import pandas as pd

    from bagpuss.mdc.config import MDCConfig

__all__: list[str] = [
    "DEFAULT_FREE",
    "ReleaseInputs",
    "ParameterSummary",
    "event_samples",
    "fit_population",
    "load_release",
    "load_simplepe_posteriors",
    "simplepe_samples",
    "selection_data",
    "summarise",
    "unsupported_events",
    "true_hyperparameters",
]

#: The hyperparameters fitted by default; the rest are held at their true values.
DEFAULT_FREE: tuple[str, ...] = ("alpha", "beta", "lam", "mpp", "sigpp", "lamb")

# Prior ranges for the fitted hyperparameters (all broad uniform).
_PRIOR_RANGES: dict[str, tuple[float, float]] = {
    "alpha": (-4.0, 12.0),
    "beta": (-4.0, 12.0),
    "lam": (0.0, 1.0),
    "mpp": (20.0, 50.0),
    "sigpp": (1.0, 10.0),
    "delta_m": (0.0, 10.0),
    "mmin": (2.0, 10.0),
    "mmax": (50.0, 100.0),
    "lamb": (-6.0, 6.0),
}


def _attr_float(attrs: Any, name: str) -> float:  # noqa: ANN401
    """Return a scalar HDF5 attribute as a float (h5py types its values loosely)."""
    return float(attrs[name])


def true_hyperparameters(
    mass: PowerLawPlusPeakMassDistribution, lamb: float = 0.0
) -> dict[str, float]:
    """Return the ``gwpopulation`` hyperparameters of a bagpuss population.

    Parameters
    ----------
    mass : PowerLawPlusPeakMassDistribution
        The population's mass model.
    lamb : float, optional
        Redshift-evolution index of the merger rate density in the
        ``gwpopulation`` convention; ``0`` is constant in the source frame.
    """
    return {
        "alpha": float(mass.alpha),
        "beta": float(mass.beta_q),
        "mmin": float(mass.m_min),
        "mmax": float(mass.m_max),
        "lam": float(mass.lambda_peak),
        "mpp": float(mass.mu_m),
        "sigpp": float(mass.sigma_m),
        "delta_m": float(mass.delta_m),
        "lamb": float(lamb),
    }


def selection_data(
    injections: SensitivityInjectionSet,
    snr: np.ndarray,
    snr_threshold: float,
    distribution: InjectionDistribution,
    analysis_time: float = 1.0,
) -> dict[str, Any]:
    r"""Return the found injections in the form ``gwpopulation`` expects.

    ``prior`` is the density the injections were drawn from with respect to
    :math:`dm_1\,dq\,dz`: the stored ``log_p_draw`` less the nuisance terms,
    plus :math:`\ln m_1` (since :math:`dm_2 = m_1\,dq`). With
    ``total_generated`` it lets ``gwpopulation.vt.ResamplingVT`` reproduce
    :meth:`~bagpuss.sensitivity.SensitivityInjectionSet.detection_probability`
    for a population which is isotropic and uniform in the marginalised
    parameters.

    Parameters
    ----------
    injections : SensitivityInjectionSet
        The sensitivity injections (all those held, detected or not).
    snr : numpy.ndarray
        Network SNR of each held injection.
    snr_threshold : float
        An injection is found if ``snr >= snr_threshold``.
    distribution : InjectionDistribution
        The distribution the injections were drawn from, which supplies the
        nuisance terms. It must have the spin and time ranges of the draw.
    analysis_time : float, optional
        Observing time in years (only used by ``gwpopulation`` to turn a rate
        into an expected number of events).

    Returns
    -------
    dict
        ``mass_1``, ``mass_ratio``, ``redshift``, ``prior`` (arrays) and
        ``total_generated``, ``analysis_time``.

    Raises
    ------
    ValueError
        If *snr* does not have one entry per injection.
    """
    snr = np.asarray(snr, dtype=float)
    if snr.shape != (len(injections),):
        raise ValueError(f"snr must have shape ({len(injections)},), got {snr.shape}")
    found = snr >= snr_threshold
    params = {name: arr[found] for name, arr in injections.as_dict().items()}
    m1 = params["m1_source"]
    log_prior = (
        injections.log_p_draw[found]
        - distribution.nuisance_log_prob(params)
        + np.log(m1)
    )
    return {
        "mass_1": m1,
        "mass_ratio": params["m2_source"] / m1,
        "redshift": params["redshift"],
        "prior": np.exp(log_prior),
        "total_generated": int(injections.n_total),
        "analysis_time": float(analysis_time),
    }


def event_samples(
    m1_source: np.ndarray, m2_source: np.ndarray, redshift: np.ndarray
) -> list[pd.DataFrame]:
    """Return one-sample "posteriors" at the true parameters of each event.

    With a single sample per event at a constant prior, the hierarchical
    likelihood is the product of the population densities at the true values,
    which is the best an analysis can do and so isolates the population model
    and the selection function from the parameter estimation.
    """
    import pandas as pd

    m1 = np.asarray(m1_source, dtype=float)
    q = np.asarray(m2_source, dtype=float) / m1
    z = np.asarray(redshift, dtype=float)
    return [
        pd.DataFrame(
            {
                "mass_1": [m1[i]],
                "mass_ratio": [q[i]],
                "redshift": [z[i]],
                "prior": [1.0],
            }
        )
        for i in range(len(m1))
    ]


def _models(
    z_max: float, cosmology: str, mass_grid: tuple[int, int] = (1000, 500)
) -> Any:  # noqa: ANN401
    from bilby.hyper.model import Model
    from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
    from gwpopulation.models.redshift import PowerLawRedshift

    return Model(
        [
            SinglePeakSmoothedMassDistribution(normalization_shape=mass_grid),
            PowerLawRedshift(z_max=z_max, cosmo_model=cosmology),
        ]
    )


def build_likelihood(
    posteriors: Sequence[pd.DataFrame],
    selection: Mapping[str, Any],
    z_max: float,
    cosmology: str,
    mass_grid: tuple[int, int] = (1000, 500),
) -> Any:  # noqa: ANN401
    """Return the ``gwpopulation`` hierarchical likelihood with selection effects.

    Parameters
    ----------
    posteriors : sequence of pandas.DataFrame
        Per-event samples with ``mass_1``, ``mass_ratio``, ``redshift`` and
        ``prior`` columns.
    selection : mapping
        As returned by :func:`selection_data`.
    z_max : float
        Maximum redshift of the redshift model (that of the draw).
    cosmology : str
        Cosmology name understood by ``wcosmo``, e.g. ``"Planck18"``.
    mass_grid : tuple of int, optional
        ``(n_m1, n_q)`` of the grid the mass model is normalised on. The cost
        of a likelihood evaluation scales with it; coarser grids are faster
        but less accurate near ``mmin``.
    """
    from gwpopulation.hyperpe import HyperparameterLikelihood
    from gwpopulation.vt import ResamplingVT

    vt = ResamplingVT(
        model=_models(z_max, cosmology, mass_grid),
        data=dict(selection),
        n_events=len(posteriors),
    )
    return HyperparameterLikelihood(
        list(posteriors),
        _models(z_max, cosmology, mass_grid),
        selection_function=vt,
    )


@dataclass(frozen=True)
class ParameterSummary:
    """How one hyperparameter's posterior compares with its true value.

    Attributes
    ----------
    name : str
        Hyperparameter name.
    truth : float
        The true value.
    median, lower, upper : float
        Posterior median and the equal-tailed credible interval.
    quantile : float
        Fraction of the posterior below the truth; values near 0 or 1 mean
        the truth is in a tail.
    z_score : float
        ``(truth - mean) / std`` of the posterior.
    inside : bool
        Whether the truth lies within ``[lower, upper]``.
    """

    name: str
    truth: float
    median: float
    lower: float
    upper: float
    quantile: float
    z_score: float
    inside: bool


def summarise(
    samples: Mapping[str, np.ndarray],
    truth: Mapping[str, float],
    credible_level: float = 0.9,
) -> list[ParameterSummary]:
    """Compare posterior *samples* with *truth* for each fitted hyperparameter.

    Parameters
    ----------
    samples : mapping
        Posterior samples keyed by hyperparameter name.
    truth : mapping
        True values; only the names present in *samples* are reported.
    credible_level : float, optional
        The central credible interval to test the truth against.
    """
    tail = 0.5 * (1.0 - credible_level)
    rows = []
    for name, value in truth.items():
        if name not in samples:
            continue
        x = np.asarray(samples[name], dtype=float)
        lo, med, hi = np.quantile(x, [tail, 0.5, 1.0 - tail])
        std = float(x.std())
        rows.append(
            ParameterSummary(
                name=name,
                truth=float(value),
                median=float(med),
                lower=float(lo),
                upper=float(hi),
                quantile=float(np.mean(x < value)),
                z_score=float((value - x.mean()) / std) if std > 0 else float("nan"),
                inside=bool(lo <= value <= hi),
            )
        )
    return rows


def unsupported_events(
    inputs: ReleaseInputs, mass_grid: tuple[int, int] = (1000, 500)
) -> list[str]:
    """Return the events with no posterior sample inside the true population.

    The hierarchical likelihood of such an event is exactly zero at the true
    hyperparameters (and, for a hard ``m_min``, over most of the prior), so
    one of them makes every fit return a likelihood of zero. It means the
    event's parameter estimation excluded the truth, which is a failure of the
    parameter estimation and not a feature of the population.

    Parameters
    ----------
    inputs : ReleaseInputs
        As from :func:`load_release`.
    mass_grid : tuple of int, optional
        See :func:`build_likelihood`.
    """
    # gwpopulation caches by parameters, not data, so a model reused over
    # events with different samples returns stale values: build one per event.
    out = []
    for name, post in zip(inputs.event_names, inputs.posteriors, strict=True):
        model = _models(inputs.z_max, inputs.cosmology, mass_grid)
        data = {k: post[k].to_numpy() for k in ("mass_1", "mass_ratio", "redshift")}
        if not np.any(np.asarray(model.prob(data, **inputs.truth)) > 0.0):
            out.append(name)
    return out


def fit_population(
    posteriors: Sequence[pd.DataFrame],
    selection: Mapping[str, Any],
    truth: Mapping[str, float],
    *,
    z_max: float,
    cosmology: str = "Planck18",
    mass_grid: tuple[int, int] = (1000, 500),
    free: Sequence[str] = DEFAULT_FREE,
    sampler: str = "dynesty",
    outdir: str = "population_fit",
    label: str = "fit",
    seed: int = 1,
    **sampler_kwargs: Any,  # noqa: ANN401
) -> tuple[pd.DataFrame, list[ParameterSummary]]:
    """Fit the mass and redshift hyperparameters and compare with *truth*.

    Hyperparameters not in *free* are held at their value in *truth*; the free
    ones get broad uniform priors.

    Parameters
    ----------
    posteriors, selection, z_max, cosmology, mass_grid
        As for :func:`build_likelihood`.
    truth : mapping
        True hyperparameters (see :func:`true_hyperparameters`).
    free : sequence of str, optional
        Hyperparameters to fit.
    sampler : str, optional
        Any sampler ``bilby`` knows (default ``"dynesty"``).
    outdir, label : str, optional
        Where ``bilby`` writes its output.
    seed : int, optional
        Sampler seed.
    **sampler_kwargs
        Passed to ``bilby.run_sampler`` (e.g. ``nlive``).

    Returns
    -------
    pandas.DataFrame, list of ParameterSummary
        The posterior samples and their comparison with *truth*.

    Raises
    ------
    ValueError
        If a free parameter has no prior range or is missing from *truth*.
    """
    import bilby
    from bilby.core.prior import Uniform

    for name in free:
        if name not in _PRIOR_RANGES:
            raise ValueError(f"no prior range for {name!r}")
    priors = bilby.core.prior.PriorDict()
    for name, value in truth.items():
        if name in free:
            priors[name] = Uniform(*_PRIOR_RANGES[name], name=name)
        else:
            priors[name] = float(value)
    missing = [name for name in free if name not in truth]
    if missing:
        raise ValueError(f"truth is missing {missing}")

    likelihood = build_likelihood(posteriors, selection, z_max, cosmology, mass_grid)
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=sampler,
        outdir=outdir,
        label=label,
        seed=seed,
        save="hdf5",  # pyright: ignore[reportArgumentType]
        clean=True,
        **sampler_kwargs,
    )
    if result is None:
        raise RuntimeError("bilby returned no result")
    samples = result.posterior
    return samples, summarise({n: samples[n].to_numpy() for n in free}, truth)


@dataclass(frozen=True)
class ReleaseInputs:
    """Everything a population fit needs from a data release.

    Attributes
    ----------
    posteriors : list of pandas.DataFrame
        One-sample "posteriors" at the true parameters of the detectable events.
    selection : dict
        As returned by :func:`selection_data`.
    truth : dict
        The true hyperparameters.
    z_max : float
        The population's maximum redshift.
    cosmology : str
        Name of the release's cosmology.
    """

    posteriors: list[pd.DataFrame]
    selection: dict[str, Any]
    truth: dict[str, float]
    z_max: float
    cosmology: str
    event_names: list[str] = field(default_factory=list)
    missing_events: list[str] = field(default_factory=list)


def simplepe_samples(
    path: str | Path,
    cosmology: FLRW,
    n_samples: int | None = 600,
    seed: int = 0,
) -> pd.DataFrame:
    r"""Read one event's ``SimplePE`` posterior as ``gwpopulation`` samples.

    The file is ``posterior_samples.dat``: detector-frame ``chirp_mass``,
    ``symmetric_mass_ratio`` and ``distance`` with an importance ``weight``.
    ``SimplePE`` with its defaults (reweighting to uniform detector-frame
    component masses and isotropic spins, and ``--distance_prior
    UniformVolume``) has the prior

    .. math:: \pi(m_1^d, m_2^d, d_L) \propto d_L^2 .

    In :math:`(m_1, q, z)` of the source frame this is :math:`\pi \propto
    d_L^2\,(1+z)^2\,m_1\,\mathrm{d}d_L/\mathrm{d}z`, the Jacobian of
    :math:`m^d = (1+z)m` and :math:`d_L(z)` in the (fixed) cosmology. The
    weights are folded into the ``prior`` column, so the hierarchical
    likelihood's mean of :math:`p/\pi` over the samples is the
    weight-normalised sum, as for a weighted posterior.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``posterior_samples.dat`` written by ``simple_pe_analysis``.
    cosmology : astropy.cosmology.FLRW
        The cosmology which relates distance and redshift.
    n_samples : int or None, optional
        Resample the weighted posterior to this many equally weighted samples
        (with replacement, so ``prior`` is then just :math:`\pi`). Events
        need equal numbers of samples for ``gwpopulation``, and the cost of a
        likelihood evaluation is proportional to the total. ``None`` keeps
        the weighted samples.
    seed : int, optional
        Seed of the resampling.

    Returns
    -------
    pandas.DataFrame
        ``mass_1``, ``mass_ratio``, ``redshift`` and ``prior``.
    """
    import pandas as pd

    data = np.genfromtxt(path, names=True)
    chirp = np.asarray(data["chirp_mass"], dtype=float)
    eta = np.asarray(data["symmetric_mass_ratio"], dtype=float)
    d_l = np.asarray(data["distance"], dtype=float)
    weight = np.asarray(data["weight"], dtype=float)
    weight = weight / weight.mean()

    z_grid = np.linspace(0.0, 6.0, 6001)
    d_grid = np.asarray(cosmology.luminosity_distance(z_grid).to("Mpc").value)  # pyright: ignore[reportAttributeAccessIssue]
    z = np.interp(d_l, d_grid, z_grid)
    d_dz = np.interp(z, z_grid, np.gradient(d_grid, z_grid))

    total = chirp * eta ** (-0.6)
    root = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
    m1_det = 0.5 * total * (1.0 + root)
    m2_det = 0.5 * total * (1.0 - root)
    m1 = m1_det / (1.0 + z)
    prior = d_l**2 * (1.0 + z) ** 2 * m1 * d_dz / 1.0e9
    frame = pd.DataFrame(
        {
            "mass_1": m1,
            "mass_ratio": m2_det / m1_det,
            "redshift": z,
            "prior": prior,
        }
    )
    if n_samples is None:
        frame["prior"] = frame["prior"] / weight
        return frame
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(frame), size=n_samples, p=weight / weight.sum())
    return frame.iloc[keep].reset_index(drop=True)


def load_simplepe_posteriors(
    release_dir: str | Path,
    working_dir: str | Path,
    cosmology: FLRW,
    n_samples: int | None = 600,
) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    """Read the ``SimplePE`` posteriors of a release's detectable events.

    Parameters
    ----------
    release_dir : str or pathlib.Path
        The release; its ``injections.h5`` names the detectable events.
    working_dir : str or pathlib.Path
        The asimov project's ``working`` directory, holding
        ``<event>/simplepe-xhm/output/posterior_samples.dat``.
    cosmology : astropy.cosmology.FLRW
        See :func:`simplepe_samples`.
    n_samples : int or None, optional
        Samples per event (see :func:`simplepe_samples`).

    Returns
    -------
    list of pandas.DataFrame, list of str, list of str
        The posteriors of the events which have one, their names, and the names
        of the events which do not (no skymap, or the analysis failed).
    """
    with h5py.File(Path(release_dir) / "injections.h5", "r") as fh:
        grp = cast(h5py.Group, fh["injections"])
        detectable = np.asarray(grp["detectable"], dtype=bool)
        names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in np.asarray(grp["event_name"])[detectable]
        ]
    found: list[pd.DataFrame] = []
    found_names: list[str] = []
    missing: list[str] = []
    for index, name in enumerate(names):
        path = Path(working_dir) / name / "simplepe-xhm" / "output"
        path = path / "posterior_samples.dat"
        if path.exists():
            found.append(simplepe_samples(path, cosmology, n_samples, seed=index))
            found_names.append(name)
        else:
            missing.append(name)
    return found, found_names, missing


def load_release(
    release_dir: str | Path,
    config: MDCConfig | None = None,
    simplepe_dir: str | Path | None = None,
) -> ReleaseInputs:
    """Read a data release into the inputs of :func:`fit_population`.

    The events are the detectable rows of ``injections.h5`` at their true
    parameters, the selection function is ``sensitivity.h5`` thresholded at
    the SNR recorded there, and the truth is the population in the release's
    ``config.yaml``. The sensitivity injections' spin and orientation
    densities are those of the default
    :class:`~bagpuss.sensitivity.InjectionDistribution`, which equal the
    population's as long as the spin magnitudes extend to 1.

    Parameters
    ----------
    release_dir : str or pathlib.Path
        A directory written by ``bagpuss mdc export-release``.
    config : MDCConfig or None, optional
        The run configuration, if not the release's own ``config.yaml``.
    simplepe_dir : str or pathlib.Path or None, optional
        An asimov project's ``working`` directory. If given, the events are the
        ``SimplePE`` posteriors found there (see
        :func:`load_simplepe_posteriors`) rather than the true parameters.

    Raises
    ------
    ValueError
        If the release has no sensitivity injections, or the population's
        spins differ from those the injections are drawn with.
    """
    from bagpuss.mdc.config import load_config
    from bagpuss.mdc.pipeline import _load_cosmology, build_population

    root = Path(release_dir)
    if config is None:
        config = load_config(root / "config.yaml")
    if not (root / "sensitivity.h5").exists():
        raise ValueError(f"{root} has no sensitivity.h5")
    if config.spin_a_max != 1.0:
        raise ValueError(
            "the sensitivity injections draw spin magnitudes up to 1, but the "
            f"population has spin_a_max={config.spin_a_max}; the marginalisation "
            "over spins would be wrong"
        )

    injections, snr = SensitivityInjectionSet.from_hdf5(
        root / "sensitivity.h5", with_snr=True
    )
    with h5py.File(root / "sensitivity.h5", "r") as fh:
        attrs = cast(h5py.Group, fh["injections"]).attrs
        threshold = _attr_float(attrs, "snr_threshold")
        distribution = InjectionDistribution(
            cosmology=_load_cosmology(str(attrs["cosmology"])),
            t_start=_attr_float(attrs, "t_start"),
            t_end=_attr_float(attrs, "t_end"),
            **{
                name: _attr_float(attrs, name)
                for name in ("z_max", "m_min", "m_max", "alpha", "beta_q", "kappa")
            },
        )
    years = (distribution.t_end - distribution.t_start) / (365.25 * 86400.0)
    selection = selection_data(injections, snr, threshold, distribution, years)

    with h5py.File(root / "injections.h5", "r") as fh:
        grp = cast(h5py.Group, fh["injections"])
        detectable = np.asarray(grp["detectable"], dtype=bool)
        detectable_mask = detectable
        events = {
            name: np.asarray(grp[name])[detectable]
            for name in ("m1_source", "m2_source", "redshift")
        }

    mass = build_population(config).mass
    assert isinstance(mass, PowerLawPlusPeakMassDistribution)
    missing: list[str] = []
    if simplepe_dir is None:
        posteriors = event_samples(**events)
        with h5py.File(root / "injections.h5", "r") as fh:
            grp = cast(h5py.Group, fh["injections"])
            names = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in np.asarray(grp["event_name"])[detectable_mask]
            ]
    else:
        posteriors, names, missing = load_simplepe_posteriors(
            root, simplepe_dir, _load_cosmology(config.cosmology)
        )
    return ReleaseInputs(
        posteriors=posteriors,
        event_names=names,
        missing_events=missing,
        selection=selection,
        truth=true_hyperparameters(mass),
        z_max=float(config.z_max),
        cosmology=config.cosmology,
    )
