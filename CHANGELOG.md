# Changelog

## 0.3.0

Injection sets, and therefore MDC releases, produced by 0.2.0 are not
reproduced by this version for the same seed: the primary-mass distribution,
the detector-frame masses and the host assignment have all changed.

### Added
- `host_luminosity_weight` (`MDCConfig`, `galaxies.host_luminosity_weight` in
  the YAML, and the injection builders): events occur in galaxies with
  probability proportional to `L**p`. The default is `1`; `0` recovers number
  weighting. The probability that an event's host is catalogued is the
  `L**p`-weighted completeness.
- `weighted_cdf` on luminosity models, and a `weight_power` argument to
  `MagnitudeLimitedSurvey.completeness`.
- `completeness.dat` from the GLADE-style export gains a host-weighted column,
  `completeness_host`; the exported README, release README and docs describe
  the weighting.

### Changed
- **Breaking:** `sample_host_galaxies` takes `rate` (a `MergerRate`) in place
  of the `redshift` override, and no longer has `n_neighbours`. A catalogued
  host is drawn from the whole catalogue with weight `L**p * R(z)/(1+z)`,
  which gives the rate-weighted redshift distribution exactly, instead of
  from the 200 catalogue galaxies nearest the trial redshift (which widened
  to dz ~ 0.1 where the catalogue is sparse).
- Hosted events no longer take a uniformly random catalogue row, whose
  redshifts followed the catalogue `n(z)` rather than the rate-weighted
  distribution.
- The `detection` extra requires minke 2.3.0, which contains the duty-cycle
  support, the corrected O4 PSDs and the inclination fix; the workaround that
  renamed `iota` to `inclination` is removed.

### Fixed
- The power-law-plus-peak primary-mass distribution mixed an unnormalised power
  law with an unnormalised Gaussian of unit peak height, so `lambda_peak` was
  not the peak fraction: with the default `lambda_peak = 0.03` about 97% of
  events were in the peak. Both components are now normalised over
  `[m_min, m_max]` before mixing.
- Injected source-frame masses are redshifted to the detector frame before
  they are handed to minke, which injects them as waveform masses at the true
  luminosity distance. Previously the chirp mass was low by `1 + z`.
- Lint, format and type checks (`ruff`, `mypy`, `pyright`) pass again.

## 0.2.0

### Added
- `bagpuss.mdc`: config-driven, sharded generation of large catalogues and
  injection sets (sky-tiled galaxy catalogue, time-sliced injections, zarr
  store, HTCondor DAG rendering) -- see `docs/mdc.rst`.
- `bagpuss mdc detect-injections` / `assemble-detections`: network SNRs for
  every injection (per-detector duty cycles, via minke), optional frames,
  asimov event blueprints for the detectable events, and a BAYESTAR skymap
  per event (`write_skymaps`). Needs the `detection` extra.
- `estimation.trigger_eta_max`: cap the simple-pe trigger's symmetric mass
  ratio, which fails at the equal-mass limit; the injected truth is kept as
  `trigger truth`.
- `bagpuss mdc export-release` / `verify-release`: assemble the sharded store
  into flat, checksummed release files (HDF5 catalogue and injection tables,
  events, skymaps, README, Zenodo metadata).
- GLADE+-style catalogue export (`bagpuss mdc export-glade`).
- Merger-rate models (`MergerRate`, `ConstantMergerRate`) with
  `expected_n_mergers` / `sample_merger_redshifts`, and rate-based injection
  counts in `build_injection_set`.
- Observed-galaxy counting and sampling (`expected_n_observed`,
  `sample_observed_redshifts`), and `sample_brighter_than`,
  `differential_comoving_volume` and `z_max` on luminosity models and universes.

### Changed
- **Breaking:** `create_injection_set` now takes `universe`, `catalogue`,
  `selection`, `population` and `n_draw` (previously `catalogue`, `population`,
  `cosmology`, `n_draw`). Host galaxies are assigned in two steps: a trial host
  from the universe, then a catalogue row if the survey completeness says the
  host would be catalogued (`host_galaxy_index == -1` otherwise); it is no
  longer drawn uniformly from the catalogue. `docs/injection.rst` is updated.

### Fixed
- `sample_host_galaxies` no longer fails on an empty catalogue.
- `MDCConfig.trigger_eta_max` is validated to lie in (0, 0.25].
- `export-release` / `verify-release`: the catalogue is written tile by tile
  (no full in-memory concatenation); `events.yaml` is checked against the
  detectable events; a damaged skymaps tar or HDF5 file is reported rather
  than raised; the README resolves the default observation window.
- BAYESTAR skymaps use the configured low-frequency cutoff throughout.

### Notes
- The detection stage needs minke 2.3.0 or later (`minke.duty_cycle` and the
  corrected O4 PSDs); the `detection` extra requires it.
