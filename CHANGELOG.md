# Changelog

## [Unreleased]

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

### Notes
- The detection stage imports `minke.duty_cycle`, which is not in a released
  minke yet (minke PR #25).
