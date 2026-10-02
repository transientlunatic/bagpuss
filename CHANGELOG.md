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
- The detection stage imports `minke.duty_cycle`, which is in minke releases
  after 2.2.1 (not 2.2.1 itself); the `detection` extra requires `minke>2.2.1`.
