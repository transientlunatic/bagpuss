Mock Data Challenge (MDC) generation
=====================================

The ``bagpuss.mdc`` subpackage is generic, config-driven tooling for
producing bagpuss catalogues and injection sets that are too large to build
in a single process — for example a full-sky run out to :math:`z_{\max}=3`,
where the observable galaxy population and the BBH injection set can each
run to hundreds of millions of rows.

Nothing in this subpackage hardcodes a specific run's physical parameters.
A run is entirely described by a YAML :class:`~bagpuss.mdc.config.MDCConfig`
file; the sharding, condor tooling, and validation logic here are reusable
across any bagpuss MDC, not just the z=3 run that motivated it. The actual
per-run config files live in the downstream project consuming bagpuss (e.g.
``faketc/mdc/z3/config.yaml``), not in this repository.

Why sharding is needed
-----------------------

Volume scales with comoving volume, which grows roughly 150x between
:math:`z_{\max}=0.3` (bagpuss's small-scale demo) and :math:`z_{\max}=3.0`.
Depending on the physical parameters chosen, a full-sky z=3 run can realise
anywhere from hundreds of millions to tens of billions of galaxies — far too
many to draw with :func:`bagpuss.catalogue.build_catalogue` in one process,
and too many to trust without checking a small piece first.

``bagpuss.mdc`` addresses this by:

* splitting galaxy-catalogue generation into independent **sky tiles**
  (:class:`~bagpuss.universe.SkyPatch`) and injection generation into
  independent **draw shards**, each small enough to run as one HTCondor job;
* writing each shard to its own self-contained zarr group, so shards can be
  generated, retried, and validated independently, with no cross-job
  coordination needed during generation;
* validating with automated sanity checks *between* pipeline stages, so a
  corrupted or mis-configured shard is caught before the (much more
  expensive) downstream stages run on top of it.

Pipeline shape
---------------

.. code-block:: text

   catalogue_tile(tile_id) x n_tiles              \
                                                     > consolidate_catalogue
   catalogue_tile(tile_id) x n_tiles              /

   consolidate_catalogue
       -> injection_shard(shard_id) x n_injection_shards
           -> assemble_injections
               -> detect_shard(shard_id) x n_injection_shards
                   -> assemble_detections
                       -> package_manifest

1. **generate-tile** (:func:`~bagpuss.mdc.pipeline.generate_catalogue_tile`)
   — draws a Poisson realisation of the observable galaxy population
   restricted to one equal-area sky tile (see *Sky tiling* below), writing
   it to ``<store>/catalogue/tile_%04d``.
2. **consolidate** (:func:`~bagpuss.mdc.pipeline.consolidate_catalogue`) —
   reads every tile's row count, builds a global row-offset index, runs
   sanity checks, and calls :func:`zarr.consolidate_metadata`.
3. **generate-injections**
   (:func:`~bagpuss.mdc.pipeline.generate_injection_shard`) — Poisson-realises
   the shard's BBH event count from ``rate_density`` over the shard's slice
   of the observation window (see *Injection sharding* below), assigns hosts
   against the *consolidated* catalogue (so ``host_galaxy_index`` is a valid
   global index), applies the configured detectability filter, and writes
   the shard to ``<store>/injections/shard_%04d``.

   .. note::

      Every injection shard loads the whole consolidated catalogue into
      memory (five float64 arrays, about 0.5 GB per 12.8 million galaxies),
      so the catalogue must fit in memory in each concurrent shard job.
      That is comfortable at the z=3 scale this was built for, but it would
      not scale to hundreds of millions of galaxies; host rows would then
      have to be sampled from the tile index and only those rows read. The
      release export, by contrast, streams the catalogue tile by tile.
4. **assemble-injections**
   (:func:`~bagpuss.mdc.pipeline.assemble_injections`) — mirrors
   ``consolidate`` for injection shards.
5. **detect-injections**
   (:func:`~bagpuss.mdc.pipeline.detect_injection_shard`) — for every
   injection in one shard, works out which detectors are observing (a
   duty-cycle schedule drawn from ``master_seed``, identical in every
   shard), injects the signal with minke and computes the network SNR. The
   SNR of **every** injection, detectable or not, is written to
   ``<store>/detections/shard_%04d`` (``network_snr``, ``detectable``,
   ``active``; row-aligned with the injection shard), so the threshold can
   be changed later without recomputing. Frame files and an asimov blueprint
   file are written, under ``run.output_dir``, only for events at or above
   ``detection.snr_threshold``. This is the only stage that needs minke
   installed.
6. **assemble-detections**
   (:func:`~bagpuss.mdc.pipeline.assemble_detections`) — checks every shard
   ran and has one SNR per injection and a blueprint per detectable event,
   then merges the per-shard blueprints into ``<output_dir>/blueprints.yaml``.
7. **package** (:func:`~bagpuss.mdc.pipeline.package_manifest`) — writes the
   run's full config, bagpuss version, and summary counts into the store's
   root attrs (and a plain ``manifest.json`` alongside it), so the data is
   self-documenting for anyone who receives just the zarr store directory.

Each stage's own process exit status is the go/no-go gate between tiers:
``consolidate``/``assemble-injections``/``assemble-detections`` raise
:class:`~bagpuss.mdc.pipeline.MDCValidationError` (nonzero CLI exit) when
their checks fail, which HTCondor DAGMan already treats as a failed node —
no separate POST script is needed to halt the DAG on bad data.

Sky tiling
----------

:class:`~bagpuss.universe.SkyPatch` wraps an existing
:class:`~bagpuss.universe.Structure` model and restricts it to a tile in
``(ra, sin(dec))`` space. Because ``sin(dec)`` is uniformly distributed for
an isotropic model, tiling on evenly-spaced ``sin(dec)`` intervals gives
exactly equal-area tiles with no polar distortion, and because redshift and
sky position are independent for an isotropic structure model,
``sample_redshifts`` needs no change at all — only ``sample_positions`` and
``survey_volume`` are restricted. :func:`bagpuss.mdc.pipeline.tile_bounds`
lays tiles out on a regular ``n_ra_tiles x n_dec_tiles`` grid, with
``tile_id = ra_index * n_dec_tiles + dec_index``.

Injection sharding
------------------

Unlike catalogue tiles (split by sky area), injection shards split the
run's observation window (``t_start``/``t_end``, defaulting to bagpuss's O3
window) into ``n_injection_shards`` equal time slices —
:func:`bagpuss.mdc.pipeline.shard_time_bounds` gives shard ``shard_id``'s
``(t_start, t_end)``. Each shard's expected event count is then a Poisson
realisation of ``rate_density * survey_volume * (t_end - t_start)``
(properly time-dilation-weighted -- see
:func:`bagpuss.population.expected_n_mergers`), so there is no
caller-chosen ``n_draw`` anywhere in the pipeline: the count emerges from
the physics, exactly as a catalogue tile's galaxy count emerges from the
luminosity function rather than a chosen target.

Zarr layout
-----------

Each shard is written as an **independent zarr group**, not a slice of one
monolithic array. Poisson-realised shard sizes aren't known until after the
draw, so pre-reserving contiguous ranges in a single global array would
need cross-job coordination; independent per-shard groups need none, and a
failed shard can simply be resubmitted without touching any other shard's
data.

.. code-block:: text

   <store>/
     catalogue/
       tile_0000/          redshifts, luminosities, apparent_magnitudes, ra, dec
       tile_0001/
       ...
       .zattrs              _index: [{tile_id, n_galaxies, offset}, ...], _total_galaxies
     injections/
       shard_0000/          m1_source, m2_source, ..., host_galaxy_index
       shard_0001/
       ...
       .zattrs              _index: [{shard_id, n_injections, offset}, ...], _total_injections
     detections/
       shard_0000/          network_snr, detectable, active (row-aligned with injections/shard_0000)
       ...
       .zattrs              _index, _total_detectable, _snr_threshold
     .zattrs                 mdc_manifest: {bagpuss_version, config, n_tiles, total_galaxies, ...}
   manifest.json              (the same manifest, as a plain file)

Each shard's own ``.zattrs`` additionally records its RNG seed, tile/shard
bounds, the bagpuss version, and the full run config — enough provenance to
know exactly how that one shard was produced without consulting anything
else. :meth:`GalaxyCatalogue.to_zarr <bagpuss.catalogue.GalaxyCatalogue.to_zarr>`
/ :meth:`from_zarr <bagpuss.catalogue.GalaxyCatalogue.from_zarr>` and their
:class:`~bagpuss.injection.InjectionSet` equivalents accept either an
already-open :class:`zarr.Group` (the shard use case) or a bare store
path/URL.

.. important::

   Store access from ``bagpuss.mdc.pipeline`` always opens with
   ``use_consolidated=False``. Every lookup goes through an exact, known
   group path (``require_group(name)``, ``attrs["_index"]``) rather than
   listing, so consolidated metadata buys nothing internally — and *using*
   it while writing new groups after ``consolidate_metadata`` has already
   run is actively unsafe (a group created after consolidation is invisible
   to one call and collides with itself on the next). Consolidated metadata
   is written once, at the end, purely for external distribution/consumer
   read performance.

Reproducibility
----------------

Every run has one ``master_seed``. :func:`bagpuss.mdc.pipeline.shard_rng`
derives each shard's RNG stream from ``(master_seed, stage_tag, shard_id)``
via :class:`numpy.random.SeedSequence`, so re-running the same shard
reproduces it exactly, and no two shards — even of different kinds — ever
share a stream.

Configuration
-------------

See :class:`bagpuss.mdc.config.MDCConfig` for the full field reference. On
disk a config file is a **nested** YAML mapping whose top-level sections
mirror the pipeline's own stages -- ``galaxies``, ``population``,
``observation``, ``detection``, ``estimation``, ``sharding``, ``run`` --
rather than one flat list of ~30 keys.
:func:`~bagpuss.mdc.config.load_config` flattens this against
:data:`bagpuss.mdc.config._SCHEMA` into :class:`MDCConfig`'s fields. A
run-specific file only needs to override what differs from the defaults
(which match bagpuss's existing small-scale demo parameters):

.. code-block:: yaml

   cosmology: Planck18

   galaxies:
     z_max: 3.0
     luminosity_function:
       phi_star: 1.61e-2
     selection:
       m_lim: 19.5
     host_luminosity_weight: 1.0  # events occur in galaxies with probability ~ L^p

   population:
     rate_density: 2.4e-8  # Mpc^-3 yr^-1

   observation:
     d_max: 800.0

   detection:
     network:
       AdvancedLIGOHanford: AdvancedLIGO_O4
       AdvancedLIGOLivingston: AdvancedLIGO_O4
       AdvancedVirgo: AdvancedVirgo_O4
     duty_cycles:
       AdvancedLIGOHanford: 0.75
       AdvancedLIGOLivingston: 0.75
       AdvancedVirgo: 0.70
     duty_cycle_mean_lock_duration: 28800.0  # 8 hours
     snr_threshold: 8.0

   sharding:
     n_ra_tiles: 16
     n_dec_tiles: 8
     n_injection_shards: 200

   run:
     master_seed: 20260720
     store: /shared/home/<user>/mdc/z3/store.zarr
     output_dir: /shared/home/<user>/mdc/z3/detections  # frames + blueprints

An unrecognised key raises immediately, naming its full dotted path (e.g.
``"galaxies.z_maxx"``), to catch typos before a job is ever submitted.

The ``detection`` section (detector network, per-detector duty cycles,
injection channel, SNR threshold, frame-generation settings) is read by the
detection stage (``detect-injections``). The ``estimation`` section sets the PE hand-off written into each event
blueprint: ``chirp_mass_margin`` is the chirp-mass prior margin, and
``trigger_eta_max`` (optional) caps the symmetric mass ratio of the starting
"trigger" at fixed chirp mass, keeping the injected truth under
``trigger truth``. simple-pe fails on equal-mass triggers (``eta = 0.25``), so
~0.249 avoids that.

``duty_cycles`` (per-detector target long-run locked fraction) and
``duty_cycle_mean_lock_duration`` (mean locked-segment length in seconds,
shared across detectors) describe a two-state renewal-process duty-cycle
model -- see :func:`minke.duty_cycle.generate_duty_cycle_schedule` in the
sibling `minke <https://github.com/transientlunatic/minke>`_ project -- so
that downstream frame-generation tooling can decide which detectors are
actually active for each event (with realistic time-correlation, not an
independent per-event coin flip) rather than assuming every detector in
``network`` observes every event.

Assembling the data release
---------------------------

The zarr store is the generation archive: catalogue tiles and injection and
detection shards, as the condor jobs wrote them. ``bagpuss mdc export-release``
turns it into a handful of ordinary files for publication (e.g. on Zenodo)
without modifying the store:

.. code-block:: console

   $ bagpuss mdc export-release --config config.yaml --out-dir release/ \
         --release-version 1.0.0 --creator "Family, Given;Affiliation"
   $ bagpuss mdc verify-release release/

``catalogue.h5``
   every catalogue galaxy, one dataset per field, in tile order -- so row
   ``i`` is the galaxy ``host_galaxy_index == i`` refers to.
``injections.h5``
   every injection in one table, with ``network_snr``, ``detectable``,
   ``active`` (which detectors were observing), ``has_skymap``, ``skymap_file``
   and ``event_name``; the ``/injections`` group is also readable by
   :meth:`~bagpuss.injection.InjectionSet.from_hdf5`.
``events.yaml``, ``skymaps.tar``
   the detectable events' asimov blueprints and their BAYESTAR skymaps, with
   skymap paths made relative to the tar.
``config.yaml``, ``README.md``, ``SHA256SUMS``, ``MANIFEST.json``
   provenance, documentation and integrity (the config has the machine-specific
   store paths removed).

``zenodo_metadata.json`` holds the deposit's metadata (title, creators,
licence, keywords) for the Zenodo form or API; it is not itself part of the
upload. ``verify-release`` re-checks the checksums, row counts, that every
catalogued host's position and redshift match its catalogue row, and that
``skymaps.tar`` holds exactly the skymaps the injection table names.

Command-line usage
-------------------

One subcommand per stage, all under ``bagpuss mdc``:

.. code-block:: console

   $ bagpuss mdc generate-tile --config config.yaml --tile-id 0
   $ bagpuss mdc consolidate --config config.yaml
   $ bagpuss mdc generate-injections --config config.yaml --shard-id 0
   $ bagpuss mdc assemble-injections --config config.yaml
   $ bagpuss mdc detect-injections --config config.yaml --shard-id 0
   $ bagpuss mdc assemble-detections --config config.yaml
   $ bagpuss mdc package --config config.yaml
   $ bagpuss mdc export-glade --config config.yaml --out-dir glade_export/

``bagpuss mdc make-dag`` renders the HTCondor DAG and per-stage submit files
for a config, without submitting anything:

.. code-block:: console

   $ bagpuss mdc make-dag \
       --config config.yaml \
       --out-dir dag \
       --bagpuss-executable /scratch/wiay/<user>/bagpuss-mdc/bagpuss/.venv/bin/bagpuss

This writes ``dag/mdc.dag`` plus one ``.sub`` file per stage
(:func:`bagpuss.mdc.condor.write_dag`), ready for ``condor_submit_dag``.
Per-stage HTCondor resource requests (``request_cpus``/``request_memory``/
``request_disk``) default to :data:`bagpuss.mdc.condor.DEFAULT_RESOURCES`
and can be overridden per stage — see the calibration note below.

Recommended incremental workflow
----------------------------------

1. **Local dry run.** Use a tiny ``phi_star`` and a small tile/shard count
   (same convention as bagpuss's own test suite — see
   ``tests/test_mdc_pipeline.py``) to validate the pipeline end-to-end on a
   laptop, in seconds, before touching the cluster at all.
2. **Single calibration tile on the cluster.** Submit *one* real-``phi_star``,
   real-``z_max`` tile as a single condor job. This measures actual
   per-shard runtime, peak memory, and output size, which should then set
   ``n_ra_tiles``/``n_dec_tiles``/``n_injection_shards`` and the
   ``request_memory``/``request_disk`` overrides passed to
   :func:`~bagpuss.mdc.condor.write_dag` — not guessed numbers.
3. **Full DAG.** Only once the calibration run's numbers are in the config
   should the full DAG be rendered and submitted.

Running on ``wiay``
--------------------

``wiay``'s ``$HOME`` is NFS-mounted (shared with all execute nodes in the
condor pool), so submit files use ``should_transfer_files = NO`` — no
condor file-transfer plumbing is needed for code or config.

``/scratch/wiay`` is fast **local** disk on the ``wiay`` host itself, not a
shared pool (other execute machines each have their own
``/scratch/<hostname>``). It's the right place for anything that benefits
from fast disk I/O — in particular the Python environment itself, since a
venv is many small files and slow to build over NFS:

.. code-block:: console

   $ curl -LsSf https://astral.sh/uv/install.sh | \
       env UV_INSTALL_DIR=/scratch/wiay/<user>/uv/bin UV_NO_MODIFY_PATH=1 sh
   $ export UV_PYTHON_INSTALL_DIR=/scratch/wiay/<user>/uv/python
   $ /scratch/wiay/<user>/uv/bin/uv python install 3.12
   $ cd /scratch/wiay/<user>/bagpuss-mdc/bagpuss   # rsync or clone the repo here
   $ /scratch/wiay/<user>/uv/bin/uv sync --python 3.12

wiay's system Python is 3.11, older than bagpuss's ``>=3.12`` requirement,
which is why ``uv``'s managed Python is used rather than the system
interpreter.

Because a venv built this way lives on ``wiay``'s own local disk, it is
**not** visible to jobs that land on the other execute machines
(``andromeda``, ``balta``, ``deimos``, ``hermes``, ``puck``, ``serenity``).
Before submitting a DAG wider than a single calibration job, either pin
jobs to ``wiay`` (``requirements = Machine == "wiay.astro.gla.ac.uk"`` in
the submit files) or replicate the environment on each node's own
``/scratch/<hostname>`` — this hasn't been decided yet and should be picked
based on how much of the (shared, opportunistic) pool the run actually
needs.

GLADE+-style catalogue export
-------------------------------

:func:`bagpuss.mdc.pipeline.export_glade_catalogue` (``bagpuss mdc
export-glade``) produces a collaborator-facing galaxy catalogue product,
structured similarly to the real `GLADE+
<https://glade.elte.hu/>`_ catalogue (Dálya et al. 2022) for familiarity —
a flat, space-delimited ASCII file, one row per galaxy, plus a companion
``README.txt`` and an exact ``(z, completeness, completeness_host)`` table
(``completeness.dat``; the last column is the completeness for event hosts under
``galaxies.host_luminosity_weight``). This is a **reduced** schema, not a full clone:
bagpuss only simulates a single photometric band and knows every value
exactly (no measurement error), so it does not fabricate GLADE+'s other
columns (multi-band photometry, cross-catalogue identifiers, stellar
mass, merger rate) — see :mod:`bagpuss.glade_export` and the written
README for the full column list and disclaimer.

The export streams the consolidated catalogue tile by tile (bounded
memory, appropriate at full z=3 scale) in the *exact* tile order
:func:`~bagpuss.mdc.pipeline.load_consolidated_catalogue` already uses,
so the exported file's row ``i`` is always the same galaxy any
:class:`~bagpuss.injection.InjectionSet` generated against the same store
references via ``host_galaxy_index == i`` — the exported ID column is
deliberately 0-based (not GLADE+'s 1-based numbering) so it can be used
directly as the host-galaxy lookup table.

.. important::

   This correspondence only holds for injections generated *before* any
   subsequent re-run of :func:`~bagpuss.mdc.pipeline.generate_catalogue_tile`/
   :func:`~bagpuss.mdc.pipeline.consolidate_catalogue` against the same
   store — reconsolidating afterwards silently invalidates any existing
   injection set's ``host_galaxy_index`` values relative to a freshly
   exported catalogue.

The same underlying functions (:func:`bagpuss.glade_export.write_glade_catalogue`,
:func:`~bagpuss.glade_export.write_completeness_curve`,
:func:`~bagpuss.glade_export.write_readme`) also work directly on a
plain, already-in-memory :class:`~bagpuss.catalogue.GalaxyCatalogue`
(demo-scale use, with no zarr store involved) — see
``faketc/scripts/make_injections.py`` for a worked example.

Distribution
-------------

Not yet decided for any specific run. Because each shard already writes
independent, chunked zarr groups with a consolidated-metadata index, the
store is already laid out for **chunked, lazy remote access** — the main
design decision is *where* to host it, not how to structure it further:

* **HTTPS on IGR storage** — simplest; consumers use ``zarr`` + ``fsspec``'s
  HTTP mapper for lazy chunked reads. Needs a web-servable path confirmed
  with whoever administers IGR storage.
* **S3-compatible bucket** — best tooling support in the zarr/``fsspec``/
  ``dask`` ecosystem; needs a bucket provisioned.
* **Zenodo DOI** — citable, but zarr's many-small-files layout is a poor
  fit (would mean tarring the store, which defeats lazy chunked access).
  Workable as a hybrid citation snapshot alongside HTTPS/S3 as the actual
  working-access route.

API reference
--------------

See :doc:`autoapi/bagpuss/mdc/index` for the full API reference.
