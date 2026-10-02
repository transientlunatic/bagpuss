"""Mock Data Challenge (MDC) generation tooling.

Generic, config-driven machinery for producing large-scale bagpuss
catalogues and injection sets that are too big to build in a single
process: the population is split into independent sky-tile and
injection-draw shards (see :mod:`bagpuss.mdc.pipeline`), each written as
its own self-contained zarr group, with an HTCondor DAG (see
:mod:`bagpuss.mdc.condor`) to run and validate the shards incrementally.

Nothing in this subpackage hardcodes a specific run's physical parameters
-- those live in a YAML config (:class:`bagpuss.mdc.config.MDCConfig`)
supplied by the caller.
"""

from __future__ import annotations

__all__: list[str] = []
