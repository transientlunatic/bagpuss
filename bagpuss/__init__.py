"""Bagpuss — simulated galaxy and gravitational-wave catalogues.

Bagpuss runs a seven-stage simulation pipeline to produce fake catalogues
for testing cosmological inference pipelines:

1. Universe simulation (cosmological + galaxy formation + luminosity models)
2. Galaxy set sampling
3. Observed galaxy catalogue (selection function)
4. Binary black hole (BBH) population
5. Gravitational-wave injection and threshold selection
6. Parameter estimation (posteriors and skymaps)
7. Cosmological inference

Institute for Gravitational Research, University of Glasgow.
"""

try:
    from bagpuss._version import __version__
except ImportError:  # pragma: no cover - source tree without a build step
    __version__ = "0.0.0"

__all__: list[str] = ["__version__"]
