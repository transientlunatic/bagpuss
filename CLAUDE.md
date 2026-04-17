# CLAUDE.md — bagpuss

## Project

Bagpuss is a Python package that generates simulated galaxy catalogues and gravitational-wave catalogues
for testing cosmological inference pipelines. It is developed at the Institute for Gravitational Research,
University of Glasgow. The package is MIT-licensed and distributed via PyPI.

## Setup

```bash
uv sync              # install all dependencies into the managed virtualenv
uv add <package>     # add a runtime dependency
uv add --dev <pkg>   # add a dev/test dependency
```

## Common commands

| Task | Command |
|---|---|
| Run tests | `python -m unittest discover -s tests` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Type check (mypy) | `mypy bagpuss/` |
| Type check (pyright) | `pyright` |
| Pre-commit (all files) | `pre-commit run --all-files` |

## Architecture

Bagpuss runs a seven-stage simulation pipeline:

1. **Universe simulation** — draw a cosmology (cosmological model + galaxy formation model + luminosity model)
2. **Galaxy set** — sample galaxies from the simulated distribution
3. **Galaxy catalogue** — apply a selection function to produce a fake observed catalogue
4. **BBH population** — simulate a binary black hole population across the galaxy set
5. **GW injection** — inject GW signals into noise (real or simulated) and determine which events are above threshold for a given detector network
6. **Parameter estimation** — run PE on super-threshold events to produce posteriors and skymaps
7. **Cosmological inference** — feed posteriors + skymaps into a downstream inference pipeline

## Code conventions

- test-driven development is used where possible, and all new code should be accompanied by tests
- Python 3.x; type annotations are expected on all public functions and classes
- Ruff is the primary linter and formatter (`ruff check` + `ruff format`); configuration lives in `pyproject.toml`
  - Ruff implements the rules of flake8 and is a black-compatible formatter — run ruff rather than black or flake8 directly
- Tests live in `tests/` and use the standard `unittest` framework
- Pre-commit hooks enforce lint and format checks before each commit

## Tooling note

`ruff` covers the functionality of both `black` (formatting) and `flake8` (linting).
Run `ruff check .` and `ruff format .` as the single source of truth for style.
Black and flake8 are not run separately.
