"""Smoke tests — verify the package and all pipeline-stage modules import cleanly."""

import unittest


class TestImports(unittest.TestCase):
    """Ensure every public module can be imported without error."""

    def test_package_imports(self) -> None:
        """Bagpuss top-level package imports."""
        import bagpuss  # noqa: F401

    def test_universe_imports(self) -> None:
        """bagpuss.universe imports."""
        import bagpuss.universe  # noqa: F401

    def test_galaxies_imports(self) -> None:
        """bagpuss.galaxies imports."""
        import bagpuss.galaxies  # noqa: F401

    def test_catalogue_imports(self) -> None:
        """bagpuss.catalogue imports."""
        import bagpuss.catalogue  # noqa: F401

    def test_population_imports(self) -> None:
        """bagpuss.population imports."""
        import bagpuss.population  # noqa: F401

    def test_injection_imports(self) -> None:
        """bagpuss.injection imports."""
        import bagpuss.injection  # noqa: F401

    def test_pe_imports(self) -> None:
        """bagpuss.pe imports."""
        import bagpuss.pe  # noqa: F401

    def test_inference_imports(self) -> None:
        """bagpuss.inference imports."""
        import bagpuss.inference  # noqa: F401

    def test_luminosity_imports(self) -> None:
        """bagpuss.luminosity imports."""
        import bagpuss.luminosity  # noqa: F401

    def test_plotting_imports(self) -> None:
        """bagpuss.plotting imports."""
        import matplotlib

        matplotlib.use("Agg")
        import bagpuss.plotting  # noqa: F401

    def test_cli_imports(self) -> None:
        """bagpuss.cli imports."""
        import bagpuss.cli  # noqa: F401

    def test_version(self) -> None:
        """Package exposes a __version__ string."""
        import bagpuss

        self.assertIsInstance(bagpuss.__version__, str)
