"""Tests for bagpuss.plotting."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bagpuss.galaxies import GalaxySet  # noqa: E402
from bagpuss.plotting import plot_universe_heatmap  # noqa: E402


def _make_galaxies(n: int = 200, seed: int = 0) -> GalaxySet:
    rng = np.random.default_rng(seed)
    return GalaxySet(
        redshifts=rng.uniform(0.0, 1.0, n),
        luminosities=10.0 ** rng.uniform(8.0, 12.0, n),
    )


class TestPlotUniverseHeatmap(unittest.TestCase):
    """Tests for plot_universe_heatmap."""

    def setUp(self) -> None:
        """Create a shared galaxy set for use across tests."""
        self.galaxies = _make_galaxies()

    def tearDown(self) -> None:
        """Close all open figures after each test."""
        plt.close("all")

    def test_returns_figure(self) -> None:
        """Return value is a matplotlib Figure."""
        fig = plot_universe_heatmap(self.galaxies)
        self.assertIsInstance(fig, plt.Figure)

    def test_figure_has_two_axes(self) -> None:
        """Figure contains two axes: main plot and colorbar."""
        fig = plot_universe_heatmap(self.galaxies)
        self.assertEqual(len(fig.axes), 2)

    def test_xlabel_contains_z(self) -> None:
        """X-axis label references redshift z."""
        fig = plot_universe_heatmap(self.galaxies)
        self.assertIn("z", fig.axes[0].get_xlabel())

    def test_ylabel_contains_log(self) -> None:
        """Y-axis label references log-luminosity."""
        fig = plot_universe_heatmap(self.galaxies)
        self.assertIn("log", fig.axes[0].get_ylabel().lower())

    def test_default_title(self) -> None:
        """Default title contains the word 'heatmap'."""
        fig = plot_universe_heatmap(self.galaxies)
        self.assertIn("heatmap", fig.axes[0].get_title().lower())

    def test_custom_title(self) -> None:
        """Custom title is applied to the axes."""
        fig = plot_universe_heatmap(self.galaxies, title="My Title")
        self.assertEqual(fig.axes[0].get_title(), "My Title")

    def test_custom_bins(self) -> None:
        """Custom bin count produces a valid figure without error."""
        fig = plot_universe_heatmap(self.galaxies, bins=10)
        self.assertIsInstance(fig, plt.Figure)

    def test_empty_galaxies_raises(self) -> None:
        """Empty GalaxySet raises ValueError."""
        empty = GalaxySet(redshifts=np.array([]), luminosities=np.array([]))
        with self.assertRaises(ValueError):
            plot_universe_heatmap(empty)

    def test_single_galaxy(self) -> None:
        """A single galaxy with bins=1 produces a valid figure."""
        single = GalaxySet(
            redshifts=np.array([0.5]),
            luminosities=np.array([1e10]),
        )
        fig = plot_universe_heatmap(single, bins=1)
        self.assertIsInstance(fig, plt.Figure)

    def test_custom_colormap(self) -> None:
        """Custom colormap name is accepted without error."""
        fig = plot_universe_heatmap(self.galaxies, colormap="plasma")
        self.assertIsInstance(fig, plt.Figure)
