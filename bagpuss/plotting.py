"""Plotting utilities for bagpuss galaxy catalogues.

Provides visualisation helpers for inspecting galaxy sets produced by the
bagpuss simulation pipeline.
"""

from __future__ import annotations

import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt

from bagpuss.galaxies import GalaxySet

__all__: list[str] = ["plot_universe_heatmap"]


def plot_universe_heatmap(
    galaxies: GalaxySet,
    *,
    bins: int = 50,
    title: str = "Galaxy density heatmap",
    colormap: str = "viridis",
    colorbar_label: str = "Count",
) -> matplotlib.figure.Figure:
    """Produce a 2-D density heatmap of redshift vs log-luminosity.

    Parameters
    ----------
    galaxies : GalaxySet
        Galaxy set to visualise.
    bins : int, optional
        Number of bins along each axis.  Default is 50.
    title : str, optional
        Figure title.
    colormap : str, optional
        Matplotlib colormap name.  Default is ``"viridis"``.
    colorbar_label : str, optional
        Label for the colour bar axis.  Default is ``"Count"``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object; the caller is responsible for saving or displaying it.

    Raises
    ------
    ValueError
        If ``galaxies`` contains no entries.
    """
    if len(galaxies) == 0:
        raise ValueError("galaxies must contain at least one entry")

    log_lum = np.log10(galaxies.luminosities)

    counts, z_edges, l_edges = np.histogram2d(
        galaxies.redshifts,
        log_lum,
        bins=bins,
    )

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(z_edges, l_edges, counts.T, cmap=colormap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$\log_{10}(L\,/\,L_\odot)$")
    ax.set_title(title)

    fig.tight_layout()
    return fig
