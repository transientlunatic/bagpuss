"""GLADE+-inspired flat-ASCII export for bagpuss galaxy catalogues.

Produces a collaborator-facing galaxy catalogue data product structured
similarly to the real GLADE+ catalogue (Dalya et al. 2022,
https://glade.elte.hu/): a flat, space-delimited ASCII file with one row
per galaxy, plus a companion column-description document and an exact
analytic completeness curve.

Unlike real GLADE+, this is a *reduced* schema: bagpuss only simulates a
single photometric band and knows every value exactly (no measurement
error), so this module does not fabricate GLADE+'s other columns
(multi-band photometry, cross-catalogue names, stellar mass, merger
rate). See :func:`write_readme` for the full disclaimer written alongside
every exported catalogue.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

import numpy as np
from astropy.cosmology import FLRW

from bagpuss.catalogue import GalaxyCatalogue, MagnitudeLimitedSurvey
from bagpuss.universe import LuminosityModel

__all__: list[str] = [
    "GLADE_COLUMNS",
    "write_glade_rows",
    "compute_absolute_magnitudes",
    "write_glade_catalogue",
    "write_completeness_curve",
    "write_readme",
]

#: Ordered ``(name, description)`` pairs for the exported catalogue's
#: columns, shared between :func:`write_glade_rows`'s column order and
#: :func:`write_readme`'s documentation so the two can never drift apart.
GLADE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ID", "0-based sequential index; matches InjectionSet.host_galaxy_index"),
    ("RA", "Right ascension, decimal degrees"),
    ("Dec", "Declination, decimal degrees"),
    ("z", "Redshift (exact simulated value, no measurement error)"),
    (
        "D_L",
        "Luminosity distance, Mpc (exact, computed from z via the run's cosmology)",
    ),
    ("m_app", "Apparent magnitude, single synthetic band"),
    ("M_abs", "Absolute magnitude, same band"),
)


def compute_absolute_magnitudes(luminosities: np.ndarray, m_sun: float) -> np.ndarray:
    """Convert solar luminosities to absolute magnitudes.

    Parameters
    ----------
    luminosities : numpy.ndarray
        Luminosities in solar luminosities, shape ``(n,)``.
    m_sun : float
        Absolute magnitude of the Sun in the band the luminosities were
        computed in -- must match the value used when the catalogue was
        built (e.g. ``MDCConfig.m_sun``, or the ``m_sun`` passed to
        :class:`~bagpuss.luminosity.SchechterLuminosityModel`).

    Returns
    -------
    numpy.ndarray
        Absolute magnitudes, shape ``(n,)``.
    """
    return np.asarray(m_sun - 2.5 * np.log10(luminosities))


def write_glade_rows(
    fileobj: TextIO,
    *,
    ids: np.ndarray,
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    redshifts: np.ndarray,
    luminosity_distances_mpc: np.ndarray,
    apparent_magnitudes: np.ndarray,
    absolute_magnitudes: np.ndarray,
) -> None:
    """Append one batch of galaxies to an open GLADE-style catalogue file.

    Streaming-friendly: writes only the rows given, with no assumption
    that this is the only or the last batch -- callers append tile by
    tile for a large, sharded catalogue (see
    :func:`bagpuss.mdc.pipeline.export_glade_catalogue`) or in one call
    for a small in-memory catalogue (see :func:`write_glade_catalogue`).

    Parameters
    ----------
    fileobj : text file object
        Already-open, writable file handle to append to.
    ids : numpy.ndarray
        0-based global row indices, shape ``(n,)``.
    ra_rad, dec_rad : numpy.ndarray
        Right ascension / declination in radians (bagpuss's internal
        convention), shape ``(n,)``. Converted to decimal degrees on
        write, matching GLADE+'s convention.
    redshifts : numpy.ndarray
        Redshifts, shape ``(n,)``.
    luminosity_distances_mpc : numpy.ndarray
        Luminosity distances in Mpc, shape ``(n,)``.
    apparent_magnitudes, absolute_magnitudes : numpy.ndarray
        Single-band apparent/absolute magnitudes, shape ``(n,)``.
    """
    ra_deg = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)
    for i in range(len(ids)):
        fileobj.write(
            f"{int(ids[i])} "
            f"{ra_deg[i]:.6f} "
            f"{dec_deg[i]:.6f} "
            f"{redshifts[i]:.6f} "
            f"{luminosity_distances_mpc[i]:.3f} "
            f"{apparent_magnitudes[i]:.4f} "
            f"{absolute_magnitudes[i]:.4f}\n"
        )


def write_glade_catalogue(
    catalogue: GalaxyCatalogue,
    cosmology: FLRW,
    path: str | Path,
    *,
    m_sun: float,
    id_offset: int = 0,
) -> None:
    """Write a whole in-memory :class:`GalaxyCatalogue` as a GLADE-style file.

    Convenience wrapper for the common (small/demo-scale) case where the
    full catalogue already fits in memory. For a large, sharded catalogue,
    use :func:`bagpuss.mdc.pipeline.export_glade_catalogue` instead, which
    streams tile by tile.

    Parameters
    ----------
    catalogue : bagpuss.catalogue.GalaxyCatalogue
        The catalogue to export. Row ``i`` becomes ID ``i + id_offset`` --
        pass the *same* catalogue object used to build any
        :class:`~bagpuss.injection.InjectionSet` alongside it, so
        ``host_galaxy_index`` values line up with this file's IDs.
    cosmology : astropy.cosmology.FLRW
        Background cosmology used to compute luminosity distances -- must
        match the cosmology the catalogue was built with.
    path : str or pathlib.Path
        Output file path. Overwritten if it exists.
    m_sun : float
        Absolute magnitude of the Sun in the catalogue's band -- see
        :func:`compute_absolute_magnitudes`.
    id_offset : int, optional
        Added to each row's 0-based index before writing. Default 0.
    """
    if len(catalogue) == 0:
        open(path, "w").close()
        return

    d_l_mpc = cosmology.luminosity_distance(catalogue.redshifts).to("Mpc").value  # pyright: ignore[reportAttributeAccessIssue]
    absolute_magnitudes = compute_absolute_magnitudes(catalogue.luminosities, m_sun)
    ids = np.arange(len(catalogue)) + id_offset
    with open(path, "w") as fileobj:
        write_glade_rows(
            fileobj,
            ids=ids,
            ra_rad=catalogue.ra,
            dec_rad=catalogue.dec,
            redshifts=catalogue.redshifts,
            luminosity_distances_mpc=np.asarray(d_l_mpc),
            apparent_magnitudes=catalogue.apparent_magnitudes,
            absolute_magnitudes=absolute_magnitudes,
        )


def write_completeness_curve(
    selection: MagnitudeLimitedSurvey,
    luminosity: LuminosityModel,
    cosmology: FLRW,
    z_max: float,
    path: str | Path,
    n_points: int = 1000,
) -> None:
    """Write a companion ``(z, completeness)`` table.

    Unlike real GLADE+, where completeness must be estimated empirically,
    bagpuss knows the true luminosity function and selection function, so
    this is computed exactly via
    :meth:`~bagpuss.catalogue.MagnitudeLimitedSurvey.completeness`.

    Parameters
    ----------
    selection : bagpuss.catalogue.MagnitudeLimitedSurvey
        The selection function the catalogue was built with.
    luminosity : bagpuss.universe.LuminosityModel
        The luminosity model the catalogue was built with.
    cosmology : astropy.cosmology.FLRW
        Background cosmology.
    z_max : float
        Maximum redshift to tabulate to.
    path : str or pathlib.Path
        Output file path. Overwritten if it exists.
    n_points : int, optional
        Number of evenly-spaced redshift grid points. Default 1000.
    """
    redshifts = np.linspace(0.0, z_max, n_points)
    completeness = selection.completeness(
        redshifts=redshifts,
        ra=np.zeros_like(redshifts),
        dec=np.zeros_like(redshifts),
        cosmology=cosmology,
        luminosity=luminosity,
    )
    with open(path, "w") as fileobj:
        fileobj.write("# z completeness\n")
        for z, c in zip(redshifts, completeness):
            fileobj.write(f"{z:.6f} {c:.6f}\n")


def write_readme(
    path: str | Path,
    *,
    m_sun: float,
    cosmology_name: str,
    bagpuss_version: str,
    n_galaxies: int,
    z_max: float,
    catalogue_filename: str = "catalogue.dat",
    completeness_filename: str = "completeness.dat",
) -> None:
    """Write the GLADE+-style companion column-description document.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file path. Overwritten if it exists.
    m_sun : float
        Absolute magnitude of the Sun in the catalogue's synthetic band.
    cosmology_name : str
        Name of the cosmology used (e.g. ``"Planck18"``).
    bagpuss_version : str
        bagpuss version that produced the catalogue.
    n_galaxies : int
        Total number of galaxies in the catalogue.
    z_max : float
        Maximum redshift of the simulated volume.
    catalogue_filename, completeness_filename : str, optional
        Filenames of the two companion data files, referenced in the text.
    """
    column_lines = "\n".join(
        f"# Column {i + 1}: {name} -- {desc}"
        for i, (name, desc) in enumerate(GLADE_COLUMNS)
    )
    text = f"""\
bagpuss fake galaxy catalogue
==============================

THIS IS NOT THE REAL GLADE+ CATALOGUE. It is a synthetic galaxy catalogue
generated by bagpuss (https://github.com/transientlunatic/bagpuss) for
testing gravitational-wave cosmological inference pipelines, structured
similarly to GLADE+ (Dalya et al. 2022, https://glade.elte.hu/) for
familiarity, but with a much smaller, honestly-reduced set of columns.

Key differences from real GLADE+:
- Single synthetic photometric band only (m_sun = {m_sun}), not real
  multi-band (B/J/H/Ks/W1/W2) photometry.
- Redshift and luminosity distance are EXACT simulated values -- there is
  no measurement error, unlike real GLADE+'s spectroscopic/photometric
  redshifts and their associated flags/uncertainties.
- No cross-catalogue identifiers (PGC, GWGC, HyperLEDA, 2MASS, WISExSCOS,
  SDSS-DR16Q), no stellar mass, no merger-rate columns. These are not
  fabricated; they are simply not simulated.

Provenance:
- bagpuss version: {bagpuss_version}
- Cosmology: {cosmology_name}
- Maximum redshift (z_max): {z_max}
- Number of galaxies: {n_galaxies}

Files:
- {catalogue_filename}: the galaxy catalogue, one row per galaxy.
- {completeness_filename}: (z, completeness) table -- the exact
  probability that a galaxy at redshift z would pass this catalogue's
  selection function, computed analytically from the known luminosity
  function (see bagpuss.catalogue.MagnitudeLimitedSurvey.completeness).
  Real GLADE+ users must estimate this empirically; this is exact.

Column description ({catalogue_filename}, space-delimited, no header row):
{column_lines}

IMPORTANT: the ID column (column 1) is 0-based and matches
InjectionSet.host_galaxy_index from the injection set generated alongside
this catalogue -- i.e. an injection with host_galaxy_index == 42
corresponds to the row with ID 42 in this file. This is a deliberate
deviation from GLADE+'s own 1-based "GLADE no" numbering, chosen so this
file can be used directly as the host-galaxy lookup table. This
correspondence only holds if the catalogue this file was exported from
was not regenerated/reconsolidated after the injection set was made
against it.
"""
    Path(path).write_text(text)
