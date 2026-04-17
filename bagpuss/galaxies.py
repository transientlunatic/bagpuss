"""Galaxy set sampling — Stage 2 of the bagpuss pipeline.

Samples a set of galaxies from the distribution defined by the simulated
Universe produced in Stage 1.  The primary data container exported by this
module is :class:`GalaxySet`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__: list[str] = ["GalaxySet"]


@dataclass
class GalaxySet:
    """A set of galaxies sampled from a simulated Universe.

    Parameters
    ----------
    redshifts : numpy.ndarray
        Redshifts of each galaxy, shape ``(n,)``.
    luminosities : numpy.ndarray
        Luminosities of each galaxy in solar luminosities, shape ``(n,)``.

    Examples
    --------
    Construct a small galaxy set directly:

    >>> import numpy as np
    >>> gs = GalaxySet(
    ...     redshifts=np.array([0.1, 0.3, 0.5]),
    ...     luminosities=np.array([1e10, 2e10, 5e9]),
    ... )
    >>> len(gs)
    3
    """

    redshifts: np.ndarray
    luminosities: np.ndarray

    def __len__(self) -> int:
        """Return the number of galaxies in the set."""
        return int(self.redshifts.shape[0])
