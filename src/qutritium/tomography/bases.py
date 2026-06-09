# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Mutually unbiased bases for qutrit tomography."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def mub_bases() -> list[NDArray[np.complex128]]:
    """The four mutually unbiased bases for a qutrit (d = 3).

    ``bases[0]`` is the computational basis; the other three are the
    Fourier-type bases ``3^{-1/2} sum_j omega^{b j^2 + k j} |j>``, where j
    indexes the state, b the basis, and k the vector within it. To measure in
    basis b, rotate the state by ``bases[b].conj().T`` and measure in the
    computational basis.

    Returns
    -------
    list[NDArray[np.complex128]]
        Four ``(3, 3)`` unitaries ``[I, B_0, B_1, B_2]``.

    References
    ----------
    Ivanović, I. D. (1981). Geometrical description of quantal state
    determination. J. Phys. A 14, 3241.

    Wootters, W. K. & Fields, B. D. (1989). Optimal state-determination by
    mutually unbiased measurements. Ann. Phys. 191, 363.
    """
    omega = np.exp(2j * np.pi / 3)
    bases = [np.eye(3, dtype=np.complex128)]  # computational basis
    for b in range(3):
        basis = np.empty((3, 3), dtype=np.complex128)
        for k in range(3):
            basis[:, k] = [omega ** (b * (j ** 2) + k * j) for j in range(3)]
        basis /= np.sqrt(3)
        bases.append(basis)
    return bases


__all__ = ["mub_bases"]
