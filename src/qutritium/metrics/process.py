# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Process-level metrics: compare two unitaries.

Channel-based metrics (Choi/Kraus inputs) are not in this release;
``qutritium.tomography.reconstruct_process`` produces Choi matrices, but a
Choi-based fidelity is not provided yet.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qutritium.metrics.utils import (
    _check_same_dim,
    _check_square_matrix,
    _check_unitary,
)


def process_fidelity(
        u_ideal: NDArray[np.complex128],
        u_actual: NDArray[np.complex128],
) -> float:
    """Process fidelity F = |tr(U_ideal^dag U_actual)|^2 / d^2.

    Inputs are validated as unitary to within ``atol=1e-8``.
    Channel-vs-channel (Choi/Kraus) fidelity is not in this release.

    Parameters
    ----------
    u_ideal, u_actual : NDArray[np.complex128]
        Target and implemented unitaries, both shape ``(d, d)``.

    Returns
    -------
    float
        Process fidelity in ``[0, 1]``. ``1`` iff ``u_actual == u_ideal``
        up to global phase.

    Raises
    ------
    ValueError
        On shape mismatch, non-square inputs, or non-unitary inputs.
    """
    u1 = np.asarray(u_ideal, dtype=np.complex128)
    u2 = np.asarray(u_actual, dtype=np.complex128)
    _check_square_matrix(u1, "u_ideal")
    _check_square_matrix(u2, "u_actual")
    _check_same_dim(u1, u2)
    _check_unitary(u1, "u_ideal")
    _check_unitary(u2, "u_actual")

    d = u1.shape[0]
    inner = np.trace(u1.conj().T @ u2)
    return float(np.abs(inner) ** 2 / d ** 2)


def average_gate_fidelity(
        u_ideal: NDArray[np.complex128],
        u_actual: NDArray[np.complex128],
) -> float:
    """Average gate fidelity F_avg = (d * F_pro + 1) / (d + 1).

    Haar-average of state fidelity between ``U_ideal|psi>`` and
    ``U_actual|psi>`` over pure inputs.

    Parameters
    ----------
    u_ideal, u_actual : NDArray[np.complex128]
        Target and implemented unitaries, shape ``(d, d)``.

    Returns
    -------
    float
        Average gate fidelity in ``[1/(d+1), 1]``.

    Raises
    ------
    ValueError
        Propagated from ``process_fidelity``.
    """
    u1 = np.asarray(u_ideal, dtype=np.complex128)
    f_pro = process_fidelity(u1, u_actual)
    d = u1.shape[0]
    return (d * f_pro + 1.0) / (d + 1.0)  # type: ignore[no-any-return]


__all__ = [
    "average_gate_fidelity",
    "process_fidelity",
]
