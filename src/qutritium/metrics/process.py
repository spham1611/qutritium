# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Process-level metrics: compare two unitaries.

Channel-based metrics (Choi/Kraus inputs) are deferred to v1.4 alongside
the noise-channel framework.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qutritium.metrics.utils import _check_same_dim, _check_square_matrix


def process_fidelity(
        u_ideal: NDArray[np.complex128],
        u_actual: NDArray[np.complex128],
) -> float:
    """Entanglement/process fidelity F_pro = |tr(U_ideal^dag U_actual)|^2 / d^2.

    Equal to 1 iff ``U_actual = U_ideal`` up to global phase. Equal to
    ``F_pro(U, V) = F_pro(V, U)`` so symmetric in its arguments. Always
    non-negative.

    Parameters
    ----------
    u_ideal : NDArray[np.complex128]
        Target unitary, shape ``(d, d)``.
    u_actual : NDArray[np.complex128]
        Implemented unitary, shape ``(d, d)``. Must match ``u_ideal``'s
        dimension.

    Returns
    -------
    float
        Process fidelity in ``[0, 1]``.

    Raises
    ------
    ValueError
        If either input is not a square 2D matrix, or their dimensions
        disagree.

    Notes
    -----
    Both inputs are assumed unitary; this is not enforced. Passing a
    non-unitary matrix returns a meaningless number rather than raising,
    so the caller is responsible for ensuring unitarity. For
    channel-vs-channel fidelity (e.g. a noisy realization compared
    against a target unitary), use the Choi-matrix framework arriving
    in v1.4.

    References
    ----------
    Horodecki, M., Horodecki, P. & Horodecki, R. (1999). *General
    teleportation channel, singlet fraction, and quasidistillation*.
    Phys. Rev. A 60, 1888.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)  # X01
    >>> process_fidelity(u, u)
    1.0
    >>> round(process_fidelity(np.eye(3, dtype=complex), u), 6)
    0.111111
    """
    u1 = np.asarray(u_ideal, dtype=np.complex128)
    u2 = np.asarray(u_actual, dtype=np.complex128)
    _check_square_matrix(u1, "u_ideal")
    _check_square_matrix(u2, "u_actual")
    _check_same_dim(u1, u2)

    d = u1.shape[0]
    inner = np.trace(u1.conj().T @ u2)
    return float(np.abs(inner) ** 2 / d ** 2)


def average_gate_fidelity(
        u_ideal: NDArray[np.complex128],
        u_actual: NDArray[np.complex128],
) -> float:
    """Average gate fidelity F_avg = (d * F_pro + 1) / (d + 1).

    The Haar-average of ``state_fidelity(U_ideal|psi>, U_actual|psi>)``
    over input states ``|psi>``. This is the figure of merit quoted in
    experimental gate-fidelity papers and is the natural metric for
    randomized benchmarking.

    Parameters
    ----------
    u_ideal : NDArray[np.complex128]
        Target unitary, shape ``(d, d)``.
    u_actual : NDArray[np.complex128]
        Implemented unitary, shape ``(d, d)``.

    Returns
    -------
    float
        Average gate fidelity in ``[1/(d+1), 1]``. Returns 1 iff
        ``u_actual = u_ideal`` up to global phase.

    Raises
    ------
    ValueError
        Propagated from :func:`process_fidelity` on shape mismatches.

    References
    ----------
    Horodecki et al. (1999); Nielsen, M. A. (2002). *A simple formula
    for the average gate fidelity of a quantum dynamical operation*.
    Phys. Lett. A 303, 249.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)  # X01
    >>> average_gate_fidelity(u, u)
    1.0
    """
    u1 = np.asarray(u_ideal, dtype=np.complex128)
    f_pro = process_fidelity(u1, u_actual)
    d = u1.shape[0]
    return (d * f_pro + 1.0) / (d + 1.0)


__all__ = [
    "average_gate_fidelity",
    "process_fidelity",
]
