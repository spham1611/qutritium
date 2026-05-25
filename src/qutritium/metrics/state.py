# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""State-level metrics: compare two density matrices (or pure states)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qutritium.metrics.utils import (_as_density_matrix, _check_same_dim, _EIG_TOL, _matrix_sqrt_hermitian)


def state_fidelity(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
) -> float:
    """Quantum-state fidelity F(rho, sigma) = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    For two pure states |psi>, |phi> this reduces to |<psi|phi>|^2. Symmetric
    in its arguments. Equal to 1 if and only if rho = sigma.

    Parameters
    ----------
    rho : NDArray[np.complex128]
        First state. Either a density matrix of shape ``(d, d)`` or a
        pure-state ket of shape ``(d,)`` or ``(d, 1)``; kets are promoted
        to ``|psi><psi|`` internally.
    sigma : NDArray[np.complex128]
        Second state. Same shape conventions as ``rho``. After promotion,
        the dimension of ``sigma`` must match that of ``rho``.

    Returns
    -------
    float
        Fidelity in ``[0, 1]``. Real-valued; small imaginary parts from
        numerical noise are discarded by taking the real component before
        squaring.

    Raises
    ------
    ValueError
        If either input is neither a ket nor a square matrix, or if the
        promoted matrices have different dimensions.

    References
    ----------
    Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum
    Information*, Cambridge University Press, 2010. §9.2.2.

    Examples
    --------
    >>> import numpy as np
    >>> psi = np.array([1, 0, 0], dtype=complex)
    >>> state_fidelity(psi, psi)
    1.0
    >>> phi = np.array([0, 1, 0], dtype=complex)
    >>> state_fidelity(psi, phi)
    0.0
    """
    rho_mat = _as_density_matrix(rho)
    sigma_mat = _as_density_matrix(sigma)
    _check_same_dim(rho_mat, sigma_mat)

    sqrt_rho = _matrix_sqrt_hermitian(rho_mat)
    inner = sqrt_rho @ sigma_mat @ sqrt_rho
    sqrt_inner = _matrix_sqrt_hermitian(inner)
    trace_val = float(np.real(np.trace(sqrt_inner)))
    return trace_val ** 2


def trace_distance(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
) -> float:
    """Trace distance T(rho, sigma) = (1/2) ||rho - sigma||_1.

    Equal to the maximum classical distinguishability of the two states
    over all POVMs (Helstrom bound). Symmetric and non-negative; T = 0
    iff rho = sigma, T = 1 iff their supports are orthogonal.

    Parameters
    ----------
    rho : NDArray[np.complex128]
        First state. Density matrix ``(d, d)`` or ket ``(d,)``/``(d, 1)``.
    sigma : NDArray[np.complex128]
        Second state. Same shape conventions as ``rho``.

    Returns
    -------
    float
        Trace distance in ``[0, 1]``.

    Raises
    ------
    ValueError
        If either input is malformed or dimensions disagree.

    References
    ----------
    Nielsen & Chuang, *Quantum Computation and Quantum Information*,
    §9.2.1.

    Examples
    --------
    >>> import numpy as np
    >>> psi = np.array([1, 0, 0], dtype=complex)
    >>> phi = np.array([0, 1, 0], dtype=complex)
    >>> trace_distance(psi, phi)
    1.0
    """
    rho_mat = _as_density_matrix(rho)
    sigma_mat = _as_density_matrix(sigma)
    _check_same_dim(rho_mat, sigma_mat)

    diff = rho_mat - sigma_mat
    # diff is Hermitian; eigvalsh is real and numerically stable
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def purity(rho: NDArray[np.complex128]) -> float:
    """Purity P(rho) = tr(rho^2).

    Bounded in ``[1/d, 1]`` for a d-dimensional system. P = 1 iff rho is
    pure; P = 1/d iff rho is the maximally mixed state I/d. Lower values
    indicate more mixed states.

    Parameters
    ----------
    rho : NDArray[np.complex128]
        State to evaluate. Density matrix ``(d, d)`` or ket
        ``(d,)``/``(d, 1)``.

    Returns
    -------
    float
        Purity in ``[1/d, 1]``.

    Raises
    ------
    ValueError
        If ``rho`` is malformed.

    Examples
    --------
    >>> import numpy as np
    >>> psi = np.array([1, 0, 0], dtype=complex)
    >>> purity(psi)
    1.0
    >>> mixed = np.eye(3, dtype=complex) / 3
    >>> round(purity(mixed), 10)
    0.3333333333
    """
    rho_mat = _as_density_matrix(rho)
    return float(np.real(np.trace(rho_mat @ rho_mat)))


def von_neumann_entropy(
        rho: NDArray[np.complex128],
        base: float = 2.0,
) -> float:
    """Von Neumann entropy S(rho) = -tr(rho log rho).

    Equal to 0 for pure states and ``log_base(d)`` for the maximally
    mixed state ``I/d``. Eigenvalues at or below numerical zero are
    dropped before the log to avoid log(0).

    Parameters
    ----------
    rho : NDArray[np.complex128]
        State to evaluate. Density matrix ``(d, d)`` or ket
        ``(d,)``/``(d, 1)``.
    base : float, optional
        Logarithm base. ``2`` (default) reports entropy in qubit-bits;
        ``np.e`` reports in nats. Must be positive and not equal to 1.

    Returns
    -------
    float
        Non-negative entropy in the chosen base. ``0`` for pure states;
        ``log_base(d)`` for the maximally mixed state.

    Raises
    ------
    ValueError
        If ``rho`` is malformed or ``base`` is non-positive or equal to 1.

    Examples
    --------
    >>> import numpy as np
    >>> psi = np.array([1, 0, 0], dtype=complex)
    >>> von_neumann_entropy(psi)
    0.0
    >>> mixed = np.eye(3, dtype=complex) / 3
    >>> round(von_neumann_entropy(mixed), 6)  # log_2(3)
    1.584963
    """
    if base <= 0 or base == 1:
        raise ValueError(f"base must be positive and != 1; got {base}.")

    rho_mat = _as_density_matrix(rho)
    eigvals = np.linalg.eigvalsh(rho_mat)
    # Drop numerical-zero (and any negative-noise) eigenvalues before log.
    # ``>=`` matches the inclusive clamping convention in utils._matrix_sqrt.
    eigvals = eigvals[eigvals >= _EIG_TOL]
    if eigvals.size == 0:
        return 0.0
    entropy_nats = -float(np.sum(eigvals * np.log(eigvals)))
    return entropy_nats / float(np.log(base))


__all__ = [
    "purity",
    "state_fidelity",
    "trace_distance",
    "von_neumann_entropy",
]
