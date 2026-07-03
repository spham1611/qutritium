# MIT License — Copyright (c) 2023-2026 Son Pham
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
    """Quantum-state fidelity F = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    Reduces to |<psi|phi>|^2 for pure states.

    Parameters
    ----------
    rho : NDArray[np.complex128]
        Density matrix ``(d, d)`` or ket ``(d,)``/``(d, 1)``. Kets are
        promoted to ``|psi><psi|`` internally.
    sigma : NDArray[np.complex128]
        Second state, same shape conventions. Dimensions must match
        ``rho`` after promotion.

    Returns
    -------
    float
        Fidelity in ``[0, 1]``.

    Raises
    ------
    ValueError
        On malformed input or dimension mismatch.
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
    """Trace distance T = (1/2) ||rho - sigma||_1.

    Parameters
    ----------
    rho, sigma : NDArray[np.complex128]
        States to compare. Density matrices ``(d, d)`` or kets
        ``(d,)``/``(d, 1)``.

    Returns
    -------
    float
        Trace distance in ``[0, 1]``. ``0`` iff ``rho == sigma``;
        ``1`` iff their supports are orthogonal.

    Raises
    ------
    ValueError
        On malformed input or dimension mismatch.
    """
    rho_mat = _as_density_matrix(rho)
    sigma_mat = _as_density_matrix(sigma)
    _check_same_dim(rho_mat, sigma_mat)

    diff = rho_mat - sigma_mat
    # diff is Hermitian; eigvalsh is real and numerically stable
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def purity(rho: NDArray[np.complex128]) -> float:
    """Purity P = tr(rho^2).

    Parameters
    ----------
    rho : NDArray[np.complex128]
        Density matrix ``(d, d)`` or ket ``(d,)``/``(d, 1)``.

    Returns
    -------
    float
        Purity in ``[1/d, 1]``. ``1`` for pure states; ``1/d`` for
        the maximally mixed state ``I/d``.

    Raises
    ------
    ValueError
        On malformed input.
    """
    rho_mat = _as_density_matrix(rho)
    return float(np.real(np.trace(rho_mat @ rho_mat)))


def von_neumann_entropy(
        rho: NDArray[np.complex128],
        base: float = 2.0,
) -> float:
    """Von Neumann entropy S = -tr(rho log rho).

    Eigenvalues below numerical zero are dropped before the log.

    Parameters
    ----------
    rho : NDArray[np.complex128]
        Density matrix ``(d, d)`` or ket ``(d,)``/``(d, 1)``.
    base : float, optional
        Logarithm base. ``2`` (default) gives bits; ``np.e`` gives nats.
        Must be positive and not equal to 1.

    Returns
    -------
    float
        Non-negative entropy. ``0`` for pure states; ``log_base(d)`` for
        the maximally mixed state ``I/d``.

    Raises
    ------
    ValueError
        On malformed input or invalid ``base``.
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
