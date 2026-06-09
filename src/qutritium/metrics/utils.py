# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Internal numerical helpers for the metrics module.

All names are underscore-prefixed and excluded from ``__all__``. Do not
import these from outside ``qutritium.metrics``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Eigenvalues with value <= _EIG_TOL are treated as numerical zero.
_EIG_TOL: float = 1e-12


def _as_density_matrix(
        state: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Promote a ket to |psi><psi|; validate and pass a density matrix through.

    Kets (shape ``(d,)`` or ``(d, 1)``) are normalized before forming the
    outer product.

    Raises
    ------
    ValueError
        If the shape is unsupported, a ket has zero norm, or a matrix is
        not Hermitian / not trace-1.
    """
    arr_state = np.asarray(state, dtype=np.complex128)
    if arr_state.ndim == 1 or (arr_state.ndim == 2 and arr_state.shape[1] == 1):
        ket = arr_state.reshape(-1, 1)
        norm = np.linalg.norm(ket)
        if norm == 0.0:
            raise ValueError("State vector has zero norm; cannot normalize.")
        ket = ket / norm
        return ket @ ket.conj().T  # type: ignore[no-any-return]
    if arr_state.ndim == 2 and arr_state.shape[0] == arr_state.shape[1]:
        if not np.allclose(arr_state, arr_state.conj().T, atol=1e-8):
            raise ValueError("Density matrix is not Hermitian.")
        if not np.isclose(np.trace(arr_state), 1.0, atol=1e-8):
            raise ValueError(
                f"Density matrix must have unit trace; got {np.trace(arr_state):.6g}."
            )
        return arr_state
    raise ValueError(
        f"Expected ket of shape (d,) or (d, 1), or density matrix (d, d); "
        f"got shape {arr_state.shape}."
    )


def _check_same_dim(a: NDArray, b: NDArray) -> None:
    """Raise ValueError if a and b have mismatched shapes."""
    if a.shape != b.shape:
        raise ValueError(
            f"Operands have mismatched shapes: {a.shape} vs {b.shape}."
        )


def _check_square_matrix(arr: NDArray, name: str) -> None:
    """Raise ValueError unless arr is a 2D square array."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a square 2D matrix; got shape {arr.shape}."
        )


def _check_unitary(arr: NDArray, name: str, atol: float = 1e-8) -> None:
    """Raise ValueError unless arr @ arr.conj().T equals the identity.

    Tolerance matches ``SU3Decomposition.__init__``.
    """
    product = arr @ arr.conj().T
    if not np.allclose(product, np.eye(arr.shape[0]), atol=atol):
        raise ValueError(
            f"{name} is not unitary (||U U† - I|| > {atol})."
        )


def _matrix_sqrt_hermitian(
        rho: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Stable sqrt of a Hermitian PSD matrix via eigendecomposition.

    Negative eigenvalues from numerical noise are clamped to zero. Faster
    and more stable than scipy.linalg.sqrtm for the Hermitian case.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_eigvals = np.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals) @ eigvecs.conj().T  # type: ignore[no-any-return]
