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
    """Promote a ket to |psi><psi|; pass a density matrix through unchanged."""
    arr = np.asarray(state, dtype=np.complex128)
    if arr.ndim == 1:
        ket = arr.reshape(-1, 1)
        return ket @ ket.conj().T  # type: ignore[no-any-return]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr @ arr.conj().T  # type: ignore[no-any-return]
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return arr
    raise ValueError(
        f"Expected ket of shape (d,) or (d, 1), or density matrix (d, d); "
        f"got shape {arr.shape}."
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


__all__: list[str] = []  # nothing is public from this module
