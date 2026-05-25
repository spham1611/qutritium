# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Statevector display helpers and unitarity check."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Coefficients with |c| <= _ZERO_TOL are treated as numerical noise.
_ZERO_TOL: float = 1e-10


def statevector_to_state(
    state: NDArray[np.complex128], n_qutrit: int,
) -> tuple[list[complex], list[str]]:
    """Extract non-zero (coefficient, ket-label) pairs from a statevector.

    Walks the statevector and returns the basis states with appreciable
    amplitude (``|c| > 1e-10``), paired with their base-3 ket labels in
    big-endian order. Useful for human-readable inspection and for
    sampling-based measurement routines.

    Parameters
    ----------
    state : NDArray[np.complex128]
        Statevector of shape ``(3**n_qutrit, 1)``.
    n_qutrit : int
        Number of qutrits the state describes. Used to format the ket
        labels and validate the shape.

    Returns
    -------
    tuple of (list of complex, list of str)
        ``(coefficients, labels)`` where ``coefficients[i]`` is the
        complex amplitude of the ket whose label is ``labels[i]``. Labels
        are zero-padded base-3 strings of length ``n_qutrit`` written
        most-significant-qutrit first (e.g. ``"021"`` for n_qutrit=3).

    Raises
    ------
    ValueError
        If ``state.shape`` does not equal ``(3 ** n_qutrit, 1)``.
    """
    expected_shape = (3 ** n_qutrit, 1)
    if state.shape != expected_shape:
        raise ValueError(
            f"State has shape {state.shape}; expected {expected_shape} "
            f"for {n_qutrit} qutrit(s)."
        )

    state_basis: list[int] = []
    state_coeff: list[complex] = []
    for i in range(3 ** n_qutrit):
        if abs(complex(state[i][0])) > _ZERO_TOL:
            state_basis.append(i)
            state_coeff.append(state[i][0])

    state_construction: list[str] = []
    for index in state_basis:
        digits = ""
        tmp = index
        for _ in range(n_qutrit):
            digits += str(int(tmp % 3))
            tmp //= 3
        state_construction.append(digits[::-1])
    return state_coeff, state_construction


def print_statevector(state: NDArray[np.complex128], n_qutrit: int) -> None:
    """Print statevector in ket notation."""
    state_coeff, state_cons = statevector_to_state(state, n_qutrit)
    print("State: ")
    for i, ket in enumerate(state_cons):
        suffix = "" if i == len(state_cons) - 1 else " + "
        print(f"{state_coeff[i]} |{ket}>{suffix}")


__all__ = [
    "print_statevector",
    "statevector_to_state",
]
