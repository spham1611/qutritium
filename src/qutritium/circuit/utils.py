# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Statevector display helpers and unitarity check."""
from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray


def statevector_to_state(
    state: NDArray[np.complex128], n_qutrit: int,
) -> tuple[list[complex], list[str]]:
    """Extract non-zero (coefficient, ket_string) pairs from a statevector."""
    expected_shape = (3 ** n_qutrit, 1)
    if state.shape != expected_shape:
        raise ValueError(
            f"State has shape {state.shape}; expected {expected_shape} "
            f"for {n_qutrit} qutrit(s)."
        )

    state_basis: list[int] = []
    state_coeff: list[complex] = []
    for i in range(3 ** n_qutrit):
        if abs(complex(state[i][0])) != 0.0:
            state_basis.append(i)
            state_coeff.append(state[i][0])

    state_construction: list[str] = []
    for index in state_basis:
        digits = ""
        tmp = index
        for _ in range(n_qutrit):
            digits += str(int(tmp % 3))
            tmp //= 3
        state_construction.append(digits)
    return state_coeff, state_construction


def print_statevector(state: NDArray[np.complex128], n_qutrit: int) -> None:
    """Print statevector in ket notation."""
    state_coeff, state_cons = statevector_to_state(state, n_qutrit)
    print("State: ")
    for i, ket in enumerate(state_cons):
        suffix = "" if i == len(state_cons) - 1 else " + "
        print(f"{state_coeff[i]} |{ket}>{suffix}")


def check_unitary(u: NDArray[np.complex128], atol: float = 1e-9) -> bool:
    """True if u @ u^H ≈ I within atol."""
    try:
        product = u @ u.conj().T
    except (ValueError, LinAlgError):
        return False
    return bool(np.allclose(product, np.eye(u.shape[0]), atol=atol))


__all__ = [
    "check_unitary",
    "print_statevector",
    "statevector_to_state",
]
