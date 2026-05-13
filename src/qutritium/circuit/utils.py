# MIT License
#
# Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Statevector utilities and unitarity checks.

This module provides:

- :func:`statevector_to_state` / :func:`print_statevector` -- inspect a
  qutrit statevector by listing its non-zero ket components.
- :func:`check_unitary` -- verify a square matrix is unitary to numerical
  tolerance.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import NDArray


def statevector_to_state(
    state: NDArray[np.complex128], n_qutrit: int,
) -> tuple[list[complex], list[str]]:
    """Convert a qutrit statevector into a sparse list of (coeff, ket) pairs.

    Parameters
    ----------
    state : ndarray
        A ``(3**n_qutrit, 1)`` complex column vector.
    n_qutrit : int
        Number of qutrits.

    Returns
    -------
    coefficients : list of complex
        Non-zero coefficients of the statevector.
    kets : list of str
        Corresponding ket-label strings (e.g. ``"012"`` for ``|0,1,2>``).

    Raises
    ------
    ValueError
        If ``state`` does not have shape ``(3**n_qutrit, 1)``.
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
    """Print a qutrit statevector in ket form to stdout."""
    state_coeff, state_cons = statevector_to_state(state, n_qutrit)
    print("State: ")
    for i, ket in enumerate(state_cons):
        suffix = "" if i == len(state_cons) - 1 else " + "
        print(f"{state_coeff[i]} |{ket}>{suffix}")


def check_unitary(u: NDArray[np.complex128], atol: float = 1e-9) -> bool:
    """Check whether a square matrix is unitary to within absolute tolerance.

    Parameters
    ----------
    u : ndarray
        Square complex matrix.
    atol : float, optional
        Absolute tolerance for the element-wise check ``U @ U.conj().T == I``.

    Returns
    -------
    bool
        ``True`` if ``u`` is unitary; ``False`` otherwise.
    """
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
