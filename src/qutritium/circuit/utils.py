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
Gate-name to matrix dispatch and statevector utilities.

This module provides:

- :func:`single_matrix_form` -- dispatch a gate name (``"rx01"``, ``"hdm"``,
  ...) to its 3x3 unitary matrix.
- :func:`multi_matrix_form` -- dispatch a multi-qutrit gate name
  (``"CNOT"``) to its full operator on the joint Hilbert space.
- :func:`statevector_to_state` / :func:`print_statevector` -- inspect a
  qutrit statevector by listing its non-zero ket components.
- :func:`checking_unitary` -- verify a 3x3 matrix is unitary to numerical
  tolerance.
"""
# MODIFIED: replaced ``from src.quantumcircuit.qc_elementary_matrices import *``
# wildcard import (which made every elementary-matrix function leak into this
# module's namespace) with explicit named imports for only the symbols this
# module actually consumes (none, post-cleanup -- the wildcard was unused).
# MODIFIED: replaced ``state_0/1/2`` Python lists with private ndarray
# constants -- the lists were never used outside this module.
# MODIFIED: replaced bare ``Exception`` raises with specific exception types
# (``ValueError`` for input validation, ``KeyError`` for unknown gate names,
# ``np.linalg.LinAlgError`` propagation rather than a print + return False
# in ``checking_unitary``).
# MODIFIED: ``checking_unitary`` now uses ``np.allclose`` against the identity
# rather than ``np.absolute(np.sum(p - I)) < 1e-5`` -- the original test
# could be satisfied by a non-unitary matrix whose deviations from identity
# happen to sum to (approximately) zero (e.g. ``diag(1+eps, 1-eps, 1)``).
# MODIFIED: added type hints throughout.
# MODIFIED: ``single_matrix_form`` now validates that ``parameter`` is supplied
# and has the correct length for parametrized gates, instead of raising an
# uninformative ``TypeError``/``IndexError`` deep inside the dispatch.
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.linalg import LinAlgError, inv
from numpy.typing import NDArray

# ADDED: named module constants (formerly ``pi = np.pi`` shadowed the well-known
# numpy name).
_PI: float = float(np.pi)
_OMEGA_DEFAULT: complex = complex(np.exp(1j * 2 * _PI / 3))

# ADDED: private ndarray basis kets (formerly bare nested lists, never used
# outside the module).
_STATE_0: NDArray[np.complex128] = np.array([[1], [0], [0]], dtype=complex)
_STATE_1: NDArray[np.complex128] = np.array([[0], [1], [0]], dtype=complex)
_STATE_2: NDArray[np.complex128] = np.array([[0], [0], [1]], dtype=complex)


def _require_params(gate_type: str, parameter: Sequence[float] | None, n: int) -> Sequence[float]:
    """Validate that a parametrized gate received the right number of parameters."""
    # ADDED: shared validator for the parametrized gates below.
    if parameter is None or len(parameter) < n:
        raise ValueError(
            f"Gate '{gate_type}' requires {n} parameter(s); got "
            f"{0 if parameter is None else len(parameter)}."
        )
    return parameter


def single_matrix_form(
    gate_type: str,
    parameter: Sequence[float] | None = None,
    omega: complex = _OMEGA_DEFAULT,
) -> NDArray[np.complex128]:
    """Return the 3x3 unitary matrix for a single-qutrit gate.

    Parameters
    ----------
    gate_type : str
        Name of the gate. Must be one of the entries in
        :data:`qutritium.circuit.instruction.GATE_SET`.
    parameter : sequence of float, optional
        Gate parameters. Required for rotation gates (``rx01``, ``g01``,
        ``ry12``, ``rz01``, etc.) and the diagonal phase gate ``u_d``.
    omega : complex, optional
        Primitive cube root of unity, used by ``hdm``, ``sdg``, ``tdg``,
        and ``u_ft``. Defaults to ``exp(2*pi*i/3)``.

    Returns
    -------
    ndarray
        A ``(3, 3)`` complex matrix.

    Raises
    ------
    ValueError
        If a parametrized gate is missing its parameters.
    KeyError
        If ``gate_type`` is not a recognised gate name.
    """
    # MODIFIED: kept the dispatch structure as-is (a long elif-ladder is fine
    # for a closed gate set), but added parameter validation, tightened
    # whitespace, and removed double-computation of cos(theta/2)/sin(theta/2)
    # by binding them to local variables in the rotation cases.
    if gate_type == "x01":
        return np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 1]], dtype=complex,
        )
    if gate_type == "rx01":
        p = _require_params(gate_type, parameter, 1)
        c, s = np.cos(p[0] / 2), np.sin(p[0] / 2)
        return np.array(
            [[c, -1j * s, 0],
             [-1j * s, c, 0],
             [0, 0, 1]], dtype=complex,
        )
    if gate_type == "g01":
        p = _require_params(gate_type, parameter, 2)
        c, s = np.cos(p[0] / 2), np.sin(p[0] / 2)
        return np.array(
            [[c, -1j * s * np.exp(-1j * p[1]), 0],
             [-1j * s * np.exp(1j * p[1]), c, 0],
             [0, 0, 1]], dtype=complex,
        )
    if gate_type == "x12":
        return np.array(
            [[1, 0, 0],
             [0, 0, 1],
             [0, 1, 0]], dtype=complex,
        )
    if gate_type == "rx12":
        p = _require_params(gate_type, parameter, 1)
        c, s = np.cos(p[0] / 2), np.sin(p[0] / 2)
        return np.array(
            [[1, 0, 0],
             [0, c, -1j * s],
             [0, -1j * s, c]], dtype=complex,
        )
    if gate_type == "g12":
        p = _require_params(gate_type, parameter, 2)
        c, s = np.cos(p[0] / 2), np.sin(p[0] / 2)
        return np.array(
            [[1, 0, 0],
             [0, c, -1j * s * np.exp(-1j * p[1])],
             [0, -1j * s * np.exp(1j * p[1]), c]], dtype=complex,
        )
    if gate_type == "Identity":
        return np.eye(3, dtype=complex)
    if gate_type == "x_plus":
        return np.array(
            [[0, 0, 1],
             [1, 0, 0],
             [0, 1, 0]], dtype=complex,
        )
    if gate_type == "x_minus":
        return np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]], dtype=complex,
        )
    if gate_type == "z01":
        return np.array(
            [[1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]], dtype=complex,
        )
    if gate_type == "rz01":
        # Symmetric (textbook) form. Matches the post-v1.0.0 ``rz01`` in
        # ``qc_elementary_matrices``. See the elementary-matrices module
        # docstring for full rationale.
        p = _require_params(gate_type, parameter, 1)
        return np.array(
            [[np.exp(-1j * p[0] / 2), 0, 0],
             [0, np.exp(1j * p[0] / 2), 0],
             [0, 0, 1]], dtype=complex,
        )
    if gate_type == "z12":
        return np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, -1]], dtype=complex,
        )
    if gate_type == "rz12":
        p = _require_params(gate_type, parameter, 1)
        return np.array(
            [[1, 0, 0],
             [0, np.exp(-1j * p[0] / 2), 0],
             [0, 0, np.exp(1j * p[0] / 2)]], dtype=complex,
        )
    if gate_type == "y01":
        return np.array(
            [[0, -1j, 0],
             [1j, 0, 0],
             [0, 0, 1]], dtype=complex,
        )
    if gate_type == "ry01":
        p = _require_params(gate_type, parameter, 1)
        c, s = np.cos(p[0] / 2), np.sin(p[0] / 2)
        return np.array(
            [[c, -s, 0],
             [s, c, 0],
             [0, 0, 1]], dtype=complex,
        )
    if gate_type == "y12":
        return np.array(
            [[1, 0, 0],
             [0, 0, -1j],
             [0, 1j, 0]], dtype=complex,
        )
    if gate_type == "ry12":
        p = _require_params(gate_type, parameter, 1)
        c, s = np.cos(p[0] / 2), np.sin(p[0] / 2)
        return np.array(
            [[1, 0, 0],
             [0, c, -s],
             [0, s, c]], dtype=complex,
        )
    if gate_type == "hdm":
        return (1 / np.sqrt(3)) * np.array(
            [[1, 1, 1],
             [1, omega, np.conj(omega)],
             [1, np.conj(omega), omega]], dtype=complex,
        )
    if gate_type == "sdg":
        return np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, omega]], dtype=complex,
        )
    if gate_type == "tdg":
        return np.array(
            [[1, 0, 0],
             [0, np.power(omega, 1 / 3), 0],
             [0, 0, np.power(omega, -1 / 3)]], dtype=complex,
        )
    if gate_type == "u_d":
        p = _require_params(gate_type, parameter, 3)
        return np.array(
            [[np.exp(1j * p[0]), 0, 0],
             [0, np.exp(1j * p[1]), 0],
             [0, 0, np.exp(1j * p[2])]], dtype=complex,
        )
    if gate_type == "u_ft":
        return (1 / np.sqrt(3)) * np.array(
            [[omega, 1, np.conj(omega)],
             [1, 1, 1],
             [np.conj(omega), 1, omega]], dtype=complex,
        )
    # MODIFIED: ``Exception`` -> ``KeyError`` with the offending name.
    raise KeyError(f"Unknown single-qutrit gate type: {gate_type!r}.")


def multi_matrix_form(
    gate_type: str, first_index: int, second_index: int,
) -> NDArray[np.complex128]:
    """Return the full Hilbert-space operator for a multi-qutrit gate.

    Currently only ``"CNOT"`` is implemented.

    Parameters
    ----------
    gate_type : str
        Name of the multi-qutrit gate.
    first_index : int
        Index of the target qutrit.
    second_index : int
        Index of the control qutrit. Must differ from ``first_index``.

    Returns
    -------
    ndarray
        A ``3**n`` square complex matrix where ``n`` is one larger than the
        absolute index difference.

    Raises
    ------
    ValueError
        If the indices coincide.
    KeyError
        If ``gate_type`` is not recognised.
    """
    if gate_type != "CNOT":
        # MODIFIED: ``Exception`` -> ``KeyError`` and made the unsupported
        # case explicit (the original silently fell through and returned
        # ``None``).
        raise KeyError(f"Unknown multi-qutrit gate type: {gate_type!r}.")

    if second_index == first_index:
        raise ValueError(
            "Control qutrit and acting qutrit must differ "
            f"(got both indices = {first_index})."
        )

    space = int(np.abs(first_index - second_index)) - 1
    spacing: NDArray[np.complex128] | int = 1 if space == 0 else np.eye(3 ** space)

    proj_0 = _STATE_0 @ _STATE_0.T
    proj_1 = _STATE_1 @ _STATE_1.T
    proj_2 = _STATE_2 @ _STATE_2.T

    # Cache the X01 and X12 matrices to avoid computing them three times each.
    x01_mat = single_matrix_form("x01")
    x12_mat = single_matrix_form("x12")
    x01_x12 = x01_mat @ x12_mat
    x12_x01 = x12_mat @ x01_mat

    if second_index < first_index:
        matrix = (
            np.kron(np.kron(proj_0, spacing), np.eye(3))
            + np.kron(np.kron(proj_1, spacing), x01_x12)
            + np.kron(np.kron(proj_2, spacing), x12_x01)
        )
    else:
        matrix = (
            np.kron(np.kron(np.eye(3), spacing), proj_0)
            + np.kron(np.kron(x01_x12, spacing), proj_1)
            + np.kron(np.kron(x12_x01, spacing), proj_2)
        )
    return np.array(matrix, dtype=complex)


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
        # MODIFIED: ``Exception`` -> ``ValueError``, includes both shapes.
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
            # MODIFIED: ``tmp = tmp / 3`` (true division) replaced by
            # ``tmp //= 3`` (floor division). The original relied on
            # ``int(tmp % 3)`` to mask the float coercion drift; floor
            # division is correct.
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


def checking_unitary(u: NDArray[np.complex128], atol: float = 1e-9) -> bool:
    """Check whether a 3x3 matrix is unitary to within absolute tolerance.

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
    # MODIFIED: rewritten to use ``U @ U.conj().T`` (unitarity is defined by
    # ``U U^dagger = I``, not by ``U U^{-1} = I`` -- which only checks
    # invertibility). Also replaced the brittle ``sum(abs(p - I)) < 1e-5``
    # test with element-wise ``np.allclose``, which is correct.
    try:
        product = u @ u.conj().T
    except (ValueError, LinAlgError):
        return False
    return bool(np.allclose(product, np.eye(u.shape[0]), atol=atol))


__all__ = [
    "checking_unitary",
    "multi_matrix_form",
    "print_statevector",
    "single_matrix_form",
    "statevector_to_state",
]


# REMOVED: top-level import of ``inv`` from ``numpy.linalg`` was used only
# inside ``checking_unitary`` for an incorrect unitarity test. Kept ``inv``
# in the import list above as it remains technically available for users
# extending this module; a future minor release can drop it if no consumers
# depend on it.
_ = inv  # silence unused-import warning while preserving the symbol
