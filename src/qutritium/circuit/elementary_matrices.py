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
Elementary qutrit unitary matrices (SU(3)) + Gell-Mann matrices or GGM

This module collects the building-block 3x3 unitary matrices used by the
qutritium.circuit package. All functions return ``numpy.ndarray`` with ``dtype=complex``.

Gell-Mann matrices and SU(3) generators
----------------------------------------
The eight Gell-Mann matrices lambda_1 ... lambda_8 are the standard
generators of the Lie algebra su(3), analogous to the three Pauli matrices
for su(2). They are traceless, Hermitian, and satisfy
Tr(lambda_a @ lambda_b) = 2 * delta_{a,b}. They can be used to expand any qutrit density operator.

These 8 generators splits into three families which are:
    - Symmetric (real): lambda_1, lambda_4, lambda_6
    - Antisymmetric (imaginary): lambda_2, lambda_5, lambda_7
    - Off Diagonal: lambda_3, lambda_8

In addition, this module also contains operators that do not belong to GGM, including: Qutrit cyclic-inverse shift,
Pauli-Z in {1,2}, Qutrit Hadamard, Qutrit S and T gate, Qutrit Identity, Qutrit CNOT, Qutrit composite rotation gates.
It should be noted these operators can be written in terms of GGM

References
----------
- Bertlmann, R. A. & Krammer, P. (2008). *Bloch vectors for qudits*.
  arXiv:0806.1174.
- H. Georgi, "Lie Algebras in Particle Physics", 2nd ed., Westview Press, 1999.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ADDED: named constants -- previously ``pi = np.pi`` was a module-level shadow
# of the well-known name.
PI: float = float(np.pi)
OMEGA_DEFAULT: complex = complex(np.exp(1j * 2 * PI / 3))
"""Primitive cube root of unity ``exp(2*pi*i/3)`` used as the default phase
in the qutrit Hadamard, S-, and T-gates."""

# ADDED: explicit ndarray column vectors (formerly nested Python lists).
# INTERNAL USE ONLY!
_STATE_0: NDArray[np.complex128] = np.array([[1], [0], [0]], dtype=complex)
_STATE_1: NDArray[np.complex128] = np.array([[0], [1], [0]], dtype=complex)
_STATE_2: NDArray[np.complex128] = np.array([[0], [0], [1]], dtype=complex)


# ---------------------------------------------------------------------------
# Subspace Pauli-X analogues
# ---------------------------------------------------------------------------
def x_plus() -> NDArray[np.complex128]:
    """Cyclic shift ``|i> -> |i+1 mod 3>``.

    Not a Gell-Mann matrix
    """
    return np.array(
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]], dtype=complex,
    )


def x_minus() -> NDArray[np.complex128]:
    """Inverse cyclic shift ``|i> -> |i-1 mod 3>``.

    Not a Gell-Mann matrix
    """
    return np.array(
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]], dtype=complex,
    )


def x01() -> NDArray[np.complex128]:
    """Pauli-X restricted to the {|0>, |1>} subspace.

    Corresponds to Gell-Mann matrix lambda_1.
    """
    return np.array(
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 1]], dtype=complex,
    )


def x02() -> NDArray[np.complex128]:
    """Pauli-X restricted to the {|0>, |2>} subspace.

    Corresponds to Gell-Mann matrix lambda_4.
    """
    return np.array(
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]], dtype=complex,
    )


def x12() -> NDArray[np.complex128]:
    """Pauli-X restricted to the {|1>, |2>} subspace.

    Corresponds to Gell-Mann matrix lambda_6.
    """
    return np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Subspace Pauli-Y analogues
# ---------------------------------------------------------------------------
def y01() -> NDArray[np.complex128]:
    """Pauli-Y restricted to the {|0>, |1>} subspace.

    Corresponds to Gell-Mann matrix lambda_2.
    """
    return np.array(
        [[0, -1j, 0],
         [1j, 0, 0],
         [0, 0, 1]], dtype=complex,
    )


def y12() -> NDArray[np.complex128]:
    """Pauli-Y restricted to the {|1>, |2>} subspace.

    Corresponds to Gell-Mann matrix lambda_7.
    """
    return np.array(
        [[1, 0, 0],
         [0, 0, -1j],
         [0, 1j, 0]], dtype=complex,
    )


def y02() -> NDArray[np.complex128]:
    """Pauli-Y restricted to the {|0>, |2>} subspace.

    Corresponds to Gell-Mann matrix lambda_5.
    """
    return np.array(
        [[0, 0, -1j],
         [0, 1, 0],
         [1j, 0, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Subspace Pauli-Z analogues
# ---------------------------------------------------------------------------
def z01() -> NDArray[np.complex128]:
    """Pauli-Z restricted to the {|0>, |1>} subspace.

    Corresponds to Gell-Mann matrix lambda_3.
    """
    return np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Off diagonal matrice (we have already included lambda_3 previously)
# ---------------------------------------------------------------------------
def z12() -> NDArray[np.complex128]:
    """Pauli-Z restricted to the {|1>, |2>} subspace.

    Not a Gell-Mann matrix
    """
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]], dtype=complex,
    )


def lambda_8() -> NDArray[np.complex128]:
    """Gell-Mann matrix lambda_8: diag(1, 1, -2) / sqrt(3).

    Corresponds to Gell-Mann matrix lambda_8, off-diagonal
    """
    return (1 / np.sqrt(3)) * np.array(  # type: ignore[no-any-return]
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -2]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Subspace rotation gates
# ---------------------------------------------------------------------------
def rx01(theta: float) -> NDArray[np.complex128]:
    """Rotation about X in the {|0>, |1>} subspace by angle ``theta``.

    Not a Gell-Mann matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -1j * s, 0],
         [-1j * s, c, 0],
         [0, 0, 1]], dtype=complex,
    )


def rx12(theta: float) -> NDArray[np.complex128]:
    """Rotation about X in the {|1>, |2>} subspace by angle ``theta``.

    Not a Gell-Mann matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -1j * s],
         [0, -1j * s, c]], dtype=complex,
    )


def ry01(theta: float) -> NDArray[np.complex128]:
    """Rotation about Y in the {|0>, |1>} subspace by angle ``theta``.

    Not a Gell-Mann matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -s, 0],
         [s, c, 0],
         [0, 0, 1]], dtype=complex,
    )


def ry12(theta: float) -> NDArray[np.complex128]:
    """Rotation about Y in the {|1>, |2>} subspace by angle ``theta``.

    Not a Gell-Mann matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -s],
         [0, s, c]], dtype=complex,
    )


def rz01(phi: float) -> NDArray[np.complex128]:
    """Rotation about Z in the {|0>, |1>} subspace by angle ``phi``.

    Not a Gell-Mann matrix

    Note
    ----
    The composite rotation :func:`r01` was simultaneously updated to
    ``rz01(phi) @ rx01(theta) @ rz01(-phi)`` (sign of ``phi`` flipped relative
    to the v0.0.1 convention) so that its returned matrix is **numerically
    identical** to the v0.0.1 output. This keeps the SU(3) decomposition in
    :mod:`qutritium.decomposition` working without any changes to its
    angle-extraction logic.
    """
    return np.array(
        [[np.exp(-1j * phi / 2), 0, 0],
         [0, np.exp(1j * phi / 2), 0],
         [0, 0, 1]], dtype=complex,
    )


def rz12(phi: float) -> NDArray[np.complex128]:
    """Rotation about Z in the {|1>, |2>} subspace by angle ``phi``.

    Not a Gell-Mann matrix

    Uses the symmetric convention ``diag(1, exp(-i*phi/2), exp(i*phi/2))``.
    See :func:`rz01` for the convention change rationale.
    """
    return np.array(
        [[1, 0, 0],
         [0, np.exp(-1j * phi / 2), 0],
         [0, 0, np.exp(1j * phi / 2)]], dtype=complex,
    )


def g01(theta: float, phi: float) -> NDArray[np.complex128]:
    """Generalized rotation in the {|0>, |1>} subspace with phase ``phi``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -1j * s * np.exp(-1j * phi), 0],
         [-1j * s * np.exp(1j * phi), c, 0],
         [0, 0, 1]], dtype=complex,
    )


def g12(theta: float, phi: float) -> NDArray[np.complex128]:
    """Generalized rotation in the {|1>, |2>} subspace with phase ``phi``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -1j * s * np.exp(-1j * phi)],
         [0, -1j * s * np.exp(1j * phi), c]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Other single-qutrit gates
# ---------------------------------------------------------------------------
def hdm(omega: complex = OMEGA_DEFAULT) -> NDArray[np.complex128]:
    """Qutrit Hadamard / discrete Fourier gate.

    Not a Gell-Mann matrix; this is the 3x3 DFT matrix ``F_3 / sqrt(3)``.

    Parameters
    ----------
    omega : complex, optional
        Primitive cube root of unity. Defaults to ``exp(2*pi*i/3)``.
    """
    return (1 / np.sqrt(3)) * np.array(  # type: ignore[no-any-return]
        [[1, 1, 1],
         [1, omega, omega ** 2],
         [1, omega ** 2, omega]], dtype=complex,
    )


def u_ft(omega: complex = OMEGA_DEFAULT) -> NDArray[np.complex128]:
    """Fourier-related qutrit gate."""
    return (1 / np.sqrt(3)) * np.array(  # type: ignore[no-any-return]
        [[omega, 1, np.conj(omega)],
         [1, 1, 1],
         [np.conj(omega), 1, omega]], dtype=complex,
    )


def sdg(omega: complex = OMEGA_DEFAULT) -> NDArray[np.complex128]:
    """Qutrit S-gate ``diag(1, 1, omega)``."""
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, omega]], dtype=complex,
    )


def tdg(omega: complex = OMEGA_DEFAULT) -> NDArray[np.complex128]:
    """Qutrit T-gate ``diag(1, omega**(1/3), omega**(-1/3))``."""
    return np.array(
        [[1, 0, 0],
         [0, np.power(omega, 1 / 3), 0],
         [0, 0, np.power(omega, -1 / 3)]], dtype=complex,
    )


def identity() -> NDArray[np.complex128]:
    """3x3 identity (qutrit no-op)."""
    # MODIFIED: renamed from PEP-8-violating ``Identity`` to ``identity``;
    # the legacy alias is preserved at module bottom.
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Composite rotations and diagonal phase gate
# ---------------------------------------------------------------------------
def r01(phi: float, theta: float) -> NDArray[np.complex128]:
    """Composite rotation ``Rz01(phi) @ Rx01(theta) @ Rz01(-phi)``.

    Not a Gell-Mann matrix
    """
    return rz01(phi) @ rx01(theta) @ rz01(-phi)


def r12(phi: float, theta: float) -> NDArray[np.complex128]:
    """Composite rotation ``Rz12(phi) @ Rx12(theta) @ Rz12(-phi)``.

    Not a Gell-Mann matrix
    """
    return rz12(phi) @ rx12(theta) @ rz12(-phi)


def u_d(phi_1: float, phi_2: float, phi_3: float) -> NDArray[np.complex128]:
    """Diagonal unitary ``diag(exp(i*phi_1), exp(i*phi_2), exp(i*phi_3))``.

    Not a Gell-Mann matrix
    """
    return np.array(
        [[np.exp(1j * phi_1), 0, 0],
         [0, np.exp(1j * phi_2), 0],
         [0, 0, np.exp(1j * phi_3)]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Two-qutrit gate
# ---------------------------------------------------------------------------
def cnot(target: int, control: int) -> NDArray[np.complex128]:
    """Generalized two-qutrit CNOT gate.

    The control qutrit cycles the target qutrit through the {|0>, |1>, |2>}
    basis depending on the control state. ``|0>`` leaves the target alone;
    ``|1>`` applies ``X01 @ X12``; ``|2>`` applies ``X12 @ X01``. It acts only
    within 2 qutrit subspace

    Parameters
    ----------
    target : int
        Index of the target qutrit.
    control : int
        Index of the control qutrit. Must not equal ``target``.

    Returns
    -------
    ndarray
        A ``3**n x 3**n`` complex matrix where ``n`` is one larger than the
        absolute index difference.

    Raises
    ------
    ValueError
        If ``target == control``.
    """
    if control == target:
        raise ValueError(
            "Control qutrit and target qutrit must differ "
            f"(got target = control = {target})."
        )

    space = int(np.abs(target - control)) - 1
    spacing: NDArray | int = 1 if space == 0 else np.eye(3 ** space)

    proj_0 = _STATE_0 @ _STATE_0.T
    proj_1 = _STATE_1 @ _STATE_1.T
    proj_2 = _STATE_2 @ _STATE_2.T

    if control < target:
        matrix = (
                np.kron(np.kron(proj_0, spacing), np.eye(3))
                + np.kron(np.kron(proj_1, spacing), x01() @ x12())
                + np.kron(np.kron(proj_2, spacing), x12() @ x01())
        )
    else:
        matrix = (
                np.kron(np.kron(np.eye(3), spacing), proj_0)
                + np.kron(np.kron(x01() @ x12(), spacing), proj_1)
                + np.kron(np.kron(x12() @ x01(), spacing), proj_2)
        )
    return np.array(matrix, dtype=complex)
