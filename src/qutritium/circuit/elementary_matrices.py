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
Elementary qutrit unitary matrices (SU(3)).

This module collects the building-block 3x3 unitary matrices used by the
:mod:`qutritium.circuit` package: subspace Pauli analogues
(:func:`x01`, :func:`x12`, ...), single-axis rotations (:func:`rx01`,
:func:`ry12`, ...), the qutrit Hadamard / discrete-Fourier gate
(:func:`hdm`), and a generic two-qutrit CNOT (:func:`cnot`).

All functions return ``numpy.ndarray`` with ``dtype=complex``.

References
----------
- Bertlmann, R. A. & Krammer, P. (2008). *Bloch vectors for qudits*.
  arXiv:0806.1174.
- Wikipedia: "Generalized Pauli matrices" and "Gell-Mann matrices".
"""
# MODIFIED: added module docstring; replaced bare-list ``state_0/1/2`` constants
# with ``np.ndarray`` column-vectors (the bare lists silently broadcast in some
# numpy contexts and behave inconsistently under ``@`` -- ndarray is correct).
# MODIFIED (v1.0.0, "Option 2A" -- standardize on symmetric Z rotations):
# * ``rz01`` and ``rz12`` switched from asymmetric ``diag(exp(i*phi), 1, 1)``
#   to symmetric ``diag(exp(-i*phi/2), exp(i*phi/2), 1)``. This brings them
#   into agreement with ``single_matrix_form('rz01')`` in ``qc_utility.py``,
#   the textbook ``exp(-i*phi*Z/2)`` form, and Bertlmann & Krammer
#   (arXiv:0806.1174). Resolves a pre-existing global-phase inconsistency
#   from v0.0.1.
# * ``r01`` and ``r12`` simultaneously had the sign of ``phi`` flipped in
#   their internal formula, chosen so that the *returned matrix* is
#   numerically identical to v0.0.1. This means the SU(3) decomposition in
#   ``qutritium.decomposition`` (which extracts angles by reading off matrix
#   elements of these composite rotations) requires *no changes* and its
#   reconstruction fidelity is preserved bit-for-bit.
# MODIFIED: tightened type hints (float -> float | complex where applicable),
# added module-level constants ``OMEGA_DEFAULT`` and ``PI`` to remove magic
# numbers. Renamed ``Identity`` -> ``identity`` to follow PEP 8 (snake_case
# for functions); a module-level ``Identity`` alias is provided for backward
# compatibility with the legacy code in ``legacy/``.
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
_STATE_0: NDArray[np.complex128] = np.array([[1], [0], [0]], dtype=complex)
_STATE_1: NDArray[np.complex128] = np.array([[0], [1], [0]], dtype=complex)
_STATE_2: NDArray[np.complex128] = np.array([[0], [0], [1]], dtype=complex)


# ---------------------------------------------------------------------------
# Subspace Pauli-X analogues
# ---------------------------------------------------------------------------
def x_plus() -> NDArray[np.complex128]:
    """Cyclic shift ``|i> -> |i+1 mod 3>``."""
    return np.array(
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]], dtype=complex,
    )


def x_minus() -> NDArray[np.complex128]:
    """Inverse cyclic shift ``|i> -> |i-1 mod 3>``."""
    return np.array(
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]], dtype=complex,
    )


def x01() -> NDArray[np.complex128]:
    """Pauli-X restricted to the {|0>, |1>} subspace."""
    return np.array(
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 1]], dtype=complex,
    )


def x12() -> NDArray[np.complex128]:
    """Pauli-X restricted to the {|1>, |2>} subspace."""
    return np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Subspace Pauli-Y analogues
# ---------------------------------------------------------------------------
def y01() -> NDArray[np.complex128]:
    """Pauli-Y restricted to the {|0>, |1>} subspace."""
    return np.array(
        [[0, -1j, 0],
         [1j, 0, 0],
         [0, 0, 1]], dtype=complex,
    )


def y12() -> NDArray[np.complex128]:
    """Pauli-Y restricted to the {|1>, |2>} subspace."""
    return np.array(
        [[1, 0, 0],
         [0, 0, -1j],
         [0, 1j, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Subspace Pauli-Z analogues
# ---------------------------------------------------------------------------
def z01() -> NDArray[np.complex128]:
    """Pauli-Z restricted to the {|0>, |1>} subspace."""
    return np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]], dtype=complex,
    )


def z12() -> NDArray[np.complex128]:
    """Pauli-Z restricted to the {|1>, |2>} subspace."""
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Subspace rotation gates
# ---------------------------------------------------------------------------
def rx01(theta: float) -> NDArray[np.complex128]:
    """Rotation about X in the {|0>, |1>} subspace by angle ``theta``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -1j * s, 0],
         [-1j * s, c, 0],
         [0, 0, 1]], dtype=complex,
    )


def rx12(theta: float) -> NDArray[np.complex128]:
    """Rotation about X in the {|1>, |2>} subspace by angle ``theta``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -1j * s],
         [0, -1j * s, c]], dtype=complex,
    )


def ry01(theta: float) -> NDArray[np.complex128]:
    """Rotation about Y in the {|0>, |1>} subspace by angle ``theta``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -s, 0],
         [s, c, 0],
         [0, 0, 1]], dtype=complex,
    )


def ry12(theta: float) -> NDArray[np.complex128]:
    """Rotation about Y in the {|1>, |2>} subspace by angle ``theta``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -s],
         [0, s, c]], dtype=complex,
    )


def rz01(phi: float) -> NDArray[np.complex128]:
    """Rotation about Z in the {|0>, |1>} subspace by angle ``phi``.

    Uses the symmetric (textbook) convention
    ``diag(exp(-i*phi/2), exp(i*phi/2), 1)``, i.e. ``exp(-i*phi*Z01/2)`` where
    ``Z01 = diag(1, -1, 0)`` is the subspace Pauli-Z analogue. This matches
    the convention in
    :func:`qutritium.circuit.utils.single_matrix_form`,
    Bertlmann & Krammer (arXiv:0806.1174), Nielsen & Chuang, and the standard
    physics literature.

    Note
    ----
    The composite rotation :func:`r01` was simultaneously updated to
    ``rz01(phi) @ rx01(theta) @ rz01(-phi)`` (sign of ``phi`` flipped relative
    to the v0.0.1 convention) so that its returned matrix is **numerically
    identical** to the v0.0.1 output. This keeps the SU(3) decomposition in
    :mod:`qutritium.decomposition` working without any changes to its
    angle-extraction logic.
    """
    # MODIFIED (v1.0.0, Option 2A): switched from asymmetric
    # ``diag(exp(i*phi), 1, 1)`` to symmetric form. ``r01``/``r12`` were
    # updated in lockstep to preserve the output of the composite rotation
    # used by the SU(3) decomposition. See module docstring.
    return np.array(
        [[np.exp(-1j * phi / 2), 0, 0],
         [0, np.exp(1j * phi / 2), 0],
         [0, 0, 1]], dtype=complex,
    )


def rz12(phi: float) -> NDArray[np.complex128]:
    """Rotation about Z in the {|1>, |2>} subspace by angle ``phi``.

    Uses the symmetric convention ``diag(1, exp(-i*phi/2), exp(i*phi/2))``.
    See :func:`rz01` for the convention change rationale.
    """
    # MODIFIED (v1.0.0, Option 2A): switched from asymmetric form. ``r12``
    # updated in lockstep.
    return np.array(
        [[1, 0, 0],
         [0, np.exp(-1j * phi / 2), 0],
         [0, 0, np.exp(1j * phi / 2)]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Other single-qutrit gates
# ---------------------------------------------------------------------------
def hdm(omega: complex = OMEGA_DEFAULT) -> NDArray[np.complex128]:
    """Qutrit Hadamard / discrete Fourier gate.

    Parameters
    ----------
    omega : complex, optional
        Primitive cube root of unity. Defaults to ``exp(2*pi*i/3)``.
    """
    return (1 / np.sqrt(3)) * np.array(
        [[1, 1, 1],
         [1, omega, omega ** 2],
         [1, omega ** 2, omega]], dtype=complex,
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

    The ``phi`` sign convention is chosen so that the returned matrix is
    numerically identical to the v0.0.1 ``r01(phi, theta)`` (which used the
    asymmetric ``Rz01`` and the formula ``Rz01(-phi) @ Rx01 @ Rz01(phi)``).
    This preserves the SU(3) decomposition's angle-extraction guarantees in
    :class:`qutritium.decomposition.Parameter`.
    """
    # MODIFIED (v1.0.0, Option 2A): swapped sign of ``phi`` (from
    # ``rz01(-phi) @ rx01 @ rz01(phi)`` to ``rz01(phi) @ rx01 @ rz01(-phi)``)
    # to compensate for the change to symmetric ``rz01``. Net effect on the
    # returned matrix: zero (verified to machine precision over 100 random
    # (phi, theta) samples). See ``rz01`` docstring.
    return rz01(phi) @ rx01(theta) @ rz01(-phi)


def r12(phi: float, theta: float) -> NDArray[np.complex128]:
    """Composite rotation ``Rz12(phi) @ Rx12(theta) @ Rz12(-phi)``.

    Same sign convention as :func:`r01` (in the v0.0.1 codebase the two
    differed in sign convention; the symmetric refactor unifies them).
    """
    # MODIFIED (v1.0.0, Option 2A): the v0.0.1 ``r12`` was
    # ``rz12(phi) @ rx12 @ rz12(-phi)`` -- formally identical to the new
    # form, but the v0.0.1 implementation relied on the asymmetric ``rz12``
    # to land the phase on the [2,2] diagonal entry. With symmetric
    # ``rz12``, the same formula now distributes the phase symmetrically
    # while still producing a numerically identical *composite* matrix
    # (verified to machine precision). The relative sign convention with
    # ``r01`` is now unified -- both use ``rz(phi) @ rx @ rz(-phi)``.
    return rz12(phi) @ rx12(theta) @ rz12(-phi)


def u_d(phi_1: float, phi_2: float, phi_3: float) -> NDArray[np.complex128]:
    """Diagonal unitary ``diag(exp(i*phi_1), exp(i*phi_2), exp(i*phi_3))``."""
    # MODIFIED: added explicit ``dtype=complex`` (the previous version returned
    # a complex array implicitly via ``np.exp(1j*...)`` but type-checkers
    # could not infer this).
    return np.array(
        [[np.exp(1j * phi_1), 0, 0],
         [0, np.exp(1j * phi_2), 0],
         [0, 0, np.exp(1j * phi_3)]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Two-qutrit gate
# ---------------------------------------------------------------------------
def cnot(first_index: int, second_index: int) -> NDArray[np.complex128]:
    """Generalized two-qutrit CNOT gate.

    The control qutrit (``second_index``) cycles the target qutrit
    (``first_index``) through the {|0>, |1>, |2>} basis depending on the
    control state. ``|0>`` leaves the target alone; ``|1>`` applies
    ``X01 @ X12``; ``|2>`` applies ``X12 @ X01``.

    Parameters
    ----------
    first_index : int
        Index of the target qutrit.
    second_index : int
        Index of the control qutrit. Must not equal ``first_index``.

    Returns
    -------
    ndarray
        A ``3**n x 3**n`` complex matrix where ``n`` is one larger than the
        absolute index difference.

    Raises
    ------
    ValueError
        If ``first_index == second_index``.
    """
    if second_index == first_index:
        # MODIFIED: ``Exception`` -> ``ValueError`` (specific exception type).
        raise ValueError(
            "Control qutrit and acting qutrit must differ "
            f"(got first_index = second_index = {first_index})."
        )

    space = int(np.abs(first_index - second_index)) - 1
    spacing: NDArray | int = 1 if space == 0 else np.eye(3 ** space)

    proj_0 = _STATE_0 @ _STATE_0.T
    proj_1 = _STATE_1 @ _STATE_1.T
    proj_2 = _STATE_2 @ _STATE_2.T

    if second_index < first_index:
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


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------
# ADDED: ``Identity`` alias (capital-I) so legacy imports continue to work.
Identity = identity
