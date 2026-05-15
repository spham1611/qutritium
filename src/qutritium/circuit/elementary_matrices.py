# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.
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

In addition, this module contains operators that do not belong to GGM,
including: qutrit cyclic/inverse shift, Pauli-Z in {1,2}, qutrit Hadamard,
S and T gates, identity, CNOT, CSUM, CPhase, SWAP, and composite rotation
gates. These operators can be written in terms of GGM.

References

- Bertlmann, R. A. & Krammer, P. (2008). *Bloch vectors for qudits*.
arXiv:0806.1174.
- Ringbauer, M. et al. (2022). *A universal qudit quantum processor
with trapped ions*. Nature Physics 18, 1053-1057.
- Wang, Y. et al. (2020). *Qudits and high-dimensional quantum
computing*. Frontiers in Physics 8, 589504.
- Vitanov, N. V. (2012). *Synthesis of arbitrary SU(3) transformations
of 8-dimensional targets*. Phys. Rev. A 85, 032331.
- H. Georgi, "Lie Algebras in Particle Physics", 2nd ed., Westview
Press, 1999.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PI: float = float(np.pi)
OMEGA_DEFAULT: complex = complex(np.exp(1j * 2 * PI / 3))
# exp(2*pi*i/3), default phase for H3, S3, T3

_STATE_0: NDArray[np.complex128] = np.array([[1], [0], [0]], dtype=complex)
_STATE_1: NDArray[np.complex128] = np.array([[0], [1], [0]], dtype=complex)
_STATE_2: NDArray[np.complex128] = np.array([[0], [0], [1]], dtype=complex)


# ===========================================================================
# Gell-Mann matrices (lambda_1 ... lambda_8)
#
#   Symmetric (real off-diagonal):  lambda_1, lambda_4, lambda_6
#   Antisymmetric (imag off-diag):  lambda_2, lambda_5, lambda_7
#   Diagonal:                       lambda_3, lambda_8
# ===========================================================================

# ---------------------------------------------------------------------------
# Symmetric: lambda_1, lambda_4, lambda_6
# ---------------------------------------------------------------------------
def lambda_1() -> NDArray[np.complex128]:
    """Gell-Mann lambda_1: off-diagonal swap in {|0>, |1>}."""
    return np.array(
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 0]], dtype=complex,
    )


def lambda_4() -> NDArray[np.complex128]:
    """Gell-Mann lambda_4: off-diagonal swap in {|0>, |2>}."""
    return np.array(
        [[0, 0, 1],
         [0, 0, 0],
         [1, 0, 0]], dtype=complex,
    )


def lambda_6() -> NDArray[np.complex128]:
    """Gell-Mann lambda_6: off-diagonal swap in {|1>, |2>}."""
    return np.array(
        [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Antisymmetric: lambda_2, lambda_5, lambda_7
# ---------------------------------------------------------------------------
def lambda_2() -> NDArray[np.complex128]:
    """Gell-Mann lambda_2: antisymmetric in {|0>, |1>}."""
    return np.array(
        [[0, -1j, 0],
         [1j, 0, 0],
         [0, 0, 0]], dtype=complex,
    )


def lambda_5() -> NDArray[np.complex128]:
    """Gell-Mann lambda_5: antisymmetric in {|0>, |2>}."""
    return np.array(
        [[0, 0, -1j],
         [0, 0, 0],
         [1j, 0, 0]], dtype=complex,
    )


def lambda_7() -> NDArray[np.complex128]:
    """Gell-Mann lambda_7: antisymmetric in {|1>, |2>}."""
    return np.array(
        [[0, 0, 0],
         [0, 0, -1j],
         [0, 1j, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Diagonal: lambda_3, lambda_8
# ---------------------------------------------------------------------------
def lambda_3() -> NDArray[np.complex128]:
    """Gell-Mann lambda_3: diag(1, -1, 0)."""
    return np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, 0]], dtype=complex,
    )


def lambda_8() -> NDArray[np.complex128]:
    """Gell-Mann lambda_8: diag(1, 1, -2) / sqrt(3)."""
    return (1 / np.sqrt(3)) * np.array(  # type: ignore[no-any-return]
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -2]], dtype=complex,
    )


# ===========================================================================
# Subspace Pauli gates (X, Y, Z embedded in 3x3 with identity on complement)
# ===========================================================================

# ---------------------------------------------------------------------------
# Pauli-X: x01 (lambda_1 + |2><2|), x02 (lambda_4 + |1><1|),
#          x12 (lambda_6 + |0><0|)
# ---------------------------------------------------------------------------
def x01() -> NDArray[np.complex128]:
    """Pauli-X in {|0>, |1>}. Equivalent to lambda_1 + |2><2|."""
    return np.array(
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 1]], dtype=complex,
    )


def x02() -> NDArray[np.complex128]:
    """Pauli-X in {|0>, |2>}. Equivalent to lambda_4 + |1><1|."""
    return np.array(
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]], dtype=complex,
    )


def x12() -> NDArray[np.complex128]:
    """Pauli-X in {|1>, |2>}. Equivalent to lambda_6 + |0><0|."""
    return np.array(
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Pauli-Y: y01 (lambda_2 + |2><2|), y02 (lambda_5 + |1><1|),
#          y12 (lambda_7 + |0><0|)
# ---------------------------------------------------------------------------
def y01() -> NDArray[np.complex128]:
    """Pauli-Y in {|0>, |1>}. Equivalent to lambda_2 + |2><2|."""
    return np.array(
        [[0, -1j, 0],
         [1j, 0, 0],
         [0, 0, 1]], dtype=complex,
    )


def y02() -> NDArray[np.complex128]:
    """Pauli-Y in {|0>, |2>}. Equivalent to lambda_5 + |1><1|."""
    return np.array(
        [[0, 0, -1j],
         [0, 1, 0],
         [1j, 0, 0]], dtype=complex,
    )


def y12() -> NDArray[np.complex128]:
    """Pauli-Y in {|1>, |2>}. Equivalent to lambda_7 + |0><0|."""
    return np.array(
        [[1, 0, 0],
         [0, 0, -1j],
         [0, 1j, 0]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Pauli-Z: z01 (lambda_3 + |2><2|), z02 (diag(1,1,-1)), z12 (diag(1,1,-1))
# ---------------------------------------------------------------------------
def z01() -> NDArray[np.complex128]:
    """Pauli-Z in {|0>, |1>}. Equivalent to lambda_3 + |2><2|."""
    return np.array(
        [[1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]], dtype=complex,
    )


def z02() -> NDArray[np.complex128]:
    """Pauli-Z in {|0>, |2>}. diag(1, 1, -1).

    As a 3x3 matrix this coincides with :func:`z12` because both
    subspaces assign eigenvalue -1 to |2> and +1 to the complement.
    They are kept as separate functions for API symmetry with the X and Y
    families.
    """
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]], dtype=complex,
    )


def z12() -> NDArray[np.complex128]:
    """Pauli-Z in {|1>, |2>}. diag(1, 1, -1)."""
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, -1]], dtype=complex,
    )


# ===========================================================================
# Cyclic shift gates
# ===========================================================================
def x_plus() -> NDArray[np.complex128]:
    """Cyclic shift |i> -> |i+1 mod 3>."""
    return np.array(
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]], dtype=complex,
    )


def x_minus() -> NDArray[np.complex128]:
    """Inverse cyclic shift |i> -> |i-1 mod 3>."""
    return np.array(
        [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]], dtype=complex,
    )


# ===========================================================================
# Subspace rotation gates: Rx, Ry, Rz for {01}, {02}, {12}
#
# Convention: R_axis_ij(theta) = exp(-i * theta/2 * sigma_axis_ij)
# ===========================================================================

# ---------------------------------------------------------------------------
# Rx rotations
# ---------------------------------------------------------------------------
def rx01(theta: float) -> NDArray[np.complex128]:
    """Rx in {|0>, |1>}. Generator: lambda_1."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -1j * s, 0],
         [-1j * s, c, 0],
         [0, 0, 1]], dtype=complex,
    )


def rx02(theta: float) -> NDArray[np.complex128]:
    """Rx in {|0>, |2>}. Generator: lambda_4."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, 0, -1j * s],
         [0, 1, 0],
         [-1j * s, 0, c]], dtype=complex,
    )


def rx12(theta: float) -> NDArray[np.complex128]:
    """Rx in {|1>, |2>}. Generator: lambda_6."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -1j * s],
         [0, -1j * s, c]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Ry rotations
# ---------------------------------------------------------------------------
def ry01(theta: float) -> NDArray[np.complex128]:
    """Ry in {|0>, |1>}. Generator: lambda_2."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -s, 0],
         [s, c, 0],
         [0, 0, 1]], dtype=complex,
    )


def ry02(theta: float) -> NDArray[np.complex128]:
    """Ry in {|0>, |2>}. Generator: lambda_5."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, 0, -s],
         [0, 1, 0],
         [s, 0, c]], dtype=complex,
    )


def ry12(theta: float) -> NDArray[np.complex128]:
    """Ry in {|1>, |2>}. Generator: lambda_7."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -s],
         [0, s, c]], dtype=complex,
    )


# ---------------------------------------------------------------------------
# Rz rotations
# ---------------------------------------------------------------------------
def rz01(phi: float) -> NDArray[np.complex128]:
    """Rz in {|0>, |1>}: diag(exp(-i*phi/2), exp(i*phi/2), 1).

    Generator: lambda_3.
    """
    return np.array(
        [[np.exp(-1j * phi / 2), 0, 0],
         [0, np.exp(1j * phi / 2), 0],
         [0, 0, 1]], dtype=complex,
    )


def rz02(phi: float) -> NDArray[np.complex128]:
    """Rz in {|0>, |2>}: diag(exp(-i*phi/2), 1, exp(i*phi/2)).

    Generator: diag(1, 0, -1).
    """
    return np.array(
        [[np.exp(-1j * phi / 2), 0, 0],
         [0, 1, 0],
         [0, 0, np.exp(1j * phi / 2)]], dtype=complex,
    )


def rz12(phi: float) -> NDArray[np.complex128]:
    """Rz in {|1>, |2>}: diag(1, exp(-i*phi/2), exp(i*phi/2)).

    Generator: diag(0, 1, -1).
    """
    return np.array(
        [[1, 0, 0],
         [0, np.exp(-1j * phi / 2), 0],
         [0, 0, np.exp(1j * phi / 2)]], dtype=complex,
    )


# ===========================================================================
# Generalized rotation gates
# ===========================================================================
def g01(theta: float, phi: float) -> NDArray[np.complex128]:
    """Generalized rotation in {|0>, |1>} with azimuthal phase ``phi``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -1j * s * np.exp(-1j * phi), 0],
         [-1j * s * np.exp(1j * phi), c, 0],
         [0, 0, 1]], dtype=complex,
    )


def g02(theta: float, phi: float) -> NDArray[np.complex128]:
    """Generalized rotation in {|0>, |2>} with azimuthal phase ``phi``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, 0, -1j * s * np.exp(-1j * phi)],
         [0, 1, 0],
         [-1j * s * np.exp(1j * phi), 0, c]], dtype=complex,
    )


def g12(theta: float, phi: float) -> NDArray[np.complex128]:
    """Generalized rotation in {|1>, |2>} with azimuthal phase ``phi``."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[1, 0, 0],
         [0, c, -1j * s * np.exp(-1j * phi)],
         [0, -1j * s * np.exp(1j * phi), c]], dtype=complex,
    )


# ===========================================================================
# Composite rotations: Rz_ij(phi) @ Rx_ij(theta) @ Rz_ij(-phi)
# ===========================================================================
def r01(phi: float, theta: float) -> NDArray[np.complex128]:
    """Composite rotation in {|0>, |1>}: Rz01(phi) @ Rx01(theta) @ Rz01(-phi)."""
    return rz01(phi) @ rx01(theta) @ rz01(-phi)


def r02(phi: float, theta: float) -> NDArray[np.complex128]:
    """Composite rotation in {|0>, |2>}: Rz02(phi) @ Rx02(theta) @ Rz02(-phi)."""
    return rz02(phi) @ rx02(theta) @ rz02(-phi)


def r12(phi: float, theta: float) -> NDArray[np.complex128]:
    """Composite rotation in {|1>, |2>}: Rz12(phi) @ Rx12(theta) @ Rz12(-phi)."""
    return rz12(phi) @ rx12(theta) @ rz12(-phi)


# ===========================================================================
# Diagonal phase gate
# ===========================================================================
def u_d(phi_1: float, phi_2: float, phi_3: float) -> NDArray[np.complex128]:
    """Diagonal unitary diag(exp(i*phi_1), exp(i*phi_2), exp(i*phi_3))."""
    return np.array(
        [[np.exp(1j * phi_1), 0, 0],
         [0, np.exp(1j * phi_2), 0],
         [0, 0, np.exp(1j * phi_3)]], dtype=complex,
    )


# ===========================================================================
# Other single-qutrit gates
# ===========================================================================
def identity() -> NDArray[np.complex128]:
    """3x3 identity (qutrit no-op)."""
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]], dtype=complex,
    )


def hdm(omega: complex = OMEGA_DEFAULT) -> NDArray[np.complex128]:
    """Qutrit Hadamard (DFT F_3 / sqrt(3))."""
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
    """Qutrit S-gate diag(1, 1, omega). S³ = I."""
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, omega]], dtype=complex,
    )


def tdg(omega: complex = OMEGA_DEFAULT) -> NDArray[np.complex128]:
    """Qutrit T-gate diag(1, omega^(1/3), omega^(-1/3)). T⁹ = I."""
    return np.array(
        [[1, 0, 0],
         [0, np.power(omega, 1 / 3), 0],
         [0, 0, np.power(omega, -1 / 3)]], dtype=complex,
    )


# ===========================================================================
# Two-qutrit gates
# ===========================================================================
def cnot(control: int, target: int) -> NDArray[np.complex128]:
    """Two-qutrit CNOT. |0>c: identity, |1>c: X01.X12, |2>c: X12.X01.

    Returns 3^n x 3^n matrix where n = |target - control| + 1.
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

    x01_mat = x01()
    x12_mat = x12()
    x01_x12 = x01_mat @ x12_mat
    x12_x01 = x12_mat @ x01_mat

    if control < target:
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


def csum() -> NDArray[np.complex128]:
    """CSUM gate: |c, t> -> |c, (t + c) mod 3>. 9x9 permutation matrix.

    Ref: Wang et al. (2020), Front. Phys. 8, 589504.
    """
    mat = np.zeros((9, 9), dtype=complex)
    for c in range(3):
        for t in range(3):
            mat[3 * c + ((t + c) % 3), 3 * c + t] = 1.0
    return mat


def csum_dag() -> NDArray[np.complex128]:
    """CSUM inverse: |c, t> -> |c, (t - c) mod 3>. Just the transpose."""
    return csum().T.copy()


def cphase() -> NDArray[np.complex128]:
    """CPhase: |c, t> -> omega^{c*t} |c, t>. Qutrit analogue of CZ.

    Related to CSUM by: CPhase = (I x F3) . CSUM . (I x F3^dag).
    """
    omega = np.exp(2j * np.pi / 3)
    diag = np.array(
        [omega ** (c * t) for c in range(3) for t in range(3)],
        dtype=complex,
    )
    return np.diag(diag)


def swap3() -> NDArray[np.complex128]:
    """SWAP: |a, b> -> |b, a>. Self-inverse permutation."""
    mat = np.zeros((9, 9), dtype=complex)
    for a in range(3):
        for b in range(3):
            mat[3 * b + a, 3 * a + b] = 1.0
    return mat
