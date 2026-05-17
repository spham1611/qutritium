# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Single-qutrit Gate classes wrapping the matrices in elementary_matrices.

Convention: R_{axis,ij}(theta) = exp(-i theta/2 * sigma_{axis,ij})
Ref: Bertlmann & Krammer (2008), arXiv:0806.1174.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import qutritium.circuit.elementary_matrices as em
from qutritium.gates.base import Gate


# ===================================================================
# Fixed (zero-parameter) gates
# ===================================================================

class I3(Gate):
    """Qutrit identity gate (3×3)."""

    def __init__(self) -> None:
        super().__init__(label="I3", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.identity()

    def inverse(self) -> "I3":
        return I3()


# --- Subspace Pauli-X ---

class X01(Gate):
    """Pauli-X in {|0⟩, |1⟩} subspace.

    Swaps |0⟩ ↔ |1⟩, leaves |2⟩ unchanged.
    Equivalent to Gell-Mann λ₁ + |2⟩⟨2|.
    """

    def __init__(self) -> None:
        super().__init__(label="X01", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.x01()

    def inverse(self) -> "X01":
        # X² = I  ⟹  X† = X
        return X01()


class X02(Gate):
    """Pauli-X in {|0⟩, |2⟩} subspace.

    Swaps |0⟩ ↔ |2⟩, leaves |1⟩ unchanged.
    Equivalent to Gell-Mann λ₄ + |1⟩⟨1|.
    """

    def __init__(self) -> None:
        super().__init__(label="X02", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.x02()

    def inverse(self) -> "X02":
        return X02()


class X12(Gate):
    """Pauli-X in {|1⟩, |2⟩} subspace.

    Swaps |1⟩ ↔ |2⟩, leaves |0⟩ unchanged.
    Equivalent to Gell-Mann λ₆ + |0⟩⟨0|.
    """

    def __init__(self) -> None:
        super().__init__(label="X12", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.x12()

    def inverse(self) -> "X12":
        return X12()


# --- Subspace Pauli-Y ---

class Y01(Gate):
    """Pauli-Y in {|0⟩, |1⟩} subspace.

    Equivalent to Gell-Mann λ₂ + |2⟩⟨2|.
    """

    def __init__(self) -> None:
        super().__init__(label="Y01", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.y01()


class Y02(Gate):
    """Pauli-Y in {|0⟩, |2⟩} subspace.

    Equivalent to Gell-Mann λ₅ + |1⟩⟨1|.
    """

    def __init__(self) -> None:
        super().__init__(label="Y02", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.y02()


class Y12(Gate):
    """Pauli-Y in {|1⟩, |2⟩} subspace.

    Equivalent to Gell-Mann λ₇ + |0⟩⟨0|.
    """

    def __init__(self) -> None:
        super().__init__(label="Y12", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.y12()


# --- Subspace Pauli-Z ---

class Z01(Gate):
    """Pauli-Z in {|0⟩, |1⟩} subspace.

    diag(1, -1, 1). Equivalent to Gell-Mann λ₃ + |2⟩⟨2|.
    """

    def __init__(self) -> None:
        super().__init__(label="Z01", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.z01()

    def inverse(self) -> "Z01":
        # Z² = I  ⟹  Z† = Z
        return Z01()


class Z02(Gate):
    """Pauli-Z in {|0⟩, |2⟩} subspace.

    diag(1, 1, -1).
    """

    def __init__(self) -> None:
        super().__init__(label="Z02", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.z02()

    def inverse(self) -> "Z02":
        return Z02()


class Z12(Gate):
    """Phase-flip on |2⟩: diag(1, 1, -1).

    Derived: |0⟩⟨0| + |1⟩⟨1| − |2⟩⟨2|. Numerically identical to Z02.
    """

    def __init__(self) -> None:
        super().__init__(label="Z12", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.z12()

    def inverse(self) -> "Z12":
        return Z12()


# --- Cyclic shift ---

class XPlus(Gate):
    """Cyclic shift |i> -> |i+1 mod 3>. X+^3 = I."""

    def __init__(self) -> None:
        super().__init__(label="X+", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.x_plus()

    def inverse(self) -> "XMinus":
        return XMinus()


class XMinus(Gate):
    """Inverse cyclic shift |i> -> |i-1 mod 3>."""

    def __init__(self) -> None:
        super().__init__(label="X-", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.x_minus()

    def inverse(self) -> "XPlus":
        return XPlus()


# --- Discrete gates with default ω ---

class H3(Gate):
    """Qutrit Hadamard (DFT F3/sqrt3). Creates equal superposition from |0>."""

    def __init__(self) -> None:
        super().__init__(label="H3", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.hdm()


class S3(Gate):
    """S gate: diag(1, 1, omega). S^3 = I."""

    def __init__(self) -> None:
        super().__init__(label="S3", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.sdg()


class T3(Gate):
    """T gate: diag(1, omega^(1/3), omega^(-1/3)). T^9 = I."""

    def __init__(self) -> None:
        super().__init__(label="T3", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.tdg()


# --- Parametric rotation gates ---

# --- Rx ---

class Rx01(Gate):
    """Rx in {|0⟩, |1⟩}."""

    def __init__(self, theta: float) -> None:
        super().__init__(label="Rx01", num_qutrits=1, params=(theta,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.rx01(self._params[0])

    def inverse(self) -> "Rx01":
        return Rx01(-self._params[0])


class Rx02(Gate):
    """Rx in {|0⟩, |2⟩}."""

    def __init__(self, theta: float) -> None:
        super().__init__(label="Rx02", num_qutrits=1, params=(theta,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.rx02(self._params[0])

    def inverse(self) -> "Rx02":
        return Rx02(-self._params[0])


class Rx12(Gate):
    """Rx in {|1⟩, |2⟩}."""

    def __init__(self, theta: float) -> None:
        super().__init__(label="Rx12", num_qutrits=1, params=(theta,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.rx12(self._params[0])

    def inverse(self) -> "Rx12":
        return Rx12(-self._params[0])


# --- Ry ---

class Ry01(Gate):
    """Ry in {|0⟩, |1⟩}."""

    def __init__(self, theta: float) -> None:
        super().__init__(label="Ry01", num_qutrits=1, params=(theta,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.ry01(self._params[0])

    def inverse(self) -> "Ry01":
        return Ry01(-self._params[0])


class Ry02(Gate):
    """Ry in {|0⟩, |2⟩}."""

    def __init__(self, theta: float) -> None:
        super().__init__(label="Ry02", num_qutrits=1, params=(theta,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.ry02(self._params[0])

    def inverse(self) -> "Ry02":
        return Ry02(-self._params[0])


class Ry12(Gate):
    """Ry in {|1⟩, |2⟩}."""

    def __init__(self, theta: float) -> None:
        super().__init__(label="Ry12", num_qutrits=1, params=(theta,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.ry12(self._params[0])

    def inverse(self) -> "Ry12":
        return Ry12(-self._params[0])


# --- Rz ---

class Rz01(Gate):
    """Rz in {|0⟩, |1⟩}."""

    def __init__(self, phi: float) -> None:
        super().__init__(label="Rz01", num_qutrits=1, params=(phi,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.rz01(self._params[0])

    def inverse(self) -> "Rz01":
        return Rz01(-self._params[0])


class Rz02(Gate):
    """Rz in {|0⟩, |2⟩}."""

    def __init__(self, phi: float) -> None:
        super().__init__(label="Rz02", num_qutrits=1, params=(phi,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.rz02(self._params[0])

    def inverse(self) -> "Rz02":
        return Rz02(-self._params[0])


class Rz12(Gate):
    """Rz in {|1⟩, |2⟩}."""

    def __init__(self, phi: float) -> None:
        super().__init__(label="Rz12", num_qutrits=1, params=(phi,))

    def matrix(self) -> NDArray[np.complex128]:
        return em.rz12(self._params[0])

    def inverse(self) -> "Rz12":
        return Rz12(-self._params[0])


# --- Generalized rotations g_ij(theta, phi) ---
# Native gate in trapped-ion implementations (Ringbauer et al., Nat. Phys. 18, 1053)

class G01(Gate):
    """Generalized rotation in {|0⟩, |1⟩} with azimuthal phase φ.

    g₀₁(θ, φ) = exp(-i θ/2 · (cos(φ)·X₀₁ + sin(φ)·Y₀₁))

    Parameters
    ----------
    theta : float
        Polar rotation angle.
    phi : float
        Azimuthal phase of the drive.
    """

    def __init__(self, theta: float, phi: float) -> None:
        super().__init__(label="G01", num_qutrits=1, params=(theta, phi))

    def matrix(self) -> NDArray[np.complex128]:
        return em.g01(self._params[0], self._params[1])

    def inverse(self) -> "G01":
        return G01(-self._params[0], self._params[1])


class G02(Gate):
    """Generalized rotation in {|0⟩, |2⟩} with azimuthal phase φ.

    g₀₂(θ, φ) = exp(-i θ/2 · (cos(φ)·X₀₂ + sin(φ)·Y₀₂))

    Parameters
    ----------
    theta : float
        Polar rotation angle.
    phi : float
        Azimuthal phase of the drive.
    """

    def __init__(self, theta: float, phi: float) -> None:
        super().__init__(label="G02", num_qutrits=1, params=(theta, phi))

    def matrix(self) -> NDArray[np.complex128]:
        return em.g02(self._params[0], self._params[1])

    def inverse(self) -> "G02":
        return G02(-self._params[0], self._params[1])


class G12(Gate):
    """Generalized rotation in {|1⟩, |2⟩} with azimuthal phase φ.

    g₁₂(θ, φ) = exp(-i θ/2 · (cos(φ)·X₁₂ + sin(φ)·Y₁₂))

    Parameters
    ----------
    theta : float
        Polar rotation angle.
    phi : float
        Azimuthal phase of the drive.
    """

    def __init__(self, theta: float, phi: float) -> None:
        super().__init__(label="G12", num_qutrits=1, params=(theta, phi))

    def matrix(self) -> NDArray[np.complex128]:
        return em.g12(self._params[0], self._params[1])

    def inverse(self) -> "G12":
        return G12(-self._params[0], self._params[1])


# ===================================================================
# Three-parameter diagonal phase gate
# ===================================================================

class Ud(Gate):
    """Diagonal phase: diag(e^{i*phi1}, e^{i*phi2}, e^{i*phi3}). Virtual-Z in hardware."""

    def __init__(self, phi1: float, phi2: float, phi3: float) -> None:
        super().__init__(label="Ud", num_qutrits=1, params=(phi1, phi2, phi3))

    def matrix(self) -> NDArray[np.complex128]:
        return em.u_d(self._params[0], self._params[1], self._params[2])

    def inverse(self) -> "Ud":
        return Ud(-self._params[0], -self._params[1], -self._params[2])


# ===================================================================
# Convenience: Fourier-related gate
# ===================================================================

class UFT(Gate):
    """Fourier-related gate U_FT."""

    def __init__(self) -> None:
        super().__init__(label="UFT", num_qutrits=1)

    def matrix(self) -> NDArray[np.complex128]:
        return em.u_ft()


__all__ = [
    # Fixed gates
    "I3", "X01", "X02", "X12",
    "Y01", "Y02", "Y12",
    "Z01", "Z02", "Z12",
    "XPlus", "XMinus",
    "H3", "S3", "T3", "UFT",
    # Parametric rotation gates
    "Rx01", "Rx02", "Rx12",
    "Ry01", "Ry02", "Ry12",
    "Rz01", "Rz02", "Rz12",
    # Generalized rotations
    "G01", "G02", "G12",
    # Diagonal phase gate
    "Ud",
]
