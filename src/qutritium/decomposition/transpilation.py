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
SU(3) decomposition into native single-qutrit rotations.

The decomposition is hardware-agnostic: it produces angle parameters and a
list of :class:`qutritium.circuit.instruction.Instruction` objects that any
backend (statevector simulator, ARTIQ, AQT, IBM via Qiskit, ...) can consume.
For more info on how to decompose SU(3) matrices into Euler angles rotations
can be found here
References
----------
- Reck, M., Zeilinger, A., Bernstein, H. J. & Bertani, P. (1994).
  *Experimental realization of any discrete unitary operator*.
  Phys. Rev. Lett. 73, 58.
- Bronzan, J. B. (1988). *Parametrization of SU(3)*.
  Phys. Rev. D 38, 1994.
- Vitanov, N. V. (2012). *Synthesis of arbitrary SU(3) transformations
  of atomic qutrits*. Phys. Rev. A 85, 032331.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.elementary_matrices import r01, r12, u_d
from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit

_PI: float = float(np.pi)


class DecompositionAngles(NamedTuple):
    """Nine Euler-like angles of the SU(3) decomposition."""
    theta1: float
    theta2: float
    theta3: float
    phi1: float
    phi2: float
    phi3: float
    phi4: float
    phi5: float
    phi6: float


class NativeDecomposition(NamedTuple):
    """Result of :meth:`SU3Decomposition.to_native`."""
    phases: NDArray
    instructions: list[Instruction]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _safe_arccos(x: float) -> float:
    """``np.arccos`` with clamping to [-1, 1] to avoid NaN from rounding."""
    return float(np.arccos(np.clip(float(x), -1.0, 1.0)))


def _round_abs(z: complex, decimals: int = 6) -> float:
    """Round the magnitude of a complex number to ``decimals`` places."""
    return float(np.round(np.absolute(z), decimals))


# ---------------------------------------------------------------------------
# Angle extraction
# ---------------------------------------------------------------------------
def _extract_angles(unitary: NDArray) -> DecompositionAngles:
    """Decompose a 3x3 SU(3) matrix into nine canonical angles.

    The canonical form is
       unitary = u_d(\\phi_6, \\phi_5, \\phi_4) \\cdot
                r_{01}(\\phi_3, \\theta_3) \\cdot
                r_{12}(\\phi_2, \\theta_2) \\cdot
                r_{01}(\\phi_1, \\theta_1)

    Parameters
    ----------
    unitary : ndarray
        A 3x3 matrix in (or close to) SU(3).

    Returns
    -------
    DecompositionAngles
        Named tuple with fields ``theta1, theta2, theta3, phi1, ..., phi6``.

    Notes
    -----
    The branching on ``|unitary[2,2]|`` handles the three degenerate cases where
    the standard formulae become singular.
    """
    abs_u22 = _round_abs(unitary[2, 2])

    if abs_u22 == 1.0:
        abs_u00 = _round_abs(unitary[0, 0])
        if abs_u00 != 0.0:
            theta_1 = phi_1 = theta_2 = phi_2 = 0.0
            phi_4 = float(np.angle(unitary[2, 2]))
            phi_5 = float(np.angle(unitary[1, 1]))
            phi_6 = float(np.angle(unitary[0, 0]))
            phi_3 = float(np.angle(unitary[1, 0])) - phi_5 + _PI / 2
            theta_3 = 2 * _safe_arccos(_round_abs(unitary[1, 1]))
        else:
            theta_1 = phi_1 = theta_2 = phi_2 = phi_3 = 0.0
            theta_3 = 2 * _safe_arccos(_round_abs(unitary[1, 1]))
            phi_4 = float(np.angle(unitary[2, 2]))
            phi_6 = float(np.angle(unitary[0, 1])) + phi_3 + _PI / 2
            phi_5 = float(np.angle(unitary[1, 0])) - phi_3 + _PI / 2

    elif abs_u22 == 0.0:
        theta_1 = 2 * _safe_arccos(_round_abs(unitary[2, 1]))
        theta_2 = _PI
        theta_3 = 2 * _safe_arccos(_round_abs(unitary[1, 2]))
        phi_1 = phi_2 = phi_3 = 0.0
        phi_4 = phi_5 = phi_6 = 0.0

        abs_u20 = _round_abs(unitary[2, 0])
        abs_u10 = _round_abs(unitary[1, 0])
        abs_u00 = _round_abs(unitary[0, 0])

        if abs_u00 != 0.0:
            phi_4 = float(np.angle(unitary[2, 1])) + _PI / 2
            phi_5 = float(np.angle(unitary[1, 2])) + _PI / 2
            phi_6 = float(np.angle(unitary[1, 1]))
        elif abs_u10 != 0.0:
            phi_4 = float(np.angle(unitary[2, 1])) + _PI / 2
            phi_5 = float(np.angle(unitary[1, 0])) + _PI / 2
            phi_6 = float(np.angle(-unitary[0, 2]))
        elif abs_u20 != 0.0:
            phi_4 = float(np.angle(-unitary[2, 0]))
            abs_u02 = _round_abs(unitary[0, 2])
            if abs_u02 != 0.0:
                phi_5 = float(np.angle(-unitary[1, 1]))
                phi_6 = float(np.angle(-unitary[0, 2]))
            else:
                phi_5 = float(np.angle(unitary[1, 2])) + _PI / 2
                phi_6 = float(np.angle(unitary[0, 1])) + _PI / 2

    else:
        phi_4 = float(np.angle(unitary[2, 2]))
        theta_2 = 2 * _safe_arccos(_round_abs(unitary[2, 2]))
        sin_half_theta2 = np.sin(theta_2 / 2)
        phi_2 = float(np.angle(unitary[2, 1])) - phi_4 + _PI / 2
        phi_1 = float(np.angle(-unitary[2, 0])) - phi_2 - phi_4
        theta_1 = 2 * _safe_arccos(
            float(np.clip(np.round(np.absolute(unitary[2, 1]) / sin_half_theta2, 6), -1.0, 1.0))
        )
        theta_3 = 2 * _safe_arccos(
            float(np.clip(np.round(np.absolute(unitary[1, 2]) / sin_half_theta2, 6), -1.0, 1.0))
        )
        phi_5 = float(np.angle(unitary[1, 2])) + phi_2 + _PI / 2
        phi_3 = float(
            np.angle(
                np.cos(theta_1 / 2) * np.cos(theta_2 / 2) * np.cos(theta_3 / 2)
                - unitary[1, 1] * np.exp(-1j * phi_5)
            )
        ) + phi_1
        phi_6 = float(np.angle(-unitary[0, 2])) + phi_3 + phi_2

    return DecompositionAngles(
        theta_1, theta_2, theta_3, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6,
    )


# ---------------------------------------------------------------------------
# Decomposition class
# ---------------------------------------------------------------------------
class SU3Decomposition:
    """Decomposition of an arbitrary 3x3 unitary into native qutrit rotations.

    Parameters
    ----------
    su3 : ndarray
        A 3x3 unitary matrix.
    qutrit_index : int
        Index of the qutrit on which the decomposed gates will act.
    n_qutrits : int
        Total number of qutrits in the parent circuit.

    Raises
    ------
    ValueError
        If ``su3`` does not have shape ``(3, 3)`` or is not unitary.
    """

    def __init__(self, su3: NDArray, qutrit_index: int, n_qutrits: int) -> None:
        if su3.shape != (3, 3):
            raise ValueError(
                f"su3 must have shape (3, 3); got {su3.shape}."
            )
        if not np.allclose(su3 @ su3.conj().T, np.eye(3), atol=1e-6):
            raise ValueError("su3 is not unitary.")
        self.su3: NDArray = su3
        self.qutrit_index: int = qutrit_index
        self.n_qutrits: int = n_qutrits
        self.angles: DecompositionAngles = _extract_angles(self.su3)

    def diagonal_phase(self) -> NDArray:
        """Return the diagonal unitary u_d(phi_6, phi_5, phi_4)."""
        return u_d(self.angles.phi6, self.angles.phi5, self.angles.phi4)

    def rotation_01_theta3(self) -> NDArray:
        """Return r_01(phi_3, theta_3)."""
        return r01(phi=self.angles.phi3, theta=self.angles.theta3)

    def rotation_01_theta1(self) -> NDArray:
        """Return r_01(phi_1, theta_1)."""
        return r01(phi=self.angles.phi1, theta=self.angles.theta1)

    def rotation_12_theta2(self) -> NDArray:
        """Return r_12(phi_2, theta_2)."""
        return r12(phi=self.angles.phi2, theta=self.angles.theta2)

    def reconstruct(self) -> NDArray:
        """Multiply the four factors back together; should equal ``su3``."""
        return (  # type: ignore[no-any-return]
            self.diagonal_phase()
            @ self.rotation_01_theta3()
            @ self.rotation_12_theta2()
            @ self.rotation_01_theta1()
        )

    def to_native(self) -> NativeDecomposition:
        """Return the decomposition as virtual-Z phases and native instructions.

        Returns
        -------
        NativeDecomposition
            ``phases``: cumulative virtual-Z phase advances on {01} and {12}.
            ``instructions``: sequence of native g01/g12 rotations.
        """
        a = self.angles
        phase01 = a.phi6 - a.phi5
        phase12 = a.phi5 - a.phi4
        instructions = [
            Instruction(
                gate_type="g01",
                first_qutrit=self.qutrit_index,
                second_qutrit=None,
                n_qutrit=self.n_qutrits,
                parameter=[a.theta1, a.phi1],
            ),
            Instruction(
                gate_type="g12",
                first_qutrit=self.qutrit_index,
                second_qutrit=None,
                n_qutrit=self.n_qutrits,
                parameter=[a.theta2, a.phi2],
            ),
            Instruction(
                gate_type="g01",
                first_qutrit=self.qutrit_index,
                second_qutrit=None,
                n_qutrit=self.n_qutrits,
                parameter=[a.theta3, a.phi3],
            ),
        ]
        return NativeDecomposition(np.array([phase01, phase12]), instructions)

    def to_circuit(self) -> QutritCircuit:
        """Return a fresh QutritCircuit realizing this decomposition."""
        from qutritium.gates import G01, G12, Ud
        qc = QutritCircuit(n_qutrit=self.n_qutrits, initial_state=None)
        a = self.angles
        qc.append(G01(a.theta1, a.phi1), first_qutrit=self.qutrit_index)
        qc.append(G12(a.theta2, a.phi2), first_qutrit=self.qutrit_index)
        qc.append(G01(a.theta3, a.phi3), first_qutrit=self.qutrit_index)
        qc.append(Ud(a.phi6, a.phi5, a.phi4), first_qutrit=self.qutrit_index)
        return qc

    def __str__(self) -> str:
        return (
            f"U_diagonal:\n{self.diagonal_phase()}\n"
            f"R_theta1:\n{self.rotation_01_theta1()}\n"
            f"R_theta2:\n{self.rotation_12_theta2()}\n"
            f"R_theta3:\n{self.rotation_01_theta3()}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


__all__ = [
    "DecompositionAngles",
    "NativeDecomposition",
    "SU3Decomposition",
]
