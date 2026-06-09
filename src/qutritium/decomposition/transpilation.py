# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""SU(3) -> native qutrit rotations via the Vitanov (2012) decomposition.

See ``SU3Decomposition`` docstring for the factor sequence and reference.
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
    """Nine angles from the SU(3) decomposition."""
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
    """Native decomposition result: phases + instructions."""
    phases: NDArray
    instructions: list[Instruction]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _safe_arccos(x: float) -> float:
    """arccos with clamping to avoid NaN."""
    return float(np.arccos(np.clip(float(x), -1.0, 1.0)))


def _round_abs(z: complex, decimals: int = 6) -> float:
    """Round |z| to given decimal places."""
    return float(np.round(np.absolute(z), decimals))


# ---------------------------------------------------------------------------
# Angle extraction
# ---------------------------------------------------------------------------
def _extract_angles(unitary: NDArray) -> DecompositionAngles:
    """Extract nine canonical angles from a 3x3 unitary. Branches on |U[2,2]|."""
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
    """Decompose an SU(3) unitary into native qutrit rotations.

    Implements the nine-angle decomposition of Vitanov (2012):

        U = u_d(phi6, phi5, phi4) . r01(phi3, theta3) . r12(phi2, theta2) . r01(phi1, theta1)

    where ``r01``, ``r12`` are composite subspace rotations and ``u_d``
    is a diagonal phase. Transpiles arbitrary single-qutrit unitaries
    onto a native gate set such as trapped-ion ``g01``/``g12`` plus
    virtual-Z phases.

    References
    ----------
    Vitanov, N. V. (2012). Synthesis of arbitrary SU(3) transformations of
    atomic qutrits. Phys. Rev. A 85, 032331.

    Examples
    --------
    >>> from qutritium.gates import H3
    >>> import numpy as np
    >>> dec = SU3Decomposition(H3().matrix(), qutrit_index=0, n_qutrits=1)
    >>> np.allclose(dec.reconstruct(), H3().matrix())
    True
    """

    def __init__(self, su3: NDArray, qutrit_index: int, n_qutrits: int) -> None:
        """Construct an SU(3) decomposition.

        Parameters
        ----------
        su3 : NDArray
            Shape ``(3, 3)``. Must be unitary to within ``atol=1e-8``.
        qutrit_index : int
            Index of the qutrit this unitary acts on.
        n_qutrits : int
            Total number of qutrits in the target register.

        Raises
        ------
        ValueError
            If ``su3`` is not ``(3, 3)`` or fails the unitarity check.
        """
        if su3.shape != (3, 3):
            raise ValueError(
                f"su3 must have shape (3, 3); got {su3.shape}."
            )
        if not np.allclose(su3 @ su3.conj().T, np.eye(3), atol=1e-8):
            raise ValueError("su3 is not unitary.")
        self.su3: NDArray = su3
        self.qutrit_index: int = qutrit_index
        self.n_qutrits: int = n_qutrits
        self.angles: DecompositionAngles = _extract_angles(self.su3)

    def diagonal_phase(self) -> NDArray:
        """u_d(phi6, phi5, phi4) factor."""
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
        """Multiply the four decomposed factors. Equals ``self.su3`` up to
        floating-point error.

        Returns
        -------
        NDArray
            Shape ``(3, 3)`` complex matrix.
        """
        return (  # type: ignore[no-any-return]
            self.diagonal_phase()
            @ self.rotation_01_theta3()
            @ self.rotation_12_theta2()
            @ self.rotation_01_theta1()
        )

    def to_native(self) -> NativeDecomposition:
        """Native ``g01``/``g12`` instructions plus two virtual-Z phases.

        The diagonal ``u_d`` factor is folded into two virtual-Z angles
        applied to the ``{|0>,|1>}`` and ``{|1>,|2>}`` subspaces.

        Returns
        -------
        NativeDecomposition
            ``(phases, instructions)`` with ``phases = [phase01, phase12]``
            and three ``Instruction`` objects in the order
            ``g01, g12, g01``.
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
        """Build a ``QutritCircuit`` from this decomposition.

        Emits the same factor sequence as ``to_native``, but as
        ``G01`` / ``G12`` / ``Ud`` gate objects appended to a fresh
        circuit.

        Returns
        -------
        QutritCircuit
            On ``n_qutrits`` qutrits; acts as ``self.su3`` on
            ``qutrit_index`` up to global phase.
        """
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
