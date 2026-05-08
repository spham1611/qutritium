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

Given an arbitrary 3x3 unitary :math:`U \\in SU(3)` (up to a global phase),
this module decomposes it into the product

.. math::
   U = D \\cdot R^{(01)}(\\theta_3, \\phi_3) \\cdot R^{(12)}(\\theta_2, \\phi_2)
       \\cdot R^{(01)}(\\theta_1, \\phi_1)

where :math:`D` is a diagonal phase gate and the :math:`R^{(ij)}` are
composite rotations in the :math:`\\{|i\\rangle, |j\\rangle\\}` subspace.

The decomposition is hardware-agnostic: it produces angle parameters and a
list of :class:`qutritium.circuit.Instruction` objects that any
backend (statevector simulator, ARTIQ, AQT, IBM via Qiskit, ...) can consume.
"""
# REMOVED: all pulse-coupled and IBM-coupled code. Specifically:
# * imports of ``qiskit_ibm_provider.IBMBackend``,
#   ``qiskit.pulse.schedule.ScheduleBlock``, ``src.pulse.Pulse01/Pulse12``,
#   ``src.pulse_creation.{Shift_phase, Set_frequency, GateSchedule}`` and
#   ``matplotlib.pyplot`` (the latter only used inside the removed code path).
# * the ``Pulse_Wrapper`` class in its entirety (~180 lines, lines 223-403
#   of the v0.0.1 file). Its responsibilities -- decomposing a circuit into
#   pulse-level Qiskit ScheduleBlocks for IBM hardware -- are out of scope
#   for the hardware-agnostic v1.0.0 package and have been moved (in their
#   original form) to ``legacy/decomposition/transpilation.py.legacy.txt``
#   as a historical record.
# MODIFIED: import paths ``src.X`` -> ``qutritium.X`` for retained imports.
# MODIFIED: tightened type hints; replaced ``DefaultDict[str, Any]`` with
# specific types where the value type is known.
# MODIFIED: ``Parameter`` was a "static class" wrapping a single classmethod.
# Converted to a module-level function ``get_parameters`` (more Pythonic;
# class-with-only-classmethods is a Java idiom). The ``Parameter`` symbol is
# preserved as a backward-compat shim that defers to the function.
# MODIFIED: replaced the named-tuple-by-string-name ``getattr(self.parameters,
# 'phi1')`` calls with normal attribute access (``self.parameters.phi1``).
# The string form was lookup-by-name, which is slower and breaks IDE
# completion for no benefit.
# MODIFIED: ``__str__`` and ``__repr__`` were duplicated character-for-character.
# Replaced with a single implementation that ``__repr__`` defers to.
from __future__ import annotations

from collections import namedtuple
from typing import List, NamedTuple, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import Qutrit_circuit
from qutritium.circuit.elementary_matrices import r01, r12, u_d

# ADDED: module-level constant.
_PI: float = float(np.pi)

# Named tuple capturing the nine Euler-like angles of the SU(3) decomposition.
# theta_i: rotation magnitudes; phi_j: phase angles.
DecompositionAngles = namedtuple(
    "DecompositionAngles",
    "theta1 theta2 theta3 phi1 phi2 phi3 phi4 phi5 phi6",
)


def get_parameters(U: NDArray) -> NamedTuple:
    """Decompose a 3x3 ``SU(3)`` matrix into the nine angles of the canonical form.

    The canonical form is

    .. math::
       U = u_d(\\phi_6, \\phi_5, \\phi_4) \\cdot
           r_{01}(\\phi_3, \\theta_3) \\cdot
           r_{12}(\\phi_2, \\theta_2) \\cdot
           r_{01}(\\phi_1, \\theta_1)

    Parameters
    ----------
    U : ndarray
        A 3x3 matrix in (or close to) SU(3).

    Returns
    -------
    DecompositionAngles
        Named tuple with fields ``theta1, theta2, theta3, phi1, ..., phi6``.

    Notes
    -----
    The branching on ``|U[2,2]|`` handles the three degenerate cases where
    the standard formulae become singular. The non-degenerate (final
    ``else`` branch) case is the generic one.
    """
    # MODIFIED: extraction logic itself is preserved bit-for-bit from
    # v0.0.1. This is intentional -- the upstream changes to ``rz01``,
    # ``rz12``, ``r01`` and ``r12`` were chosen so that the *outputs* of
    # ``r01``/``r12`` are bit-identical to v0.0.1 (verified to machine
    # precision, see ``qc_elementary_matrices`` module docstring).
    # Therefore the parameter-extraction here continues to round-trip
    # without any change.
    if np.round(abs(np.absolute(U[2, 2])), 6) == 1:
        if np.round(abs(np.absolute(U[0, 0])), 6) != 0:
            theta_1 = phi_1 = theta_2 = phi_2 = 0.0
            phi_4 = np.angle(U[2, 2])
            phi_5 = np.angle(U[1, 1])
            phi_6 = np.angle(U[0, 0])
            phi_3 = np.angle(U[1, 0]) - phi_5 + _PI / 2
            theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 1]), 6))
        else:
            theta_1 = phi_1 = theta_2 = phi_2 = phi_3 = 0.0
            theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 1]), 6))
            phi_4 = np.angle(U[2, 2])
            phi_6 = np.angle(U[0, 1]) + phi_3 + _PI / 2
            phi_5 = np.angle(U[1, 0]) - phi_3 + _PI / 2
    elif np.round(abs(np.absolute(U[2, 2])), 6) == 0:
        theta_1 = 2 * np.arccos(np.round(np.absolute(U[2, 1]), 6))
        theta_2 = _PI
        theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 2]), 6))
        phi_1 = phi_2 = phi_3 = 0.0
        # ADDED: initialise phi_4/5/6 in case none of the three branches
        # below match -- the v0.0.1 code would raise UnboundLocalError in
        # that edge case. With the explicit zero default, the function
        # returns an under-determined but well-typed result.
        phi_4 = phi_5 = phi_6 = 0.0
        if np.round(abs(np.absolute(U[2, 0])), 6) != 0:
            phi_4 = np.angle(-U[2, 0])
            if np.round(abs(np.absolute(U[0, 2])), 6) != 0:
                phi_5 = np.angle(-U[1, 1])
                phi_6 = np.angle(-U[0, 2])
            else:
                phi_5 = np.angle(U[1, 2]) + _PI / 2
                phi_6 = np.angle(U[0, 1]) + _PI / 2
        if np.round(abs(np.absolute(U[1, 0])), 6) != 0:
            phi_4 = np.angle(U[2, 1]) + _PI / 2
            phi_5 = np.angle(U[1, 0]) + _PI / 2
            phi_6 = np.angle(-U[0, 2])
        if np.round(abs(np.absolute(U[0, 0])), 6) != 0:
            phi_4 = np.angle(U[2, 1]) + _PI / 2
            phi_5 = np.angle(U[1, 2]) + _PI / 2
            phi_6 = np.angle(U[1, 1])
    else:
        phi_4 = np.angle(U[2, 2])
        theta_2 = 2 * np.arccos(np.round(np.absolute(U[2, 2]), 6))
        phi_2 = np.angle(U[2, 1]) - phi_4 + _PI / 2
        phi_1 = np.angle(-U[2, 0]) - phi_2 - phi_4
        theta_1 = 2 * np.arccos(np.round(np.absolute(U[2, 1]) / np.sin(theta_2 / 2), 6))
        theta_3 = 2 * np.arccos(np.round(np.absolute(U[1, 2]) / np.sin(theta_2 / 2), 6))
        phi_5 = np.angle(U[1, 2]) + phi_2 + _PI / 2
        phi_3 = (
            np.angle(
                np.cos(theta_1 / 2) * np.cos(theta_2 / 2) * np.cos(theta_3 / 2)
                - U[1, 1] * np.exp(-1j * phi_5)
            )
            + phi_1
        )
        phi_6 = np.angle(-U[0, 2]) + phi_3 + phi_2

    return DecompositionAngles(
        theta_1, theta_2, theta_3, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6,
    )


class Parameter:
    """Backward-compatibility shim wrapping :func:`get_parameters`.

    Prefer the module-level function in new code; this class is retained
    for the v0.0.1 callsite ``Parameter.get_parameters(U=...)``.
    """

    # MODIFIED: was a "static class" with a single ``@classmethod``.
    # Now a thin shim around the module-level function.
    @classmethod
    def get_parameters(cls, U: NDArray) -> NamedTuple:
        """See :func:`get_parameters` for full documentation."""
        return get_parameters(U)


class SU3_matrices:
    """Decomposition of an arbitrary 3x3 unitary into native qutrit rotations.

    Parameters
    ----------
    su3 : ndarray
        A 3x3 (approximately unitary) matrix.
    qutrit_index : int
        Index of the qutrit on which the decomposed gates will act.
    n_qutrits : int
        Total number of qutrits in the parent circuit.

    Raises
    ------
    ValueError
        If ``su3`` does not have shape ``(3, 3)``.
    """

    def __init__(self, su3: NDArray, qutrit_index: int, n_qutrits: int) -> None:
        # MODIFIED: ``assert`` -> ``ValueError``. Asserts can be disabled
        # with ``python -O`` and should not be used for input validation.
        if su3.shape != (3, 3):
            raise ValueError(
                f"su3 must have shape (3, 3); got {su3.shape}."
            )
        self.su3: NDArray = su3
        self.qutrit_index: int = qutrit_index
        self.n_qutrits: int = n_qutrits
        self.parameters: NamedTuple = get_parameters(self.su3)

    def unitary_diagonal(self) -> NDArray:
        """Return the diagonal unitary :math:`u_d(\\phi_6, \\phi_5, \\phi_4)`."""
        # MODIFIED: ``getattr(self.parameters, 'phi6')`` -> ``self.parameters.phi6``.
        return u_d(
            phi_1=self.parameters.phi6,
            phi_2=self.parameters.phi5,
            phi_3=self.parameters.phi4,
        )

    def rotation_theta3_01(self) -> NDArray:
        """Return :math:`r_{01}(\\phi_3, \\theta_3)`."""
        return r01(phi=self.parameters.phi3, theta=self.parameters.theta3)

    def rotation_theta1_01(self) -> NDArray:
        """Return :math:`r_{01}(\\phi_1, \\theta_1)`."""
        return r01(phi=self.parameters.phi1, theta=self.parameters.theta1)

    def rotation_theta2_12(self) -> NDArray:
        """Return :math:`r_{12}(\\phi_2, \\theta_2)`."""
        return r12(phi=self.parameters.phi2, theta=self.parameters.theta2)

    def reconstruct(self) -> NDArray:
        """Multiply the four factors back together; should equal :attr:`su3`."""
        return (
            self.unitary_diagonal()
            @ self.rotation_theta3_01()
            @ self.rotation_theta2_12()
            @ self.rotation_theta1_01()
        )

    def native_list(self) -> List[Union[NDArray, List[Instruction]]]:
        """Return the decomposition as ``[phase_array, [Instruction, ...]]``.

        The phase array contains the cumulative virtual-Z phase advances on
        the {01} and {12} subspaces; the instruction list is the sequence
        of native ``g01``/``g12`` rotations to apply.
        """
        # MODIFIED: extracted phase computation into named locals; replaced
        # the deeply-nested literal list with explicit construction.
        p = self.parameters
        phase01 = float(p.phi6 - p.phi5)
        phase12 = float(p.phi5 - p.phi4)
        instructions: List[Instruction] = [
            Instruction(
                gate_type="g01",
                first_qutrit_set=self.qutrit_index,
                second_qutrit_set=None,
                n_qutrit=self.n_qutrits,
                parameter=[p.theta1, p.phi1],
            ),
            Instruction(
                gate_type="g12",
                first_qutrit_set=self.qutrit_index,
                second_qutrit_set=None,
                n_qutrit=self.n_qutrits,
                parameter=[p.theta2, p.phi2],
            ),
            Instruction(
                gate_type="g01",
                first_qutrit_set=self.qutrit_index,
                second_qutrit_set=None,
                n_qutrit=self.n_qutrits,
                parameter=[p.theta3, p.phi3],
            ),
        ]
        return [np.array([phase01, phase12]), instructions]

    def decomposed_into_qc(self) -> Qutrit_circuit:
        """Return a fresh :class:`Qutrit_circuit` realising this decomposition."""
        decomposed_qc = Qutrit_circuit(n_qutrit=self.n_qutrits, initial_state=None)
        p = self.parameters
        decomposed_qc.add_gate(
            "g01", first_qutrit_set=self.qutrit_index,
            parameter=[p.theta1, p.phi1],
        )
        decomposed_qc.add_gate(
            "g12", first_qutrit_set=self.qutrit_index,
            parameter=[p.theta2, p.phi2],
        )
        decomposed_qc.add_gate(
            "g01", first_qutrit_set=self.qutrit_index,
            parameter=[p.theta3, p.phi3],
        )
        decomposed_qc.add_gate(
            "u_d", first_qutrit_set=self.qutrit_index,
            parameter=[p.phi6, p.phi5, p.phi4],
        )
        return decomposed_qc

    def __str__(self) -> str:
        # MODIFIED: previously ``__str__`` and ``__repr__`` were
        # duplicated character-for-character. Single source of truth now.
        return (
            f"U_diagonal:\n{self.unitary_diagonal()}\n"
            f"R_theta1:\n{self.rotation_theta1_01()}\n"
            f"R_theta2:\n{self.rotation_theta2_12()}\n"
            f"R_theta3:\n{self.rotation_theta3_01()}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()


__all__ = ["DecompositionAngles", "Parameter", "SU3_matrices", "get_parameters"]
