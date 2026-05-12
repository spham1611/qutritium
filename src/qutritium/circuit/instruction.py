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
Single-gate instruction class and the registry of supported gate names.

An :class:`Instruction` represents one application of a qutrit gate inside
:class:`qutritium.circuit.qutrit_circuit.Qutrit_circuit`. It holds the gate type,
target qutrit indices, parameters, and a precomputed effect matrix that the
simulator can apply to a statevector.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.utils import multi_matrix_form, single_matrix_form

# ADDED: type alias for the union of valid gate-type names.
GATE_SET: frozenset[str] = frozenset({
    "identity",
    "x_plus", "x_minus",
    "sdg",
    "tdg",
    "CNOT",
    "g01", "g12",
    "x01", "x02", "x12",
    "y01", "y02", "y12",
    "z01", "z12",
    "rx01", "rx12",
    "ry01", "ry12",
    "rz01", "rz12",
    "u_d",
    "hdm",
    "u_ft",
})


class Instruction:
    """A single gate application within a qutrit circuit.

    An :class:`Instruction` precomputes the full ``3**n_qutrit``-dimensional
    effect matrix at construction time so that the simulator can apply each
    gate by a single matrix-vector multiplication.

    Parameters
    ----------
    gate_type : str
        Name of the gate. Must be in :data:`GATE_SET` unless ``custom`` is
        ``True``.
    n_qutrit : int
        Total number of qutrits in the parent circuit.
    first_qutrit_set : int
        Index of the (first / target) qutrit. Must satisfy
        ``0 <= first_qutrit_set < n_qutrit``.
    second_qutrit_set : int or None, optional
        Index of the second (control) qutrit for two-qutrit gates. ``None``
        for single-qutrit gates.
    parameter : sequence of float, optional
        Gate parameters for parametrized gates (e.g. rotation angles).
    inverse : bool, optional
        If ``True``, the Hermitian conjugate of the gate matrix is used.
    custom : bool, optional
        If ``True``, ``custom_matrix`` is taken as the gate matrix verbatim
        and ``gate_type`` is treated as a free-form label (not validated
        against :data:`GATE_SET`).
    custom_matrix : ndarray, optional
        The custom 3x3 unitary, required when ``custom`` is ``True``.

    Raises
    ------
    ValueError
        If ``gate_type`` is not in :data:`GATE_SET` (and ``custom`` is
        ``False``), or if a custom gate is requested without a matrix.
        If ''custom'' and ''custom_matrix'' is not None, the matrix dimensions
        should be 3x3 or 9x9 for two qutrits.
    IndexError
        If ``first_qutrit_set`` or ``second_qutrit_set`` is out of range.
    """

    def __init__(
        self,
        gate_type: str,
        n_qutrit: int,
        first_qutrit_set: int,
        second_qutrit_set: int | None = None,
        parameter: Sequence[float] | None = None,
        inverse: bool = False,
        custom: bool = False,
        custom_matrix: NDArray | None = None,
    ) -> None:
        # Validate qutrit range
        if not 0 <= first_qutrit_set < n_qutrit:
            raise IndexError(
                f"first_qutrit_set={first_qutrit_set} is out of range "
                f"[0, {n_qutrit})."
            )
        if second_qutrit_set is not None and not 0 <= second_qutrit_set < n_qutrit:
            raise IndexError(
                f"second_qutrit_set={second_qutrit_set} is out of range "
                f"[0, {n_qutrit})."
            )

        self._type: str = gate_type
        self.n_qutrit: int = n_qutrit
        self.qutrit_dimension: int = 3 ** n_qutrit
        self.first_qutrit: int = first_qutrit_set
        self.second_qutrit: int | None = second_qutrit_set
        self.parameter: Sequence[float] | None = parameter
        self._is_inverse: bool = inverse
        self._is_custom: bool = custom
        self._is_two_qutrit_gate: bool = second_qutrit_set is not None

        if not custom:
            self._verify_gate()
        elif custom_matrix is None:
            raise ValueError("custom=True requires custom_matrix to be supplied.")

        # Validate custom matrix dimensions
        if custom and custom_matrix is not None:
            if self._is_two_qutrit_gate:
                expected = 9  # two-qutrit gate: 3^2 x 3^2
            else:
                expected = 3  # single-qutrit gate: 3 x 3
            if custom_matrix.shape != (expected, expected):
                raise ValueError(
                    f"custom_matrix has shape {custom_matrix.shape}; "
                    f"expected ({expected}, {expected}) for a "
                    f"{'two' if self._is_two_qutrit_gate else 'single'}-qutrit gate."
                )

        base_matrix: NDArray

        if self._is_two_qutrit_gate:
            base_matrix = multi_matrix_form(
                gate_type=self._type,
                first_index=self.first_qutrit,
                second_index=self.second_qutrit,  # type: ignore[arg-type]
            )
        elif custom:
            assert custom_matrix is not None
            base_matrix = custom_matrix
        else:
            base_matrix = single_matrix_form(
                gate_type=self._type, parameter=self.parameter,
            )

        self.gate_matrix: NDArray = (
            np.asarray(base_matrix).conj().T if self._is_inverse
            else np.asarray(base_matrix)
        )
        # Compute local matrix and expands into full-register matrix that acts on all qutrits in our circuit
        self._effect_matrix: NDArray = self._effect()

    def _effect(self) -> NDArray:
        """Compute the full ``3**n``-dim effect matrix from the local gate."""
        if not self._is_two_qutrit_gate:
            if self.n_qutrit == 1:
                return self.gate_matrix
            if self.first_qutrit == 0:
                effect_matrix = np.einsum(
                    "ik,jl",
                    self.gate_matrix,
                    np.eye(int(self.qutrit_dimension / 3)),
                ).reshape(self.qutrit_dimension, self.qutrit_dimension)
            else:
                # Calculate tensor product on the left for example I_left ⊗ gate_of_qutrit, then take
                # that ⊗ I_right
                effect_matrix = np.einsum(
                    "ik,jl",
                    np.eye(3 ** self.first_qutrit),
                    self.gate_matrix,
                ).reshape(3 ** (self.first_qutrit + 1), 3 ** (self.first_qutrit + 1))
                effect_matrix = np.einsum(
                    "ik,jl",
                    effect_matrix,
                    np.eye(3 ** (self.n_qutrit - self.first_qutrit - 1)),
                ).reshape(self.qutrit_dimension, self.qutrit_dimension)
            return effect_matrix  # type: ignore[no-any-return]

        # Two-qutrit gate path. Same calculation as the previous one
        assert self.second_qutrit is not None
        second = self.second_qutrit
        left = min(self.first_qutrit, second)
        right = max(self.first_qutrit, second)
        if left == 0:
            effect_matrix = np.einsum(
                "ik,jl",
                self.gate_matrix,
                np.eye(3 ** (self.n_qutrit - right - 1)),
            ).reshape(self.qutrit_dimension, self.qutrit_dimension)
        else:
            effect_matrix = np.einsum(
                "ik,jl", np.eye(3 ** left), self.gate_matrix,
            ).reshape(3 ** (right + 1), 3 ** (right + 1))
            effect_matrix = np.einsum(
                "ik,jl",
                effect_matrix,
                np.eye(3 ** (self.n_qutrit - right - 1)),
            ).reshape(self.qutrit_dimension, self.qutrit_dimension)
        return effect_matrix  # type: ignore[no-any-return]

    def _verify_gate(self) -> None:
        """Verify ``self._type`` is a recognized gate name."""
        if self._type not in GATE_SET:
            raise ValueError(
                f"Gate type {self._type!r} is not in GATE_SET. "
                f"Supported gates: {sorted(GATE_SET)}."
            )

    @property
    def effect_matrix(self) -> NDArray:
        """The ``3**n``-dim matrix representing this gate on the full register."""
        return self._effect_matrix

    def type(self) -> str:
        """Return the gate-type string."""
        return self._type

    def print(self) -> None:
        """Print a one-line human-readable description of this instruction."""
        if not self._is_two_qutrit_gate:
            if self.parameter is None:
                print(f"Gate {self._type}, acting qutrit: {self.first_qutrit}")
            else:
                print(
                    f"Gate {self._type} with parameter {list(self.parameter)}, "
                    f"acting qutrit: {self.first_qutrit}"
                )
        else:
            print(
                f"Gate {self._type}, acting qutrit: {self.first_qutrit}, "
                f"control qutrit: {self.second_qutrit}"
            )

    def inverse(self) -> Instruction:
        """Return a new Instruction that is the Hermitian conjugate of this one."""
        return Instruction(
            gate_type=self._type,
            n_qutrit=self.n_qutrit,
            first_qutrit_set=self.first_qutrit,
            second_qutrit_set=self.second_qutrit,
            parameter=self.parameter,
            inverse=not self._is_inverse,
            custom=self._is_custom,
            custom_matrix=self.gate_matrix if self._is_custom else None,
        )
