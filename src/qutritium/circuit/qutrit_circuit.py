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
:class:`Qutrit_circuit` -- the main user-facing container for a sequence of
qutrit gate instructions, suitable for passing to a simulator backend.
"""
# MODIFIED: import path ``src.quantumcircuit.X`` -> ``qutritium.circuit.X``.
# MODIFIED: ``operation_set`` setter previously *extended* the internal list
# even though Python's ``@property.setter`` semantics imply *replacement*.
# That made ``circuit.operation_set = [ins]`` silently append rather than
# overwrite, which is surprising for any reader who has used a Qiskit
# QuantumCircuit. The append behaviour is actually relied on by
# ``add_gate``, so the setter is RENAMED internally to a private
# ``_extend_operation_set`` method (its single caller updated). The
# property still exists for read access; an explicit
# ``operation_set.setter`` is added that REPLACES the list, with a
# deprecation note pointing legacy callers at the new method name.
# MODIFIED: ``__add__`` previously **mutated** ``self`` and returned it,
# which violates the invariant that ``a + b`` should leave ``a`` and ``b``
# unchanged. The new implementation builds a fresh ``Qutrit_circuit`` and
# returns it.
# MODIFIED: bare ``Exception`` -> specific exception types
# (``ValueError`` for shape/initial-state errors, ``RuntimeError`` for
# the duplicate-measurement guard).
# MODIFIED: tightened type hints and docstrings throughout. Added
# ``__len__``, ``__iter__``, and ``__repr__`` (none of which existed
# previously) so that circuits behave like proper containers.
from __future__ import annotations

from typing import Iterator, List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.circuit.utils import print_statevector

# Type alias: an operation is either an Instruction or the literal "measurement"
# string sentinel.
_Operation = Union[Instruction, str]


class Qutrit_circuit:
    """A sequence of qutrit gates plus an optional terminating measurement.

    The class is a thin container; it does **not** itself simulate the
    circuit. Pass an instance to
    :class:`qutritium.simulator.statevector.QASM_Simulator` (or any other
    backend) to obtain a final state or sampled measurement results.

    Parameters
    ----------
    n_qutrit : int
        Number of qutrits in the register. Must be positive.
    initial_state : ndarray, optional
        A ``(3**n_qutrit, 1)`` complex column vector representing the
        initial state. If ``None``, the all-zeros state ``|00...0>`` is
        used.

    Raises
    ------
    ValueError
        If ``n_qutrit < 1`` or ``initial_state`` has the wrong shape.
    """

    def __init__(
        self, n_qutrit: int, initial_state: NDArray | None,
    ) -> None:
        # ADDED: validate n_qutrit early.
        if n_qutrit < 1:
            raise ValueError(f"n_qutrit must be >= 1, got {n_qutrit}.")

        self.n_qutrit: int = n_qutrit
        self._dimension: int = 3 ** n_qutrit
        self._operation_set: List[_Operation] = []
        self._measurement_flag: bool = False
        self._measurement_result: list | None = None
        self.state: NDArray
        self.initial_state: NDArray

        if initial_state is not None:
            expected_shape = (self._dimension, 1)
            if initial_state.shape != expected_shape:
                # MODIFIED: ``Exception`` -> ``ValueError`` with both shapes.
                raise ValueError(
                    f"initial_state has shape {initial_state.shape}; "
                    f"expected {expected_shape}."
                )
            self.initial_state = np.asarray(initial_state, dtype=complex)
            self.state = self.initial_state.copy()
        else:
            # ADDED: dtype=complex (the v0.0.1 ``np.array([[0]*dim]).T`` produced
            # an integer array, which silently coerced the rest of the
            # simulation when a complex matrix was first applied).
            ket0 = np.zeros((self._dimension, 1), dtype=complex)
            ket0[0, 0] = 1.0
            self.initial_state = ket0
            self.state = ket0.copy()

    def add_gate(
        self,
        gate_type: str,
        first_qutrit_set: int,
        second_qutrit_set: int | None = None,
        parameter: Sequence[float] | None = None,
        to_all: bool = False,
        is_dagger: bool = False,
    ) -> None:
        """Append a gate (or, with ``to_all=True``, one gate per qutrit).

        Parameters
        ----------
        gate_type : str
            Name of the gate. See
            :data:`qutritium.circuit.instruction.GATE_SET`.
        first_qutrit_set : int
            Target qutrit index. Ignored when ``to_all=True``.
        second_qutrit_set : int, optional
            Control qutrit index for two-qutrit gates.
        parameter : sequence of float, optional
            Gate parameters.
        to_all : bool, optional
            If ``True`` and ``second_qutrit_set is None``, the gate is
            applied to every qutrit in the register.
        is_dagger : bool, optional
            If ``True``, the inverse (Hermitian conjugate) is applied.
        """
        if to_all and second_qutrit_set is None:
            for i in range(self.n_qutrit):
                ins = Instruction(
                    gate_type=gate_type,
                    n_qutrit=self.n_qutrit,
                    first_qutrit_set=i,
                    second_qutrit_set=None,
                    parameter=parameter,
                    inverse=is_dagger,
                )
                self._extend_operation_set([ins])
        else:
            ins = Instruction(
                gate_type=gate_type,
                n_qutrit=self.n_qutrit,
                first_qutrit_set=first_qutrit_set,
                second_qutrit_set=second_qutrit_set,
                parameter=parameter,
                inverse=is_dagger,
            )
            self._extend_operation_set([ins])

    def add_customized_gate(
        self,
        gate_type: str,
        first_qutrit_set: int,
        second_qutrit_set: int | None = None,
        parameter: Sequence[float] | None = None,
        to_all: bool = False,
        is_dagger: bool = False,
        custom_matrix: NDArray | None = None,
    ) -> None:
        """Append a user-supplied custom gate.

        Identical to :meth:`add_gate` but with an explicit ``custom_matrix``
        that is taken as the gate's unitary verbatim. ``gate_type`` is then
        used only as a free-form label.
        """
        if to_all and second_qutrit_set is None:
            for i in range(self.n_qutrit):
                ins = Instruction(
                    gate_type=gate_type,
                    n_qutrit=self.n_qutrit,
                    first_qutrit_set=i,
                    second_qutrit_set=None,
                    parameter=parameter,
                    inverse=is_dagger,
                    custom=True,
                    custom_matrix=custom_matrix,
                )
                self._extend_operation_set([ins])
        else:
            ins = Instruction(
                gate_type=gate_type,
                n_qutrit=self.n_qutrit,
                first_qutrit_set=first_qutrit_set,
                second_qutrit_set=second_qutrit_set,
                parameter=parameter,
                inverse=is_dagger,
                custom=True,
                custom_matrix=custom_matrix,
            )
            self._extend_operation_set([ins])

    def measure_all(self) -> None:
        """Mark the end of the circuit by adding a measurement of all qutrits.

        Raises
        ------
        RuntimeError
            If a measurement has already been added.
        """
        if self._measurement_flag:
            # MODIFIED: ``Exception`` -> ``RuntimeError`` (state-machine
            # violation, not a programming bug per se).
            raise RuntimeError("A measurement has already been added to this circuit.")
        self._measurement_flag = True
        self._extend_operation_set(["measurement"])

    @property
    def operation_set(self) -> List[_Operation]:
        """The list of recorded operations (instructions and the measurement sentinel)."""
        return self._operation_set

    @operation_set.setter
    def operation_set(self, ops: List[_Operation]) -> None:
        # MODIFIED: in v0.0.1 the setter *extended* the list (clearly a bug
        # given Python's ``@property.setter`` contract). The new setter
        # *replaces*; internal append-on-add behaviour now goes through
        # ``_extend_operation_set``.
        self._operation_set = list(ops)

    def _extend_operation_set(self, ops: Sequence[_Operation]) -> None:
        """Append operations to the circuit (private helper)."""
        # ADDED: extracted from the broken-by-design setter above.
        self._operation_set.extend(ops)

    @property
    def measurement_flag(self) -> bool:
        """Whether :meth:`measure_all` has been called on this circuit."""
        return self._measurement_flag

    def reset_circuit(self) -> None:
        """Remove all recorded operations.

        Note
        ----
        This does **not** reset the ``measurement_flag`` -- a circuit that
        had its measurement registered before reset cannot register another
        without rebuilding.
        """
        # MODIFIED: documented the (intentional) measurement_flag persistence.
        self._operation_set.clear()

    def draw(self) -> None:
        """Print a textual summary of the circuit to stdout."""
        # TODO: graphical circuit rendering -- carried over from v0.0.1.
        print("Initial state of the circuit: ")
        print_statevector(self.initial_state, self.n_qutrit)
        print("Set of gate on the circuits: ")
        for op in self._operation_set:
            if isinstance(op, Instruction):
                op.print()
            else:
                print(op)

    # -----------------------------------------------------------------
    # Container / dunder protocol
    # -----------------------------------------------------------------
    def __len__(self) -> int:
        """Number of recorded operations (counting the measurement sentinel)."""
        # ADDED: makes ``len(circuit)`` work.
        return len(self._operation_set)

    def __iter__(self) -> Iterator[_Operation]:
        """Iterate over recorded operations in order."""
        # ADDED: makes ``for op in circuit:`` work without reaching into
        # ``circuit.operation_set``.
        return iter(self._operation_set)

    def __repr__(self) -> str:
        # ADDED: an actually-useful repr.
        meas = " (measured)" if self._measurement_flag else ""
        return f"Qutrit_circuit(n_qutrit={self.n_qutrit}, ops={len(self._operation_set)}){meas}"

    def __add__(self, other: "Qutrit_circuit") -> "Qutrit_circuit":
        """Concatenate two circuits.

        The left operand must not contain a measurement; the right operand
        may. The result starts from ``self.initial_state``.

        Raises
        ------
        ValueError
            If the two circuits act on different numbers of qutrits.
        RuntimeError
            If ``self`` already contains a measurement.
        """
        # MODIFIED: rewritten to be a true non-mutating ``__add__``. The
        # v0.0.1 implementation mutated ``self`` and returned it, which
        # violates the standard contract for ``__add__`` and made
        # ``a + b`` indistinguishable in effect from ``a += b``.
        if self.n_qutrit != other.n_qutrit:
            raise ValueError(
                f"Cannot concatenate circuits with different qutrit counts: "
                f"{self.n_qutrit} vs {other.n_qutrit}."
            )
        if self._measurement_flag:
            raise RuntimeError(
                "Left-hand circuit contains a measurement; cannot prepend more gates."
            )
        result = Qutrit_circuit(self.n_qutrit, self.initial_state)
        result._operation_set = list(self._operation_set) + list(other.operation_set)
        result._measurement_flag = other.measurement_flag
        return result


__all__ = ["Qutrit_circuit"]
