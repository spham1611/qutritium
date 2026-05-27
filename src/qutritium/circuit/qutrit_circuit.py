# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""QutritCircuit: container for a sequence of qutrit gate instructions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Iterator, List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.gates import CSUM, H3, X01
from qutritium.gates.base import Gate

# Type alias: an operation is either an Instruction or the literal "measurement"
# string sentinel. Used for internal QutritCircuit class only
_Operation = Union[Instruction, str]


class QutritCircuit:
    """Ordered list of gate operations with an optional terminal measurement.

    Pass to ``QASMSimulator`` or ``DensityMatrixSimulator`` to evaluate.
    """

    def __init__(
            self, n_qutrit: int, initial_state: NDArray | None,
    ) -> None:
        """Construct a QutritCircuit.

        Parameters
        ----------
        n_qutrit : int
            Number of qutrits in the register. Must be ``>= 1``.
        initial_state : NDArray or None
            Initial statevector of shape ``(3 ** n_qutrit, 1)``. ``None``
            defaults to ``|0...0>``.

        Raises
        ------
        ValueError
            If ``n_qutrit < 1`` or ``initial_state`` has the wrong shape.
        """
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
                raise ValueError(
                    f"initial_state has shape {initial_state.shape}; "
                    f"expected {expected_shape}."
                )
            self.initial_state = np.asarray(initial_state, dtype=complex)
            self.state = self.initial_state.copy()
        else:
            ket0 = np.zeros((self._dimension, 1), dtype=complex)
            ket0[0, 0] = 1.0
            self.initial_state = ket0
            self.state = ket0.copy()

    def append(
            self,
            gate: "Gate",
            first_qutrit: int,
            second_qutrit: int | None = None,
    ) -> None:
        """Add a gate to the circuit. For the adjoint, pass ``gate.inverse()``.

        Parameters
        ----------
        gate : Gate
            Gate instance from ``qutritium.gates``.
        first_qutrit : int
            Target qutrit index (0-based). Control qutrit for two-qutrit
            gates by convention.
        second_qutrit : int or None, optional
            Required for two-qutrit gates, ``None`` otherwise.

        Raises
        ------
        TypeError
            If ``gate`` is not a ``Gate`` instance.
        ValueError
            If ``gate.num_qutrits`` and the provided qutrit indices are
            inconsistent.
        IndexError
            If a qutrit index is outside ``[0, n_qutrit]``.
        """
        # Runtime import to avoid circular dependency:
        # gates → elementary_matrices ← instruction ← qutrit_circuit
        from qutritium.gates.base import Gate as _Gate

        if not isinstance(gate, _Gate):
            raise TypeError(
                f"append() expects a Gate instance, got {type(gate).__name__}."
            )

        # Validate qutrit consistency with gate width
        if gate.num_qutrits == 2 and second_qutrit is None:
            raise ValueError(
                f"{gate.label} is a 2-qutrit gate and requires second_qutrit."
            )
        if gate.num_qutrits == 1 and second_qutrit is not None:
            raise ValueError(
                f"{gate.label} is a 1-qutrit gate and does not accept second_qutrit."
            )

        mat = gate.matrix()

        # Build Instruction with Gate reference preserved
        ins = Instruction(
            gate_type=gate.label,
            n_qutrit=self.n_qutrit,
            first_qutrit=first_qutrit,
            second_qutrit=second_qutrit,
            parameter=list(gate.params) if gate.params else None,
            custom=True,
            custom_matrix=np.asarray(mat, dtype=complex),
            gate=gate,
        )
        self._extend_operation_set([ins])

    def measure_all(self) -> None:
        """Add measurement. Can only be called once.

        This only add string ''measurement'' to our instruction set -> It
        will be handled by statevector.py file

        """
        if self._measurement_flag:
            raise RuntimeError("A measurement has already been added to this circuit.")
        self._measurement_flag = True
        self._extend_operation_set(["measurement"])

    @property
    def operation_set(self) -> List[_Operation]:
        """List of operations."""
        return self._operation_set

    @operation_set.setter
    def operation_set(self, ops: List[_Operation]) -> None:
        self._operation_set = list(ops)

    def _extend_operation_set(self, ops: Sequence[_Operation]) -> None:
        """Append operations to the circuit (private helper)."""
        self._operation_set.extend(ops)

    @property
    def measurement_flag(self) -> bool:
        """True if measure_all() was called."""
        return self._measurement_flag

    def reset_circuit(self) -> None:
        """Remove all recorded operations.

        Note
        ----
        This does **not** reset the ``measurement_flag`` -- a circuit that
        had its measurement registered before reset cannot register another
        without rebuilding.
        """
        self._operation_set.clear()

    def gate_count(self) -> int:
        """Number of gates operations, excluding measurements"""
        return sum(1 for operation in self._operation_set if isinstance(operation, Instruction))

    def depth(
            self,
            filter_function: Callable[[Instruction], bool] = lambda _: True,
    ) -> int:
        """Circuit depth

        This function mirrors Qiskit Circuit depth function by calculating the depth based
        on Instruction filter
        Parameters
        ----------
        filter_function : Callable[[Instruction], bool], optional
            Predicate selecting which instructions contribute to the depth
            count. Default counts every gate

        Returns
        -------
        int
            Length of the critical path. ``0`` for an empty circuit.

        Example:
        -------
        >>> qc = QutritCircuit(3, None)
        >>> qc.append(H3(), 0)
        >>> qc.append(H3(), 1)
        >>> qc.append(X01(), 2)
        >>> qc.depth()
        1

        >>> qc2 = QutritCircuit(3, None)
        >>> qc2.append(H3(), 0)
        >>> qc2.append(H3(), 1)
        >>> qc2.append(X01(), 2)
        >>> qc2.depth(lambda ins: ins.second_qutrit is not None)
        0
        """

        depth = [0] * self.n_qutrit
        for operation in self._operation_set:
            if not isinstance(operation, Instruction):
                continue
            assert isinstance(operation, Instruction)
            if operation.second_qutrit is None:
                d = (operation.first_qutrit,)
            else:
                d = (operation.first_qutrit, operation.second_qutrit)  # type: ignore[assignment]
            new_depth = max(depth[ind] for ind in d)
            if filter_function(operation):
                new_depth += 1
            for ind in d:
                depth[ind] = new_depth
        return max(depth, default=0)

    def draw(self) -> str:
        """Render the circuit as a text diagram.

        Returns
        -------
        str
            Multi-line text diagram. Wrap in ``print(...)`` to display.

        Example
        -------
        >>> qc = QutritCircuit(2, None)
        >>> qc.append(H3(), first_qutrit=0)
        >>> qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
        >>> qc.measure_all()
        >>> print(qc.draw())
        q0: ─ H3 ─ CSUM─● ─ M ─
        q1: ───── CSUM─○ ─ M ─
        """
        # Collect gate events per time-step (each Instruction is one step)
        steps: list[dict[int, str]] = []
        for op in self._operation_set:
            if isinstance(op, Instruction):
                slot: dict[int, str] = {}
                gate_obj = op.gate
                label = gate_obj.label if gate_obj is not None else op.type

                if op.parameter and not (gate_obj and gate_obj.params):
                    params_str = ",".join(f"{p:.2g}" for p in op.parameter)
                    label = f"{label}({params_str})"
                elif gate_obj and gate_obj.params:
                    params_str = ",".join(f"{p:.2g}" for p in gate_obj.params)
                    label = f"{label}({params_str})"

                if op.second_qutrit is not None:
                    # Two-qutrit gate: control line gets ●, target gets ○
                    slot[op.first_qutrit] = f"{label}─●"
                    slot[op.second_qutrit] = f"{label}─○"
                else:
                    slot[op.first_qutrit] = label
                steps.append(slot)
            elif op == "measurement":
                steps.append({q: "M" for q in range(self.n_qutrit)})

        if not steps:
            return "\n".join(f"q{q}: ─" for q in range(self.n_qutrit))

        # Determine column widths (each step is one column)
        col_widths: list[int] = []
        for step in steps:
            max_w = max((len(step.get(q, "")) for q in range(self.n_qutrit)), default=1)
            col_widths.append(max(max_w, 1))

        # Build lines
        lines: list[str] = []
        for q in range(self.n_qutrit):
            prefix = f"q{q}: "
            segments: list[str] = []
            for i, step in enumerate(steps):
                cell = step.get(q, "")
                w = col_widths[i]
                if cell:
                    padded = f" {cell} ".center(w + 4, "─")
                else:
                    padded = "─" * (w + 4)
                segments.append(padded)
            lines.append(prefix + "".join(segments) + "─")

        return "\n" + "\n".join(lines) + "\n"

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the final matrix 3^n x 3^n unitary matrix.

        Gates are multiplied in time order, most recent on the left

        Raises
        ------
        RuntimeError
            If the circuit contains a measurement
        """
        if self._measurement_flag:
            raise RuntimeError("Measurement is present")
        result = np.eye(3 ** self.n_qutrit, dtype=complex)
        for operation in self._operation_set:
            assert isinstance(operation, Instruction)
            result = operation.effect_matrix @ result
        return result

    def __len__(self) -> int:
        """Number of recorded operations (counting the measurement sentinel)."""
        return len(self._operation_set)

    def __iter__(self) -> Iterator[_Operation]:
        """Iterate over operations."""
        return iter(self._operation_set)

    def __repr__(self) -> str:
        meas = " (measured)" if self._measurement_flag else ""
        return f"QutritCircuit(n_qutrit={self.n_qutrit}, ops={len(self._operation_set)}){meas}"

    def __add__(self, other: "QutritCircuit") -> "QutritCircuit":
        """Concatenate two circuits. Left must not have a measurement."""
        if self.n_qutrit != other.n_qutrit:
            raise ValueError(
                f"Cannot concatenate circuits with different qutrit counts: "
                f"{self.n_qutrit} vs {other.n_qutrit}."
            )
        if self._measurement_flag:
            raise RuntimeError(
                "Left-hand circuit contains a measurement; cannot prepend more gates."
            )
        result = QutritCircuit(self.n_qutrit, self.initial_state)
        result._operation_set = list(self._operation_set) + list(other.operation_set)
        result._measurement_flag = other.measurement_flag
        return result


__all__ = ["QutritCircuit"]
