# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""QutritCircuit: container for a sequence of qutrit gate instructions."""
from __future__ import annotations

from typing import Iterator, List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.gates import CSUM, H3
from qutritium.gates.base import Gate

# Type alias: an operation is either an Instruction or the literal "measurement"
# string sentinel. Used for internal QutritCircuit class only
_Operation = Union[Instruction, str]


class QutritCircuit:
    """Ordered gate list + optional measurement. Pass to QASMSimulator to run."""

    def __init__(
            self, n_qutrit: int, initial_state: NDArray | None,
    ) -> None:
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
        """Add a Gate to the circuit. Pass ``gate.inverse()`` for the adjoint."""
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
        """Add measurement. Can only be called once."""
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
        # MODIFIED: in v0.0.1 the setter *extended* the list (clearly a bug
        # given Python's ``@property.setter`` contract). The new setter
        # *replaces*; internal append-on-add behavior now goes through
        # ``_extend_operation_set``.
        self._operation_set = list(ops)

    def _extend_operation_set(self, ops: Sequence[_Operation]) -> None:
        """Append operations to the circuit (private helper)."""
        # ADDED: extracted from the broken-by-design setter above.
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
        # MODIFIED: documented the (intentional) measurement_flag persistence.
        self._operation_set.clear()

    def draw(self, output: str = "text") -> str:
        """Render the circuit as a text diagram.

        Parameters
        ----------
        output : str
            ``"text"`` (default) prints and returns the diagram string.

        Returns
        -------
        str
            Multi-line text diagram.

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
                label = gate_obj.label if gate_obj is not None else op.type  # type: ignore[attr-defined]

                if op.parameter and not (gate_obj and gate_obj.params):  # type: ignore[attr-defined]
                    params_str = ",".join(f"{p:.2g}" for p in op.parameter)
                    label = f"{label}({params_str})"
                elif gate_obj and gate_obj.params:  # type: ignore[attr-defined]
                    params_str = ",".join(f"{p:.2g}" for p in gate_obj.params)  # type: ignore[attr-defined]
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
            diagram = "\n".join(f"q{q}: ─" for q in range(self.n_qutrit))
            print(diagram)
            return diagram

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

        diagram = "\n" + "\n".join(lines) + "\n"
        print(diagram)
        return diagram

    def __len__(self) -> int:
        """Number of recorded operations (counting the measurement sentinel)."""
        # ADDED: makes ``len(circuit)`` work.
        return len(self._operation_set)

    def __iter__(self) -> Iterator[_Operation]:
        """Iterate over operations."""
        return iter(self._operation_set)

    def __repr__(self) -> str:
        # ADDED: an actually-useful repr.
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
