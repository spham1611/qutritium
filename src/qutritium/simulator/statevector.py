# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Statevector simulator. Applies gates sequentially, samples Born-rule outcomes."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.circuit.utils import statevector_to_state

if TYPE_CHECKING:

    from matplotlib.figure import Figure


_VALID_PLOT_TYPES: Tuple[str, ...] = ("histogram", "line", "dot")


class QASMSimulator:
    """Simulate a QutritCircuit by matrix-vector multiplication."""

    def __init__(self, qc: QutritCircuit) -> None:
        self.circuit: QutritCircuit = qc
        self.n_qutrit: int = qc.n_qutrit
        self.initial_state: NDArray = qc.initial_state
        self.state: NDArray = self.initial_state.copy()
        self._measurement_flag: bool = qc.measurement_flag
        self._operation_set: List = list(qc.operation_set)
        self._spam_error_active: bool = False
        self._error_meas: List[Tuple[str, float]] = []
        self._measurement_result: List[str] = []
        self._simulation_flag: bool = False

    def add_SPAM_noise(
        self, p_prep: float, p_meas: float, error_type: str = "Pauli_error",
    ) -> None:
        """Add SPAM noise. Only Pauli_error is supported."""
        # MODIFIED: input validation added.
        if not 0.0 <= p_prep <= 1.0:
            raise ValueError(f"p_prep must be in [0, 1]; got {p_prep}.")
        if not 0.0 <= p_meas <= 1.0:
            raise ValueError(f"p_meas must be in [0, 1]; got {p_meas}.")

        if error_type != "Pauli_error":
            # MODIFIED: previously the non-Pauli case silently ``pass``ed.
            raise NotImplementedError(
                f"Unsupported SPAM error_type {error_type!r}; only "
                f"'Pauli_error' is currently implemented."
            )

        self._error_meas = [
            ("x_plus", p_meas / 2),
            ("x_minus", p_meas / 2),
            ("identity", 1 - p_meas),
        ]
        error_prep = [
            ("x_plus", p_prep / 2),
            ("x_minus", p_prep / 2),
            ("identity", 1 - p_prep),
        ]
        probs_prep = [entry[1] for entry in error_prep]
        rng = np.random.default_rng()  # MODIFIED: use Generator API not legacy.
        for i in range(self.n_qutrit):
            choice = rng.choice(len(error_prep), p=probs_prep)
            error_effect = Instruction(
                gate_type=error_prep[choice][0],
                n_qutrit=self.n_qutrit,
                first_qutrit=i,
                second_qutrit=None,
                parameter=None,
            )
            self._operation_set.insert(0, error_effect)

        self._spam_error_active = True

    def _simulation(self) -> None:
        """Apply all gates to get the final statevector."""
        if self._measurement_flag:
            # The last entry is the "measurement" sentinel; skip it.
            for op in self._operation_set[:-1]:
                self.state = op.effect_matrix @ self.state
        else:
            for op in self._operation_set:
                self.state = op.effect_matrix @ self.state
        self._simulation_flag = True

    def run(self, num_shots: int = 1024) -> None:
        """Run simulation and sample num_shots measurement outcomes."""
        # MODIFIED: validate num_shots and measurement presence up-front.
        if num_shots <= 0:
            raise ValueError(f"num_shots must be positive; got {num_shots}.")
        if not self._measurement_flag:
            raise RuntimeError(
                "Circuit does not contain a measurement; "
                "call ``measure_all()`` on the circuit before ``run()``."
            )

        if not self._simulation_flag:
            self._simulation()

        rng = np.random.default_rng()
        measured_state = self.state
        if self._spam_error_active:
            probs_meas = [entry[1] for entry in self._error_meas]
            measured_state = measured_state.copy()
            for i in range(self.n_qutrit):
                choice = rng.choice(len(self._error_meas), p=probs_meas)
                error_effect = Instruction(
                    gate_type=self._error_meas[choice][0],
                    n_qutrit=self.n_qutrit,
                    first_qutrit=i,
                    second_qutrit=None,
                    parameter=None,
                )
                measured_state = error_effect.effect_matrix @ measured_state

        state_coeff, state_construction = statevector_to_state(measured_state, self.n_qutrit)
        probs = np.array([np.abs(c) ** 2 for c in state_coeff])
        # MODIFIED: vectorize the multinomial sampling instead of a Python
        # loop over ``num_shots`` iterations.
        sampled = rng.choice(len(state_construction), size=num_shots, p=probs)
        self._measurement_result = [state_construction[i] for i in sampled]

    def get_counts(self) -> Dict[str, int]:
        """Measurement histogram {outcome: count}."""
        if not self._measurement_result:
            # MODIFIED: ``Exception`` -> ``RuntimeError``. Also covers the
            # case where ``run`` was never called (empty list, not None).
            raise RuntimeError("No measurement result yet; call ``run()`` first.")
        # MODIFIED: O(n) Counter replaces O(n^2) ``set + list.count``.
        return dict(Counter(self._measurement_result))

    def return_final_state(self) -> NDArray:
        """Final statevector (runs simulation if needed)."""
        if not self._simulation_flag:
            self._simulation()
        return self.state

    def result(self) -> List[str]:
        """Raw list of sampled outcomes."""
        if not self._measurement_result:
            raise RuntimeError("No measurement result yet; call ``run()`` first.")
        return self._measurement_result

    def density_matrix(self) -> NDArray:
        """Pure-state density matrix |psi><psi|."""
        if not self._simulation_flag:
            self._simulation()
        # MODIFIED: ``state @ state.T`` -> ``state @ state.conj().T``. The
        # original was wrong for any state with non-trivial complex phases.
        return self.state @ self.state.conj().T  # type: ignore[no-any-return]

    def plot(self, plot_type: str = "histogram") -> "Figure":
        """Plot counts. plot_type: "histogram", "line", or "dot"."""
        # MODIFIED: lazy import so the simulator itself doesn't drag in
        # matplotlib at module load.
        import matplotlib.pyplot as plt

        if plot_type not in _VALID_PLOT_TYPES:
            raise ValueError(
                f"plot_type must be one of {_VALID_PLOT_TYPES}; got {plot_type!r}."
            )

        result_dict = self.get_counts()
        keys = list(result_dict.keys())
        values = list(result_dict.values())

        fig, ax = plt.subplots()
        if plot_type == "histogram":
            ax.bar(keys, values)
        elif plot_type == "line":
            ax.plot(keys, values)
        else:  # "dot"
            ax.scatter(keys, values)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Counts")
        ax.set_title(f"Measurement counts ({plot_type})")
        # MODIFIED: removed unconditional ``plt.show()``; callers can decide
        # whether to show, save, or embed the figure.
        return fig


__all__ = ["QASMSimulator"]
