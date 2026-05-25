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
    """Statevector simulator for a ``QutritCircuit``.

    Applies each gate's full-register matrix to the running statevector
    via matrix-vector multiplication, then (optionally) samples
    measurement outcomes from the Born distribution. Exact and noiseless
    by default; SPAM-style Pauli noise can be added via
    :meth:`add_SPAM_noise`.

    Parameters
    ----------
    qc : QutritCircuit
        The circuit to simulate. Its operations and initial state are
        copied into the simulator, so subsequent edits to ``qc`` do not
        affect this simulator instance.

    Notes
    -----
    Memory scales as ``3 ** n_qutrit`` for the statevector and up to
    ``9 ** n_qutrit`` per stored gate matrix (lazy). Practical limit is
    around ``n_qutrit ≤ 10`` on a typical laptop. For larger systems or
    non-unitary evolution, use the density-matrix simulator (coming in
    v1.3).
    """

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
        """Add state-preparation and measurement (SPAM) noise to the circuit.

        Models SPAM as cyclic-shift Pauli errors on every qutrit, applied
        once before all gates (preparation noise) and once after the final
        gate (measurement noise). Each qutrit's error is sampled
        independently from {x_plus, x_minus, identity} with probabilities
        ``(p/2, p/2, 1-p)``.

        Parameters
        ----------
        p_prep : float
            Probability of a Pauli error during state preparation, per
            qutrit. Must lie in ``[0, 1]``.
        p_meas : float
            Probability of a Pauli error during measurement, per qutrit.
            Must lie in ``[0, 1]``.
        error_type : str, optional
            SPAM error channel. Currently only ``"Pauli_error"`` is
            implemented.

        Raises
        ------
        ValueError
            If ``p_prep`` or ``p_meas`` is outside ``[0, 1]``.
        NotImplementedError
            If ``error_type`` is anything other than ``"Pauli_error"``.

        Notes
        -----
        This API will be deprecated when the v1.4 noise-channel framework
        lands. The replacement will be a ``NoiseModel`` attached to the
        simulator rather than mutating the operation list.
        """
        if not 0.0 <= p_prep <= 1.0:
            raise ValueError(f"p_prep must be in [0, 1]; got {p_prep}.")
        if not 0.0 <= p_meas <= 1.0:
            raise ValueError(f"p_meas must be in [0, 1]; got {p_meas}.")

        if error_type != "Pauli_error":
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
        rng = np.random.default_rng()
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
        """Simulate the circuit and sample measurement outcomes.

        Runs (or reuses) the statevector evolution, then samples
        ``num_shots`` projective measurements from the computational-basis
        Born distribution. Results are stored internally and accessible
        via :meth:`get_counts` and :meth:`result`.

        Parameters
        ----------
        num_shots : int, optional
            Number of measurement shots to sample. Must be positive.
            Defaults to ``1024``.

        Raises
        ------
        ValueError
            If ``num_shots`` is non-positive.
        RuntimeError
            If the circuit does not contain a measurement (call
            ``QutritCircuit.measure_all()`` before instantiating the
            simulator).
        """
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
        sampled = rng.choice(len(state_construction), size=num_shots, p=probs)
        self._measurement_result = [state_construction[i] for i in sampled]

    def get_counts(self) -> Dict[str, int]:
        """Measurement histogram {outcome: count}."""
        if not self._measurement_result:
            raise RuntimeError("No measurement result yet; call ``run()`` first.")
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
        return self.state @ self.state.conj().T  # type: ignore[no-any-return]

    def plot(self, plot_type: str = "histogram") -> "Figure":
        """Plot the measurement-count distribution.

        Parameters
        ----------
        plot_type : str, optional
            Plot style. One of ``"histogram"`` (default), ``"line"``, or
            ``"dot"``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object. Caller decides whether to ``show()``,
            ``savefig()``, or embed.

        Raises
        ------
        ValueError
            If ``plot_type`` is not one of the supported values.
        RuntimeError
            If the simulator has no measurement results yet (call
            :meth:`run` first).
        """
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
        return fig


__all__ = ["QASMSimulator"]
