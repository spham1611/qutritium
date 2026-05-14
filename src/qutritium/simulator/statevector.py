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
:class:`QASMSimulator` -- a numerically exact statevector simulator for
:class:`qutritium.circuit.qutrit_circuit.QutritCircuit` instances.

The simulator applies each gate's precomputed effect matrix in sequence and
samples a multinomial distribution over computational-basis states for the
final measurement.
"""
# MODIFIED: import paths updated to the new ``qutritium.*`` layout.
# MODIFIED: ``add_SPAM_noise`` previously called
# ``self._operation_set.insert(__index=0, __object=...)``, which is INVALID
# Python -- ``list.insert`` does not accept dunder keyword arguments. The
# call would raise ``TypeError`` the first time the SPAM-noise codepath
# was exercised (the v0.0.1 tests did not exercise this path, so the bug
# was latent). Replaced with ``self._operation_set.insert(0, ...)``.
# MODIFIED: ``density_matrix`` was using ``state @ state.T`` (transpose) for
# what should be the outer product of a (potentially complex) statevector.
# That returns ``ψψ^T``, NOT ``ψψ^*`` -- it's only correct for real
# statevectors. Fixed to ``state @ state.conj().T``, which gives the proper
# pure-state density matrix.
# MODIFIED: bare ``Exception`` -> specific exception types throughout.
# MODIFIED: ``run`` previously could be called without ``measure_all`` being
# called first AND with ``num_shots`` valid, but only raised if
# ``self._measurement_flag`` was False AFTER the simulation step ran.
# Reordered so the validation happens first.
# MODIFIED: ``plot`` now uses ``ax.set_*`` methods rather than implicit
# ``plt.*`` state, returns the Figure for testability, and lazy-imports
# matplotlib so the simulator itself doesn't require matplotlib at import
# time. (Previously the module-level ``import matplotlib.pyplot`` made the
# simulator unimportable in headless environments without matplotlib.)
# MODIFIED: ``get_counts`` uses ``collections.Counter`` instead of an O(n^2)
# ``dict((x, list.count(x)) for x in set(list))`` comprehension.
# MODIFIED: type hints, docstrings, dropped ``return None`` ``simulation``
# step ambiguity, etc.
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.circuit.utils import statevector_to_state

if TYPE_CHECKING:
    # Type-only; the runtime ``plot`` codepath imports matplotlib lazily.
    from matplotlib.figure import Figure


# ADDED: tuple of supported plot kinds, validated against in ``plot``.
_VALID_PLOT_TYPES: Tuple[str, ...] = ("histogram", "line", "dot")


class QASMSimulator:
    """Statevector simulator for a :class:`QutritCircuit`.

    The simulator multiplies each instruction's ``effect_matrix`` into the
    running statevector, then (if a measurement was registered) samples
    ``num_shots`` outcomes from the resulting Born distribution.

    Parameters
    ----------
    qc : QutritCircuit
        The circuit to simulate. Its ``operation_set`` is consumed but not
        mutated.
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
        """Add a State-Preparation-And-Measurement noise model to the simulation.

        Parameters
        ----------
        p_prep : float
            Probability of a preparation error (per qutrit).
        p_meas : float
            Probability of a measurement error (per qutrit).
        error_type : str
            Currently only ``"Pauli_error"`` is supported.

        Notes
        -----
        Preparation errors are applied immediately as gate insertions at the
        front of the operation list. Measurement errors are stored and
        applied during :meth:`run`, immediately before sampling.
        """
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
        """Run the gate sequence to produce the final statevector."""
        if self._measurement_flag:
            # The last entry is the "measurement" sentinel; skip it.
            for op in self._operation_set[:-1]:
                self.state = np.einsum("ij,jk", op.effect_matrix, self.state)
        else:
            for op in self._operation_set:
                self.state = np.einsum("ij,jk", op.effect_matrix, self.state)
        self._simulation_flag = True

    def run(self, num_shots: int = 1024) -> None:
        """Simulate the circuit and (if measured) sample ``num_shots`` outcomes.

        Parameters
        ----------
        num_shots : int
            Number of measurement shots to sample. Must be positive.

        Raises
        ------
        ValueError
            If ``num_shots <= 0``.
        RuntimeError
            If the circuit does not contain a measurement.
        """
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
        if self._spam_error_active:
            probs_meas = [entry[1] for entry in self._error_meas]
            for i in range(self.n_qutrit):
                choice = rng.choice(len(self._error_meas), p=probs_meas)
                error_effect = Instruction(
                    gate_type=self._error_meas[choice][0],
                    n_qutrit=self.n_qutrit,
                    first_qutrit=i,
                    second_qutrit=None,
                    parameter=None,
                )
                self._operation_set.insert(0, error_effect)

        state_coeff, state_construction = statevector_to_state(self.state, self.n_qutrit)
        probs = np.array([np.abs(c) ** 2 for c in state_coeff])
        # MODIFIED: vectorize the multinomial sampling instead of a Python
        # loop over ``num_shots`` iterations.
        sampled = rng.choice(len(state_construction), size=num_shots, p=probs)
        self._measurement_result = [state_construction[i] for i in sampled]

    def get_counts(self) -> Dict[str, int]:
        """Return a histogram of sampled measurement outcomes."""
        if not self._measurement_result:
            # MODIFIED: ``Exception`` -> ``RuntimeError``. Also covers the
            # case where ``run`` was never called (empty list, not None).
            raise RuntimeError("No measurement result yet; call ``run()`` first.")
        # MODIFIED: O(n) Counter replaces O(n^2) ``set + list.count``.
        return dict(Counter(self._measurement_result))

    def return_final_state(self) -> NDArray:
        """Run the simulation if needed and return the final statevector."""
        if not self._simulation_flag:
            self._simulation()
        return self.state

    def result(self) -> List[str]:
        """Return the raw list of sampled measurement outcomes."""
        if not self._measurement_result:
            raise RuntimeError("No measurement result yet; call ``run()`` first.")
        return self._measurement_result

    def density_matrix(self) -> NDArray:
        """Return the pure-state density matrix ``|psi><psi|`` of the final state."""
        if not self._simulation_flag:
            self._simulation()
        # MODIFIED: ``state @ state.T`` -> ``state @ state.conj().T``. The
        # original was wrong for any state with non-trivial complex phases.
        return self.state @ self.state.conj().T  # type: ignore[no-any-return]

    def plot(self, plot_type: str = "histogram") -> "Figure":
        """Plot the measurement counts.

        Parameters
        ----------
        plot_type : str
            One of ``"histogram"``, ``"line"``, ``"dot"``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object, for further customisation or saving.

        Raises
        ------
        ValueError
            If ``plot_type`` is not recognised.
        """
        # MODIFIED: lazy import so the simulator itself doesn't drag in
        # matplotlib at module load.
        import matplotlib.pyplot as plt  # noqa: PLC0415 (deliberate lazy import)

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
