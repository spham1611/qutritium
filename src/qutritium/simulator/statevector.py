# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Statevector simulator. Applies gates sequentially, samples Born-rule outcomes."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.simulator.base import Simulator

if TYPE_CHECKING:
    from qutritium.channels.noise_model import NoiseModel


class QASMSimulator(Simulator):
    """Statevector simulator for a ``QutritCircuit``.

    Applies each gate's full-register matrix to the running statevector
    and (optionally) samples measurement outcomes. Generally faster than
    ``DensityMatrixSimulator``; for mixed states, use that instead.
    """

    name = "qasm_simulator"

    def __init__(self, circuit: QutritCircuit) -> None:
        """Initialize a ``QASMSimulator`` instance.

        Parameters
        ----------
        circuit : QutritCircuit
        """
        super().__init__(circuit)
        self.initial_state = self.circuit.initial_state
        self.state: NDArray = self.initial_state.copy()

    def _simulation(self) -> None:
        """Run the simulator."""
        if self._simulation_flag:
            return
        operations = (self._operation_set[:-1] if self._measurement_flag else self._operation_set)
        for operation in operations:
            assert isinstance(operation, Instruction)
            self.state = operation.effect_matrix @ self.state
        self._simulation_flag = True

    def probabilities(self) -> NDArray[np.float64]:
        """Born-rule probabilities

        For a state |psi>, the probability is given by: |<k|psi>|^2 where
        k is the computational basis
        """
        if not self._simulation_flag:
            self._simulation()
        return np.abs(self.state.flatten()) ** 2  # type: ignore[no-any-return]

    def return_final_state(self) -> NDArray:
        """Return the final state of the simulator."""
        if not self._simulation_flag:
            self._simulation()
        return self.state

    def density_matrix(self) -> NDArray[np.complex128]:
        """Pure-state density matrix |psi><psi|"""
        if not self._simulation_flag:
            self._simulation()
        return self.state @ self.state.conj().T  # type: ignore[no-any-return]

    def set_noise_model(self, noise_model: NoiseModel) -> None:
        """Set the noise model.

        However, this only accepts the readout error only!

        Raises
        ------
        NotImplementedError
            If the model carries Kraus (gate or prep) errors.
        RuntimeError
            If the simulation has already run.
        """
        if noise_model.has_kraus_errors:
            raise NotImplementedError(
                "QASMSimulator can not represent Kraus noise. Use DensityMatrixSimulator instead.")
        super().set_noise_model(noise_model)


__all__ = ["QASMSimulator"]
