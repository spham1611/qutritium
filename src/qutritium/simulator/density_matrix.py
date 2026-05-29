# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Density-matrix simulator."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.simulator.base import Simulator


class DensityMatrixSimulator(Simulator):
    """Density-matrix simulator for a :class:`QutritCircuit`.

    Evolves the density matrix via ``rho -> U rho U^dag``. Memory scales
    as ``9^n_qutrit``; use only for small registers or mixed states.
    Otherwise, prefer
    :class:`~qutritium.simulator.statevector.QASMSimulator`.
    """

    name = "density_matrix_simulator"

    def __init__(self, circuit: QutritCircuit) -> None:
        """Initialize a :class:`DensityMatrixSimulator`.

        Parameters
        ----------
        circuit : QutritCircuit
            Circuit to simulate. The initial state is now the density matrix
            ``|psi><psi|``.
        """
        super().__init__(circuit)
        psi = self.circuit.initial_state
        self.state: NDArray = psi @ psi.conj().T

    def _simulation(self) -> None:
        """Evolve the density matrix through the gate sequence."""
        if self._simulation_flag:
            return
        operations = (self._operation_set[:-1] if self._measurement_flag
                      else self._operation_set)
        for operation in operations:
            assert isinstance(operation, Instruction)
            unitary = operation.effect_matrix
            self.state = unitary @ self.state @ unitary.conj().T
        self._simulation_flag = True

    def probabilities(self) -> NDArray[np.float64]:
        """Born-rule probabilities <k|rho|k> (the diagonal of rho)."""
        if not self._simulation_flag:
            self._simulation()
        return np.real(np.diag(self.state))

    def return_final_state(self) -> NDArray:
        """Final density matrix (runs the simulation if needed)."""
        if not self._simulation_flag:
            self._simulation()
        return self.state

    def expectation_value(self, observable: NDArray[np.complex128]) -> float:
        """Expectation value ``<O> = tr(rho O)`` for a Hermitian observable.

        Parameters
        ----------
        observable : NDArray[np.complex128]
            Shape ``(3^n, 3^n)``. Hermiticity is validated.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If ``observable`` has the wrong shape or is not Hermitian.
        """
        dimension = 3 ** self.n_qutrit
        obs = np.asarray(observable, dtype=np.complex128)
        if obs.shape != (dimension, dimension):
            raise ValueError(
                f"Observable must have shape ({dimension}, {dimension}); "
                f"got {obs.shape}."
            )
        if not np.allclose(obs, obs.conj().T, atol=1e-8):
            raise ValueError("Observable must be Hermitian.")
        if not self._simulation_flag:
            self._simulation()
        return float(np.real(np.trace(self.state @ obs)))

    def partial_trace(self, keep_indices: list[int]) -> NDArray[np.complex128]:
        """Reduced density matrix on the qutrits in ``keep_indices``.

        Traces out every qutrit not in ``keep_indices``. The output is
        indexed in ascending qutrit order regardless of the order of
        ``keep_indices``.

        Parameters
        ----------
        keep_indices : list[int]
            Qutrit indices to retain. Must be a non-empty, duplicate-free
            subset of ``range(n_qutrit)``.

        Returns
        -------
        NDArray
            Shape ``(3^k, 3^k)`` where ``k = len(keep_indices)``.

        Raises
        ------
        ValueError
            If ``keep_indices`` is empty, contains duplicates, or has
            out-of-range indices.
        """
        if not keep_indices:
            raise ValueError("keep_indices must not be empty.")
        keep_set = set(keep_indices)
        if len(keep_set) != len(keep_indices):
            raise ValueError("keep_indices must not contain duplicates.")
        if not all(0 <= q < self.n_qutrit for q in keep_indices):
            raise ValueError(
                f"keep_indices must be in range [0, {self.n_qutrit}); "
                f"got {keep_indices}."
            )
        if not self._simulation_flag:
            self._simulation()

        n = self.n_qutrit
        # Reshape the 3^n x 3^n matrix into 2n tensor axes
        rho = self.state.reshape((3,) * (2 * n))
        # Trace out qutrits not kept, highest index first so the axis
        # labels of the lower (kept) qutrits stay valid as the tensor shrinks.
        trace_out = sorted(
            (q for q in range(n) if q not in keep_set), reverse=True,
        )
        for q in trace_out:
            remaining_n = rho.ndim // 2
            rho = np.trace(rho, axis1=q, axis2=q + remaining_n)
        k = len(keep_set)
        return rho.reshape(3 ** k, 3 ** k)


__all__ = ["DensityMatrixSimulator"]
