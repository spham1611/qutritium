# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Simulators for qutrit circuits: statevector and density matrix."""

from qutritium.simulator.base import Simulator
from qutritium.simulator.density_matrix import DensityMatrixSimulator
from qutritium.simulator.statevector import QASMSimulator as QASMSimulator
from qutritium.simulator.statevector import StatevectorSimulator

# ``QASMSimulator`` is a deprecated alias for ``StatevectorSimulator`` (removed in
# v2.0); re-exported here for backward compatibility but kept out of ``__all__``.
__all__ = ["DensityMatrixSimulator", "Simulator", "StatevectorSimulator"]
