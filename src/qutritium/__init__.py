# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.
"""Qutritium — qutrit quantum computing library.

Re-exports the main classes at package level for convenience.
"""
from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.decomposition.transpilation import SU3Decomposition
from qutritium.metrics import (
    average_gate_fidelity,
    process_fidelity,
    purity,
    state_fidelity,
    trace_distance,
    von_neumann_entropy,
)
from qutritium.simulator.density_matrix import DensityMatrixSimulator
from qutritium.simulator.statevector import QASMSimulator

__version__ = "1.5.0"

__all__ = [
    "DensityMatrixSimulator",
    "Instruction",
    "QASMSimulator",
    "QutritCircuit",
    "SU3Decomposition",
    "__version__",
    # Metrics
    "average_gate_fidelity",
    "process_fidelity",
    "purity",
    "state_fidelity",
    "trace_distance",
    "von_neumann_entropy",
]
