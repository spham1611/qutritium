# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.
"""Qutritium — qutrit quantum computing library.

Re-exports the main classes at package level for convenience.
"""
from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.decomposition.transpilation import SU3Decomposition
from qutritium.simulator.statevector import QASMSimulator

__version__ = "1.2.0"

__all__ = [
    "Instruction",
    "QASMSimulator",
    "QutritCircuit",
    "SU3Decomposition",
    "__version__",
]
