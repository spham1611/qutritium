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

# ADDED: top-level package docstring + curated public API surface (formerly empty file).
"""
Qutritium: a hardware-agnostic Python library for qutrit (3-level) quantum
computing.

The package provides:

- :mod:`qutritium.circuit` -- qutrit gate definitions, instructions and
  circuit construction (SU(3) elementary matrices, single- and multi-qutrit
  gates).
- :mod:`qutritium.simulator` -- a numerically exact statevector simulator for
  qutrit circuits (no external hardware backend required).
- :mod:`qutritium.decomposition` -- decomposition of arbitrary SU(3) unitaries
  into native :math:`R_{01}`, :math:`R_{12}` and diagonal phase rotations.

The public API re-exports the most common symbols at package level so that
``from qutritium import QutritCircuit, QASMSimulator`` works directly.

History
-------
Earlier versions (0.0.x) shipped Qiskit-pulse-based calibration routines for
running qutrits on IBM quantum hardware. That hardware-specific code now lives
under the top-level ``legacy/`` directory of the repository as a historical
record of the work presented at the Munich Quantum Software Conference 2023;
it is **not** importable from the installed package. See ``legacy/README.md``
for details.
"""
from qutritium.circuit.instruction import Instruction
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.decomposition.transpilation import SU3Decomposition
from qutritium.simulator.statevector import QASMSimulator

__version__ = "1.0.0"

__all__ = [
    "Instruction",
    "QASMSimulator",
    "QutritCircuit",
    "SU3Decomposition",
    "__version__",
]
