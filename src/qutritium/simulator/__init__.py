# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

# MODIFIED: replaced ``from .QASM_backend import *`` (wildcard, leaks numpy
# and matplotlib) with an explicit re-export of the public class.
"""Statevector simulator for qutrit circuits."""

from .statevector import QASMSimulator

__all__ = ["QASMSimulator"]
