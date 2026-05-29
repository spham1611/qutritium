# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Qutrit gate library. Import gates from here: ``from qutritium.gates import H3, CSUM``."""
from qutritium.gates.base import Gate
# Single-qutrit gates
from qutritium.gates.single_qutrit import (G01, G02, G12, H3, I3, Rx01, Rx02, Rx12, Ry01, Ry02, Ry12, Rz01, Rz02, Rz12,
                                           S3, T3, Ud, UFT, X01, X02, X12, XMinus, XPlus, Y01, Y02, Y12, Z01, Z02, Z12)
# Two-qutrit gates
from qutritium.gates.two_qutrit import CNOT3, CPhase, CPhaseDag, CSUM, CSUMDag, SWAP3

__all__ = [
    # Base class
    "Gate",
    # Single-qutrit fixed gates
    "I3", "X01", "X02", "X12",
    "Y01", "Y02", "Y12",
    "Z01", "Z02", "Z12",
    "XPlus", "XMinus",
    "H3", "S3", "T3", "UFT",
    # Single-qutrit parametric gates
    "Rx01", "Rx02", "Rx12",
    "Ry01", "Ry02", "Ry12",
    "Rz01", "Rz02", "Rz12",
    "G01", "G02", "G12",
    "Ud",
    # Two-qutrit gates
    "CSUM", "CSUMDag",
    "CNOT3",
    "CPhase", "CPhaseDag",
    "SWAP3",
]
