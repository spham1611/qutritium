# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Qutrit gate library. Import gates from here: ``from qutritium.gates import H3, CSUM``."""

from qutritium.gates.base import Gate

# Single-qutrit gates
from qutritium.gates.single_qutrit import (
    G01,
    G02,
    G12,
    H3,
    I3,
    S3,
    T3,
    UFT,
    X01,
    X02,
    X12,
    Y01,
    Y02,
    Y12,
    Z01,
    Z02,
    Z12,
    Rx01,
    Rx02,
    Rx12,
    Ry01,
    Ry02,
    Ry12,
    Rz01,
    Rz02,
    Rz12,
    Ud,
    XMinus,
    XPlus,
)

# Two-qutrit gates
from qutritium.gates.two_qutrit import CNOT3, CSUM, SWAP3, CPhase, CPhaseDag, CSUMDag

__all__ = [
    "CNOT3",
    # Two-qutrit gates
    "CSUM",
    "G01",
    "G02",
    "G12",
    "H3",
    # Single-qutrit fixed gates
    "I3",
    "S3",
    "SWAP3",
    "T3",
    "UFT",
    "X01",
    "X02",
    "X12",
    "Y01",
    "Y02",
    "Y12",
    "Z01",
    "Z02",
    "Z12",
    "CPhase",
    "CPhaseDag",
    "CSUMDag",
    # Base class
    "Gate",
    # Single-qutrit parametric gates
    "Rx01",
    "Rx02",
    "Rx12",
    "Ry01",
    "Ry02",
    "Ry12",
    "Rz01",
    "Rz02",
    "Rz12",
    "Ud",
    "XMinus",
    "XPlus",
]
