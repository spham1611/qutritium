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
First-class qutrit gate library.

Usage::

    from qutritium.gates import H3, X01, Rx12, CSUM

    h = H3()
    print(h.matrix())       # 3x3 unitary
    print(h.label)           # "H3"
    print(h.num_qutrits)     # 1
    print(h.is_unitary())    # True

    rx = Rx12(theta=np.pi/3)
    print(rx.params)         # (1.0472...,)
    print(rx.inverse())      # Rx12(-1.0472)
"""
from qutritium.gates.base import Gate
# Single-qutrit gates
from qutritium.gates.single_qutrit import (G01, G02, G12, H3, I3, Rx01, Rx02, Rx12, Ry01, Ry02, Ry12, Rz01, Rz02, Rz12,
                                           S3, T3, Ud, UFT, X01, X02, X12, XMinus, XPlus, Y01, Y02, Y12, Z01, Z02, Z12)
# Two-qutrit gates
from qutritium.gates.two_qutrit import (CNOT3, CPhase, CSUM, CSUMDag, SWAP3)

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
    "CPhase",
    "SWAP3",
]
