# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Two-qutrit gates: CSUM, CPhase, SWAP3, CNOT3.

Ref: Wang et al., Front. Phys. 8, 589504 (2020).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import qutritium.circuit.elementary_matrices as em
from qutritium.gates.base import Gate


class CSUM(Gate):
    """CSUM: |c, t> -> |c, (t+c) mod 3>. Standard qutrit entangling gate."""

    def __init__(self) -> None:
        super().__init__(label="CSUM", num_qutrits=2)

    def matrix(self) -> NDArray[np.complex128]:
        return em.csum()

    def inverse(self) -> CSUMDag:
        return CSUMDag()


class CSUMDag(Gate):
    """CSUM inverse: |c, t> -> |c, (t-c) mod 3>."""

    def __init__(self) -> None:
        super().__init__(label="CSUM†", num_qutrits=2)

    def matrix(self) -> NDArray[np.complex128]:
        return em.csum_dag()

    def inverse(self) -> CSUM:
        return CSUM()


class CNOT3(Gate):
    """Legacy CNOT from v0.0.1. Equivalent to CSUM on adjacent qutrits."""

    def __init__(self) -> None:
        super().__init__(label="CNOT3", num_qutrits=2)

    def matrix(self) -> NDArray[np.complex128]:
        return em.cnot(control=0, target=1)


class CPhase(Gate):
    """CPhase: |c,t> -> omega^{c*t} |c,t>. Qutrit CZ analogue."""

    def __init__(self) -> None:
        super().__init__(label="CPhase", num_qutrits=2)

    def matrix(self) -> NDArray[np.complex128]:
        return em.cphase()

    def inverse(self) -> CPhaseDag:
        return CPhaseDag()


class CPhaseDag(Gate):
    """CPhase inverse: |c,t> -> omega^{-c*t} |c,t>."""

    def __init__(self) -> None:
        super().__init__(label="CPhase†", num_qutrits=2)

    def matrix(self) -> NDArray[np.complex128]:
        return em.cphase().conj().T  # type: ignore[no-any-return]

    def inverse(self) -> CPhase:
        return CPhase()


class SWAP3(Gate):
    """SWAP: |a,b> -> |b,a>. Self-inverse."""

    def __init__(self) -> None:
        super().__init__(label="SWAP3", num_qutrits=2)

    def matrix(self) -> NDArray[np.complex128]:
        return em.swap3()

    def inverse(self) -> SWAP3:
        return SWAP3()


__all__ = [
    "CNOT3",
    "CSUM",
    "SWAP3",
    "CPhase",
    "CPhaseDag",
    "CSUMDag",
]
