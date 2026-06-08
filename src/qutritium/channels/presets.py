# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Standard single-qutrit noise channels as Kraus operator."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import qutritium.circuit.elementary_matrices as em
from qutritium.channels.base import Channel

_OMEGA: complex = complex(np.exp(2j * np.pi / 3))


def _validate_p(p: float, p_name: str = "p") -> None:
    """Validate probability within range."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{p_name} must be between 0 and 1; got {p}")


# Define Weyl operators
def _clock() -> NDArray[np.complex128]:
    "Clock operator Z = diag(1, omega, omega^2), not sdg() in 'elementary_matrices.py'"
    return np.diag([1, _OMEGA, _OMEGA ** 2]).astype(np.complex128)


def _weyl(a: int, b: int) -> NDArray[np.complex128]:
    """Weyl operator W_ab = X^a Z^b where X is the 'x_plus()' in 'elementary_matrices.py'."""
    x_pow = np.linalg.matrix_power(em.x_plus(), a)
    z_pow = np.linalg.matrix_power(_clock(), b)
    return x_pow @ z_pow  # type: ignore[no-any-return]


def depolarizing_channel(p: float) -> Channel:
    """Depolarizing noise channel E(rho) = (1-p) rho + p (I/3)). For p = 1, gives I/3."""
    _validate_p(p)
    kraus = [np.sqrt(1 - 8 * p / 9) * em.identity()]
    for a in range(3):
        for b in range(3):
            if (a, b) != (0, 0):
                kraus.append(np.sqrt(p / 9) * _weyl(a, b))
    return Channel(kraus)


def dephasing_channel(p: float) -> Channel:
    """Dephasing noise channel E(rho) = (1-p) rho + p diag(rho). For p = 1, gives diagonal."""
    _validate_p(p)
    kraus = [
        np.sqrt(1 - 2 * p / 3) * em.identity(),
        np.sqrt(p / 3) * _clock(),
        np.sqrt(p / 3) * (_clock() @ _clock()),
    ]
    return Channel(kraus)


def amplitude_damping_channel(gamma_10: float, gamma_21: float | None = None) -> Channel:
    """Ladder decay from |1> -> |0> and |2> -> |1>. This is for one rung only."""
    if gamma_21 is None:
        gamma_21 = gamma_10
    _validate_p(gamma_21, p_name="gamma_21")
    _validate_p(gamma_10, p_name="gamma_10")
    a0 =
