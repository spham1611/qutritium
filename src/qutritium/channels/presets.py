# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Standard single-qutrit noise channels as Kraus operator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import qutritium.circuit.elementary_matrices as em
from qutritium.channels.base import Channel

_OMEGA: complex = complex(np.exp(2j * np.pi / 3))


# Validate methods
def _validate_p(p: float, p_name: str = "p") -> None:
    """Validate probability within range."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{p_name} must be between 0 and 1; got {p}")


def _validate_unitary_name(name: str) -> NDArray[np.complex128]:
    """Validate the unitary name, mainly used for Pauli channels.
    Returns if it is correct.
    """
    names = {
        "identity": em.identity,
        "x_plus": em.x_plus,
        "x_minus": em.x_minus,
        "z": _clock,
        "z2": lambda: _clock() @ _clock(),
        "x01": em.x01,
        "x02": em.x02,
        "x12": em.x12,
        "y01": em.y01,
        "y02": em.y02,
        "y12": em.y12,
        "z01": em.z01,
        "z02": em.z02,
        "z12": em.z12,
    }
    if name not in names:
        raise ValueError(
            f"Unknown Pauli operator {name!r}; supported: {sorted(names)}."
        )
    return names[name]()


# Define Weyl operators
def _clock() -> NDArray[np.complex128]:
    """Clock operator Z = diag(1, omega, omega^2), not ``sdg()`` in ``elementary_matrices``."""
    return np.diag([1, _OMEGA, _OMEGA ** 2]).astype(np.complex128)


def _weyl(a: int, b: int) -> NDArray[np.complex128]:
    """Weyl operator W_ab = X^a Z^b, with shift X = ``x_plus()`` and clock Z.

    References
    ----------
    Gottesman, D. (1998). Fault-tolerant quantum computation with
    higher-dimensional systems. arXiv:quant-ph/9802007.

    Schwinger, J. (1960). Unitary operator bases. PNAS 46, 570.
    """
    x_pow = np.linalg.matrix_power(em.x_plus(), a)
    z_pow = np.linalg.matrix_power(_clock(), b)
    return x_pow @ z_pow  # type: ignore[no-any-return]


# Channel
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


def amplitude_damping_channel(
        gamma_10: float, gamma_21: float | None = None
) -> Channel:
    """Ladder decay.

    This is for one rung only with the nearest neighbor transition. The channel equation
    follows from Grassl's paper (arXiv:1509.06829).

    Parameters
    ----------
    gamma_10: float
        Decay rate from |1> -> |0>.
    gamma_21: float | None
        Decay rate from |2> -> |1>. Defaults to ``gamma_10``.

    Returns
    -------
    Channel
        The single-qutrit ladder-relaxation channel.
    """
    if gamma_21 is None:
        gamma_21 = gamma_10
    _validate_p(gamma_21, p_name="gamma_21")
    _validate_p(gamma_10, p_name="gamma_10")
    kraus_0 = np.diag([1, np.sqrt(1 - gamma_10), np.sqrt(1 - gamma_21)]).astype(
        np.complex128
    )
    kraus_1 = np.zeros((3, 3)).astype(np.complex128)
    kraus_1[0, 1] = np.sqrt(gamma_10)
    kraus_2 = np.zeros((3, 3)).astype(np.complex128)
    kraus_2[1, 2] = np.sqrt(gamma_21)
    return Channel([kraus_0, kraus_1, kraus_2])


def pauli_channel(probabilities: dict[str, float]) -> Channel:
    """Pauli channel.

    Parameters
    ----------
    probabilities: dict[str, float]
        Mapping from Pauli/Weyl operator name to its probability.

    Returns
    -------
    Channel
        The mixed-unitary Pauli channel.

    Raises
    ------
    ValueError
        If any probability is < 0, or the probabilities do not sum to 1.
    """
    if any(prob < 0 for prob in probabilities.values()):
        raise ValueError("All probabilities must be non-negative.")
    total = sum(probabilities.values())
    if not np.isclose(total, 1, atol=1e-8):
        raise ValueError("Completeness condition is not satisfied.")
    kraus = [
        np.sqrt(prob) * _validate_unitary_name(name)
        for name, prob in probabilities.items()
        if prob > 0
    ]
    return Channel(kraus)


__all__ = [
    "amplitude_damping_channel",
    "dephasing_channel",
    "depolarizing_channel",
    "pauli_channel",
]
