# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""NoiseMode: attach to simulator."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qutritium.channels.base import Channel
from qutritium.channels.presets import pauli_channel
from qutritium.channels.readout import ReadoutError


def _symmetric_confusion_matrix(p_meas: float, n_qutrit: int) -> NDArray:
    """Simple symmetric confusion model to n_qutrits.

    For example: P(j = 1 | i = 0) = P(j = 2 | i = 0)

    Parameters
    ----------
    p_meas: float
        Measurement error rate.
    n_qutrit: int
        Number of qutrits.

    Returns
    -------
    NDArray
        The ``3^n x 3^n`` column-stochastic confusion matrix.
    """
    single = (1 - p_meas) * np.eye(3) + (p_meas / 2) * (np.ones((3, 3)) - np.eye(3))
    expand_tensor = single
    for _ in range(n_qutrit - 1):
        expand_tensor = np.kron(expand_tensor, single)  # type: ignore[assignment]
    return expand_tensor


class NoiseModel:
    """Container, set on the simulator."""

    def __init__(self) -> None:
        """Ctor."""
        self._gate_errors: dict[str, Channel] = {}
        self._prep_errors: list[tuple[Channel, int | None]] = []
        self._readout: ReadoutError | None = None

    def add_quantum_error(self, channel: Channel, gate_label: str) -> None:
        """Apply ``channel`` after every gate whose label matches ``gate_label``."""
        self._gate_errors[gate_label] = channel

    def add_prep_error(self, channel: Channel, qutrit: int | None = None) -> None:
        """Apply ``channel`` before all gates.

        If ``qutrit`` is None, apply to all qutrits.
        """
        self._prep_errors.append((channel, qutrit))

    def add_readout_error(self, readout: ReadoutError) -> None:
        """Apply confusion matrix measurement."""
        self._readout = readout

    def error_for(self, gate_label: str) -> Channel | None:
        """Return the channel corresponding to the gate_label, None if no results."""
        return self._gate_errors.get(gate_label)

    @property
    def prep_errors(self) -> list[tuple[Channel, int | None]]:
        """Return the prep errors."""
        return self._prep_errors

    @property
    def readout(self) -> ReadoutError | None:
        """Return the readout error."""
        return self._readout

    @property
    def has_kraus_errors(self) -> bool:
        """Return True if gate or prep errors are set."""
        return bool(self._gate_errors or self._prep_errors)


class SPAMNoiseModel(NoiseModel):
    """State prep + readout error.

    Prep error is a symmetric Pauli mix on every qutrit
    ``{x_plus: p_prep/2, x_minus: p_prep/2, identity: 1 - p_prep}``; readout
    is a symmetric confusion matrix with off-diagonal mass ``p_meas``.
    """

    def __init__(self, p_prep: float, p_meas: float, n_qutrit: int = 1) -> None:
        """Ctor.

        Parameters
        ----------
        p_prep: float
            Prep error.
        p_meas: float
            Measurement error.
        n_qutrit: int
            Number of qutrits.

        Raises
        ------
        ValueError
            If ``p_prep`` and ``p_meas`` are not within [0, 1].
        """
        if not 0 <= p_prep <= 1 or not 0 <= p_meas <= 1:
            raise ValueError(f"Prep error and measurement error are outside [0, 1]; "
                             f"got {p_prep} and {p_meas}.")
        super().__init__()
        prep = pauli_channel({"x_plus": p_prep / 2, "x_minus": p_prep / 2, "identity": 1 - p_prep})
        self.add_prep_error(prep)
        self.add_readout_error(ReadoutError(_symmetric_confusion_matrix(p_meas, n_qutrit)))


__all__ = ["NoiseModel", "SPAMNoiseModel"]
