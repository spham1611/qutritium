# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""base.py: Container for simulators. Subclasses implement evolution and sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from qutritium.circuit.qutrit_circuit import QutritCircuit

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from qutritium.channels.noise_model import NoiseModel

_VALID_PLOT_TYPES: tuple[str, ...] = ("histogram", "line", "dot")


def _index_to_ket_label(i: int, n_qutrit: int) -> str:
    """Convert basis index to base-3 ket label.


    Parameters
    ----------
    i : int
        number to be converted
    n_qutrit : int
        number of qutrits

    Returns
    -------
    ket_label : str
    """
    assert (
            0 <= i < 3 ** n_qutrit
    ), f"i={i} out of [0, {3 ** n_qutrit}) for n_qutrit={n_qutrit}."
    digits: list[str] = []
    # Loop LSB
    for _ in range(n_qutrit):
        digits.append(str(i % 3))
        i //= 3
    # Reverse to MSB, following big-endian order
    return "".join(reversed(digits))


class Simulator(ABC):
    """Abstract class for simulator."""

    name: str = "simulator"  # Intend to be modified by subclasses

    def __init__(self, qc: QutritCircuit) -> None:
        """Constructor.

        Parameters
        ----------
        qc : QutritCircuit
            QutritCircuit object

        """
        self.circuit = qc
        self.n_qutrit = qc.n_qutrit
        self._operation_set = list(qc.operation_set)
        self._measurement_flag = qc.measurement_flag
        self._measurement_result: list[str] = []
        self._simulation_flag = False
        self._noise_model: NoiseModel | None = None

    @abstractmethod
    def _simulation(self) -> None:
        ...

    @abstractmethod
    def return_final_state(self) -> NDArray:
        ...

    @abstractmethod
    def probabilities(self) -> NDArray[np.float64]:
        """Return Born probabilities"""
        ...

    def run(self, num_shots: int = 1024, seed: int | None = None) -> None:
        """Evolve the circuit for num_shots times.

        Results stored in ``_measurement_result`` and accessed via
        ``get_counts`` or ``result``.

        Parameters
        ----------
        num_shots : int
        seed : int | None, optional
            Seed for the shot-sampling random-number generator.

        Raises
        ------
        ValueError
            If ``num_shots`` is zero or negative.
        RuntimeError
            If the circuit has no measurement.
        """
        if num_shots <= 0:
            raise ValueError("'num_shots' must be a positive integer")
        if not self._measurement_flag:
            raise RuntimeError(
                "Circuit has no measurement; please call measure_all() before calling run()"
            )
        if not self._simulation_flag:
            self._simulation()

        # Clip tiny negative entries.
        probs = self.probabilities()
        if self._noise_model is not None and self._noise_model.readout is not None:
            probs = self._noise_model.readout.apply(probs)
        probs = np.clip(probs, 0.0, None)
        probs = probs / np.sum(probs)
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(probs), size=num_shots, p=probs)
        self._measurement_result = list(
            _index_to_ket_label(int(i), self.n_qutrit) for i in sample
        )

    def get_counts(self) -> dict[str, int]:
        """Measurement histogram."""
        if not self._measurement_result:
            raise RuntimeError("No measurement result; call run() first")
        return dict(Counter(self._measurement_result))

    def result(self) -> list[str]:
        """Raw list of measurement results."""
        if not self._measurement_result:
            raise RuntimeError("No measurement result; call run() first")
        return self._measurement_result

    def set_noise_model(self, noise_model: NoiseModel) -> None:
        """Attach a noise model. Set it before the first run().

        The simulation is cached once it runs, so a model attached afterward
        would be silently ignored — we raise instead.

        Raises
        ------
        RuntimeError
            If the simulation has already run.
        """
        if self._simulation_flag:
            raise RuntimeError(
                "Attach the noise model before the first run(); "
                "build a fresh simulator to change it."
            )
        self._noise_model = noise_model

    def plot(self, plot_type: str = "histogram") -> Figure:
        """Using matplotlib to plot the measurement-count distribution.

        Parameters
        ----------
        plot_type : str, optional
            ``"histogram"`` (default), ``"line"``, or ``"dot"``.

        Returns
        -------
        matplotlib.figure.Figure
            matplot Figure object for further improvement.
        """
        if plot_type not in _VALID_PLOT_TYPES:
            raise ValueError(
                f"'plot_type' must be one of {_VALID_PLOT_TYPES}; got {plot_type}"
            )

        counts = self.get_counts()
        keys = list(counts.keys())
        values = list(counts.values())
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        if plot_type == "histogram":
            ax.bar(keys, values)
        elif plot_type == "line":
            ax.plot(keys, values)
        else:
            ax.scatter(keys, values)

        ax.set_xlabel("Outcomes")
        ax.set_ylabel("Counts")
        ax.set_title(f"Measurement counts ({plot_type})")

        return fig


__all__ = ["Simulator"]
