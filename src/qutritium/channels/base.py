# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Channel: a CPTP map stored as Kraus operators."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _completeness_check(kraus_operators: list[NDArray], dimension: int) -> None:
    """Raise if sum_i K_i^dag K_i != I (trace preservation)."""
    sum_ks = sum(k.conj().T @ k for k in kraus_operators)
    if not np.allclose(sum_ks, np.eye(dimension), atol=1e-8):
        raise ValueError("Sum of Kraus operators does not satisfy completeness condition.")


def _dimension_check(kraus_operators: list[NDArray], dimension: int) -> None:
    """Raise if any Kraus operator is not (dimension, dimension)."""
    for k in kraus_operators:
        if k.shape != (dimension, dimension):
            raise ValueError(
                f"Kraus operator must have shape ({dimension}, {dimension}); got {k.shape}."
            )


class Channel:
    """A CPTP map as Kraus operators K_i, with sum_i K_i^dag K_i = I."""

    def __init__(self, kraus_operators: list[NDArray], num_qutrits: int = 1) -> None:
        """Ctor.

        Parameters
        ----------
        kraus_operators : list[NDArray]
            Describe how our density matrix evolve under environment interaction.
        num_qutrits : int
            number of qutrits, defaults to 1.

        Raises
        ------
        ValueError
            If a Kraus operator does not have shape (dimension, dimension).
            If the sum of Kraus operators does not satisfy completeness condition.
            If ``num_qutrits < 1``.
        """
        dimension = 3 ** num_qutrits
        ks = [np.asarray(k, dtype=complex) for k in kraus_operators]

        # Validate Kraus operators
        if num_qutrits < 1:
            raise ValueError("Number of qutrits must be at least 1.")
        _dimension_check(ks, dimension)
        _completeness_check(ks, dimension)

        self.num_qutrits = num_qutrits
        self._kraus = ks
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Dimension of Kraus operators."""
        return self._dimension  # type: ignore[no-any-return]

    @property
    def kraus(self) -> list[NDArray]:
        """Kraus operator list (a copy; mutating it does not affect the channel)."""
        return list(self._kraus)

    @kraus.setter
    def kraus(self, kraus_operators: list[NDArray]) -> None:
        """Modify the Kraus operator list, assume a complete reset.

        Would raise if ``kraus_operators`` do not satisfy the above
        conditions. If they do, the Kraus operator list is replaced.

        Parameters
        ----------
        kraus_operators : list[NDArray]
            List of Kraus operators.
        """
        ks = [np.asarray(k, dtype=complex) for k in kraus_operators]
        _dimension_check(ks, self.dimension)
        _completeness_check(ks, self.dimension)
        self._kraus = ks

    def apply_kraus_op(self, rho: NDArray) -> NDArray:
        """Apply channel to density matrix."""
        return sum(k @ rho @ k.conj().T for k in self._kraus)  # type: ignore[no-any-return]


__all__ = ["Channel"]
