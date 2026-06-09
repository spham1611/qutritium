# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""ReadoutError: classical readout error as a confusion matrix."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ReadoutError:
    """Confusion matrix ``A`` where ``A[j, i] = P(observe j | true i)``.

    Applied to the probability vector: ``p_measured = A @ p_ideal``. This
    is a classical post-process.
    """

    def __init__(self, confusion_matrix: NDArray) -> None:
        """Ctor.

        Raises
        ------
        ValueError
            If the confusion matrix is not a square and not power of 3.
            If the confusion matrix has negative entries.
            If each column of the confusion matrix does not sum to 1.

        """
        array = np.asarray(confusion_matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("confusion matrix must be a square matrix.")
        n = round(np.log(array.shape[0]) / np.log(3)) if array.shape[0] > 0 else -1
        if array.shape[0] < 1 or 3 ** n != array.shape[0]:
            raise ValueError("confusion matrix dimension must be power of 3.")
        if np.any(array < -1e-12):
            raise ValueError("confusion matrix has negative entries.")
        if not np.allclose(array.sum(axis=0), 1, atol=1e-8):
            raise ValueError("confusion matrix columns must sum to 1.")
        self.confusion_matrix = array

    def apply(self, state_vector: NDArray) -> NDArray:
        """Apply the readout error to the ideal state vector."""
        output = self.confusion_matrix @ state_vector
        return output  # type: ignore[no-any-return]

    @classmethod
    def from_single_qutrit(cls, a: NDArray, n_qutrit: int) -> ReadoutError:
        """From a single qutrit confusion matrix -> into 3^n confusion matrix."""
        full_matrix = np.asarray(a, dtype=float)
        for _ in range(n_qutrit - 1):
            full_matrix = np.kron(full_matrix, a)
        return cls(full_matrix)


__all__ = ["ReadoutError"]
