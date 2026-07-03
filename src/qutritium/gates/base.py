# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Gate ABC. All gates have .matrix(), .label, .num_qutrits, .params, .inverse()."""

from __future__ import annotations

import abc

import numpy as np
from numpy.typing import NDArray


class Gate(abc.ABC):
    """Base class for qutrit gates."""

    def __init__(
            self,
            label: str,
            num_qutrits: int,
            params: tuple[float, ...] = (),
    ) -> None:
        """Construct a Gate.

        Parameters
        ----------
        label : str
            Display name for the gate (e.g. ``"X01"``, ``"H3"``).
        num_qutrits : int
            Gate width. Must be ``1`` or ``2``.
        params : tuple of float, optional
            Numerical parameters such as rotation angles. Empty for
            zero-parameter gates.

        Raises
        ------
        ValueError
            If ``num_qutrits`` is not ``1`` or ``2``.
        """
        if num_qutrits not in (1, 2):
            raise ValueError(f"num_qutrits must be 1 or 2; got {num_qutrits}.")
        self._label: str = label
        self._num_qutrits: int = num_qutrits
        self._params: tuple[float, ...] = tuple(params)

    # ----- public API -----

    @property
    def label(self) -> str:
        """Gate name."""
        return self._label

    @property
    def num_qutrits(self) -> int:
        """1 or 2."""
        return self._num_qutrits

    @property
    def params(self) -> tuple[float, ...]:
        """Gate parameters."""
        return self._params

    @property
    def num_params(self) -> int:
        """len(params)."""
        return len(self._params)

    @abc.abstractmethod
    def matrix(self) -> NDArray[np.complex128]:
        """Return the unitary matrix (3x3 or 9x9)."""

    def inverse(self) -> Gate:
        """Return gate^dagger. Subclasses can override (e.g. Rx(-theta))."""
        return _DaggerGate(self)

    def is_unitary(self, atol: float = 1e-9) -> bool:
        """Check ``M @ M.conj().T == I`` within ``atol`` (default ``1e-9``)."""
        m = self.matrix()
        product = m @ m.conj().T
        return bool(np.allclose(product, np.eye(m.shape[0]), atol=atol))

    def __repr__(self) -> str:
        if self._params:
            param_str = ", ".join(f"{p:.4f}" for p in self._params)
            return f"{self._label}({param_str})"
        return self._label

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Gate):
            return NotImplemented
        return (
                self._label == other._label
                and self._num_qutrits == other._num_qutrits
                and np.allclose(self.matrix(), other.matrix(), atol=1e-12)
        )

    def __hash__(self) -> int:
        return hash((self._label, self._num_qutrits, self._params))


class _DaggerGate(Gate):
    """Internal wrapper for gate.inverse(). Not public API."""

    def __init__(self, original: Gate) -> None:
        super().__init__(
            label=f"{original.label}†",
            num_qutrits=original.num_qutrits,
            params=original.params,
        )
        self._original: Gate = original

    def matrix(self) -> NDArray[np.complex128]:
        return np.asarray(
            self._original.matrix().conj().T,
            dtype=np.complex128,
        )

    def inverse(self) -> Gate:
        # (A†)† = A
        return self._original


__all__ = ["Gate"]
