# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Instruction: one gate applied to specific qutrit(s) in a circuit."""
from __future__ import annotations

from functools import cached_property
from typing import Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import qutritium.circuit.elementary_matrices as em

if TYPE_CHECKING:
    from qutritium.gates.base import Gate

GATE_SET: frozenset[str] = frozenset({
    "identity",
    "x_plus", "x_minus",
    "sdg", "tdg",
    "CNOT",
    "g01", "g02", "g12",
    "x01", "x02", "x12",
    "y01", "y02", "y12",
    "z01", "z02", "z12",
    "rx01", "rx02", "rx12",
    "ry01", "ry02", "ry12",
    "rz01", "rz02", "rz12",
    "u_d",
    "hdm", "u_ft",
    # Two-qutrit gates (v1.1.0)
    "csum", "csum_dag", "cphase", "swap3",
})


class Instruction:
    """One gate application in a circuit. ``effect_matrix`` is lazy.

    Most users construct ``Instruction`` indirectly via
    ``QutritCircuit.append``; the direct constructor is the lower-level
    path used by the string-based API and by ``SU3Decomposition``.

    Parameters
    ----------
    gate_type : str
        Gate name; must be in ``GATE_SET`` unless ``custom=True``.
    n_qutrit : int
        Total qutrit count for the register this instruction targets.
    first_qutrit : int
        Target qutrit index in ``[0, n_qutrit)``. Control qutrit for
        two-qutrit gates.
    second_qutrit : int or None, optional
        Second qutrit for two-qutrit gates (target). ``None`` for
        single-qutrit gates.
    parameter : Sequence[float] or None, optional
        Numerical parameters required by parametric gates (e.g.
        rotation angles for ``rx01``).
    inverse : bool, optional
        Apply the conjugate transpose of the resolved matrix.
    custom : bool, optional
        Set to ``True`` to bypass the ``GATE_SET`` lookup and supply a
        matrix via ``custom_matrix``.
    custom_matrix : NDArray or None, optional
        Required when ``custom=True``. Must be ``(3, 3)`` for a
        single-qutrit gate or ``(9, 9)`` for a two-qutrit gate.
    gate : Gate or None, optional
        Optional reference to the originating ``Gate`` object. Preserved
        for ``QutritCircuit.draw`` to render correct labels.

    Raises
    ------
    IndexError
        If ``first_qutrit`` or ``second_qutrit`` is outside
        ``[0, n_qutrit)``.
    ValueError
        If ``gate_type`` is unknown (and ``custom=False``), if
        ``custom=True`` without a ``custom_matrix``, or if
        ``custom_matrix`` has the wrong shape for the gate width.
    """

    def __init__(
            self,
            gate_type: str,
            n_qutrit: int,
            first_qutrit: int,
            second_qutrit: int | None = None,
            parameter: Sequence[float] | None = None,
            inverse: bool = False,
            custom: bool = False,
            custom_matrix: NDArray | None = None,
            gate: "Gate | None" = None,
    ) -> None:
        # Validate qutrit range
        if not 0 <= first_qutrit < n_qutrit:
            raise IndexError(
                f"first_qutrit={first_qutrit} is out of range "
                f"[0, {n_qutrit})."
            )
        if second_qutrit is not None and not 0 <= second_qutrit < n_qutrit:
            raise IndexError(
                f"second_qutrit={second_qutrit} is out of range "
                f"[0, {n_qutrit})."
            )

        self._type: str = gate_type
        self.n_qutrit: int = n_qutrit
        self.qutrit_dimension: int = 3 ** n_qutrit
        self.first_qutrit: int = first_qutrit
        self.second_qutrit: int | None = second_qutrit
        self.parameter: Sequence[float] | None = parameter
        self._is_inverse: bool = inverse
        self._is_custom: bool = custom
        self._is_two_qutrit_gate: bool = second_qutrit is not None
        self.gate: "Gate | None" = gate  # Gate reference for introspection

        if not custom:
            self._verify_gate()
        elif custom_matrix is None:
            raise ValueError("custom=True requires custom_matrix to be supplied.")

        # Validate custom matrix dimensions
        if custom and custom_matrix is not None:
            if self._is_two_qutrit_gate:
                expected = 9
            else:
                expected = 3
            if custom_matrix.shape != (expected, expected):
                raise ValueError(
                    f"custom_matrix has shape {custom_matrix.shape}; "
                    f"expected ({expected}, {expected}) for a "
                    f"{'two' if self._is_two_qutrit_gate else 'single'}-qutrit gate."
                )

        base_matrix: NDArray

        if custom:
            assert custom_matrix is not None
            base_matrix = custom_matrix
        else:
            base_matrix = self._resolve_gate_matrix()

        self.gate_matrix: NDArray = (
            np.asarray(base_matrix).conj().T if self._is_inverse
            else np.asarray(base_matrix)
        )

    # ------------------------------------------------------------------
    # Gate name -> matrix dispatch
    # ------------------------------------------------------------------
    def _require_params(self, n: int) -> Sequence[float]:
        """Check we have enough parameters."""
        if self.parameter is None or len(self.parameter) < n:
            raise ValueError(
                f"Gate '{self._type}' requires {n} parameter(s); got "
                f"{0 if self.parameter is None else len(self.parameter)}."
            )
        return self.parameter

    def _resolve_gate_matrix(self) -> NDArray[np.complex128]:
        """Map gate name -> matrix from elementary_matrices.py"""
        gt = self._type

        # Static single-qutrit gates
        if gt == "identity":
            return em.identity()
        if gt == "x01":
            return em.x01()
        if gt == "x02":
            return em.x02()
        if gt == "x12":
            return em.x12()
        if gt == "y01":
            return em.y01()
        if gt == "y02":
            return em.y02()
        if gt == "y12":
            return em.y12()
        if gt == "z01":
            return em.z01()
        if gt == "z02":
            return em.z02()
        if gt == "z12":
            return em.z12()
        if gt == "x_plus":
            return em.x_plus()
        if gt == "x_minus":
            return em.x_minus()

        # Omega-dependent gates
        if gt == "hdm":
            return em.hdm()
        if gt == "u_ft":
            return em.u_ft()
        if gt == "sdg":
            return em.sdg()
        if gt == "tdg":
            return em.tdg()

        # Single-parameter rotation gates
        if gt == "rx01":
            p = self._require_params(1)
            return em.rx01(p[0])
        if gt == "rx02":
            p = self._require_params(1)
            return em.rx02(p[0])
        if gt == "rx12":
            p = self._require_params(1)
            return em.rx12(p[0])
        if gt == "ry01":
            p = self._require_params(1)
            return em.ry01(p[0])
        if gt == "ry02":
            p = self._require_params(1)
            return em.ry02(p[0])
        if gt == "ry12":
            p = self._require_params(1)
            return em.ry12(p[0])
        if gt == "rz01":
            p = self._require_params(1)
            return em.rz01(p[0])
        if gt == "rz02":
            p = self._require_params(1)
            return em.rz02(p[0])
        if gt == "rz12":
            p = self._require_params(1)
            return em.rz12(p[0])

        # Two-parameter generalized rotation gates
        if gt == "g01":
            p = self._require_params(2)
            return em.g01(p[0], p[1])
        if gt == "g02":
            p = self._require_params(2)
            return em.g02(p[0], p[1])
        if gt == "g12":
            p = self._require_params(2)
            return em.g12(p[0], p[1])

        # Three-parameter diagonal phase gate
        if gt == "u_d":
            p = self._require_params(3)
            return em.u_d(p[0], p[1], p[2])

        # Two-qutrit gates (v1.1.0)
        if gt == "CNOT":
            assert self.second_qutrit is not None
            return em.cnot(control=self.first_qutrit, target=self.second_qutrit)
        if gt == "csum":
            return em.csum()
        if gt == "csum_dag":
            return em.csum_dag()
        if gt == "cphase":
            return em.cphase()
        if gt == "swap3":
            return em.swap3()

        raise KeyError(f"Unknown gate type: {gt!r}.")

    @cached_property
    def effect_matrix(self) -> NDArray:
        """Full-register effect matrix, computed lazily on first access."""
        if self._is_two_qutrit_gate:
            assert self.second_qutrit is not None
            lo = min(self.first_qutrit, self.second_qutrit)
            hi = max(self.first_qutrit, self.second_qutrit)
            left_dim = 3 ** lo
            right_dim = 3 ** (self.n_qutrit - hi - 1)
        else:
            left_dim = 3 ** self.first_qutrit
            right_dim = 3 ** (self.n_qutrit - self.first_qutrit - 1)

        if left_dim == 1 and right_dim == 1:
            return self.gate_matrix
        return np.kron(
            np.kron(np.eye(left_dim, dtype=complex), self.gate_matrix),
            np.eye(right_dim, dtype=complex),
        )

    def _verify_gate(self) -> None:
        """Check gate name is in GATE_SET."""
        if self._type not in GATE_SET:
            raise ValueError(
                f"Gate type {self._type!r} is not in GATE_SET. "
                f"Supported gates: {sorted(GATE_SET)}."
            )

    @property
    def type(self) -> str:
        """Gate name."""
        return self._type

    def describe(self) -> str:
        """Human-readable description of this instruction."""
        if not self._is_two_qutrit_gate:
            if self.parameter is None:
                return f"Gate {self._type}, first_qutrit: {self.first_qutrit}"
            return (
                f"Gate {self._type} with parameter {list(self.parameter)}, "
                f"first_qutrit: {self.first_qutrit}"
            )
        return (
            f"Gate {self._type}, first_qutrit (control): {self.first_qutrit}, "
            f"second_qutrit (target): {self.second_qutrit}"
        )

    def inverse(self) -> Instruction:
        """Return the conjugate-transpose instruction."""
        inverted_gate = self.gate.inverse() if self.gate is not None else None
        return Instruction(
            gate_type=self._type,
            n_qutrit=self.n_qutrit,
            first_qutrit=self.first_qutrit,
            second_qutrit=self.second_qutrit,
            parameter=self.parameter,
            inverse=not self._is_inverse,
            custom=self._is_custom,
            custom_matrix=self.gate_matrix if self._is_custom else None,
            gate=inverted_gate,
        )
