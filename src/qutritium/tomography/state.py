# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Single-qutrit state tomography."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import qutritium.circuit.elementary_matrices as em
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.gates.base import Gate
from qutritium.tomography.bases import mub_bases


class _MUBWrapper(Gate):
    """Single-qutrit gate wrapping a fixed MUB rotation (a ``custom`` gate)."""

    def __init__(self, matrix: NDArray, label: str) -> None:
        """Ctor."""
        super().__init__(label, 1)
        self._matrix: NDArray = np.array(matrix, dtype=np.complex128)

    def matrix(self) -> NDArray[np.complex128]:
        return self._matrix


def _copy_prep(prep: QutritCircuit) -> QutritCircuit:
    """Clone the ``prep`` circuit. Does not mutate the original."""
    cloned_prep = QutritCircuit(prep.n_qutrit, prep.initial_state)
    cloned_prep.operation_set = [op for op in prep.operation_set if op != "measurement"]
    return cloned_prep


def _mub_projectors() -> list[NDArray]:
    """The 12 rank-1 projectors |v><v| of the four MUBs, basis-major order."""
    projectors: list[NDArray] = []
    for basis in mub_bases():
        for k in range(3):
            # Keep the vector in (3, 1) shape so v @ v.conj().T is an outer product.
            v = basis[:, [k]]
            projectors.append(v @ v.conj().T)
    return projectors


def _counts_to_probs(counts_basis: list[dict[str, int]]) -> NDArray:
    """Flatten four basis count-dicts into a length-12 probability vector.

    Parameters
    ----------
    counts_basis : list[dict[str, int]]
        Four MUB count-dicts, each mapping an outcome label to its shot count.

    Returns
    -------
    NDArray
        Flattened probability vector, basis-major (basis outer, outcome inner).

    Raises
    ------
    ValueError
        If ``counts_basis`` does not contain four MUBs, or a basis has no counts.
    """
    if len(counts_basis) != 4:
        raise ValueError(f"Counts basis must contain 4 MUBs; got {len(counts_basis)}.")
    probs: list[float] = []
    for counts in counts_basis:
        total = sum(counts.values())
        if total == 0:
            raise ValueError("A measurement basis has zero total counts.")
        for k in range(3):
            probs.append(counts.get(str(k), 0) / total)
    return np.asarray(probs, dtype=float)


def _project_closest_rho(rho: NDArray) -> NDArray[np.complex128]:
    """Project a unit-trace Hermitian matrix onto the closest density matrix.

    The linear least squares method in ``reconstruct_state`` can have negative
    entries -> To solve it, we implement algorithm introduced by
    Smollin, Gambetta & Smith (2012) which 'redistribute' the negative probabilities
    to other indices. The idea is to 0 out and spread the deficit over the rest.

    References
    ----------
    Smolin, J. A., Gambetta, J. M. & Smith, G. (2012). Efficient method for
    computing the maximum-likelihood quantum state from measurements with
    additive Gaussian noise. Phys. Rev. Lett. 108, 070502.
    """
    # eigh (rho is Hermitian) returns real eigenvalues ascending; reverse to descending.
    eigen_vals, eigen_vecs = np.linalg.eigh(rho)
    lam = eigen_vals[::-1].copy()
    a = 0.0
    for i in range(len(lam) - 1, -1, -1):
        if lam[i] + a / (i + 1) < 0:
            a += lam[i]
            lam[i] = 0.0
        else:
            lam[: i + 1] += a / (i + 1)
            break
    lam = lam[::-1]
    return (eigen_vecs * lam) @ eigen_vecs.conj().T  # type: ignore[no-any-return]


def state_tomography_circuits(prep: QutritCircuit) -> list[QutritCircuit]:
    """Build the four MUB measurement circuits.

    Each circuit follows prep state -> B_b^dag (basis rotation) -> measure.

    Parameters
    ----------
    prep : QutritCircuit
        Single-qutrit state-prep circuit, no measurement.

    Returns
    -------
    list[QutritCircuit]
        Four circuits, in ``mub_bases`` order. Counts are fed to ``reconstruct_state``.

    Raises
    ------
    ValueError
        If ``prep`` is not single-qutrit or already contains a measurement.
    """
    if prep.n_qutrit != 1:
        raise ValueError(
            f"State tomography is single-qutrit only in this version; got {prep.n_qutrit}."
        )
    if prep.measurement_flag:
        raise ValueError("Prep must not contain a measurement.")

    circuits: list[QutritCircuit] = []
    for b, basis in enumerate(mub_bases()):
        circuit = _copy_prep(prep)
        rotation = basis.conj().T  # B_b^dag
        # Skip appending the identity rotation (the computational basis).
        if not np.allclose(rotation, np.eye(3)):
            circuit.append(_MUBWrapper(rotation, label=f"MUB{b}dag"), first_qutrit=0)
        circuit.measure_all()
        circuits.append(circuit)
    return circuits


def reconstruct_state(
        counts_basis: list[dict[str, int]], method: str = "lls"
) -> NDArray[np.complex128]:
    """Reconstruct rho from MUB measurement counts via linear least squares.

    If we write the density matrix in the generalized Gell-Mann (GGM) basis,

        rho = I/3 + (1/2) sum_i^8 s_i lambda_i,

    then every measured probability is linear in the Bloch vector s, so we can
    just solve for it. Following Thew et al. (2002), s is the least-squares
    solution of

        s = argmin || A s - y ||^2,

    with ``A[m, i] = (1/2) Tr(P_m lambda_i)`` over the 12 MUB projectors ``P_m``
    and ``y_m = p_m - 1/3`` (``p_m`` is the measured probability of that outcome).
    After constructing ``rho``, user can choose the approximate the ``rho`` using
    ``_project_closest_rho`` which guarantees density matrix condition of semi-positive.

    Parameters
    ----------
    counts_basis : list[dict[str, int]]
        Four MUB count-dicts in ``state_tomography_circuits`` order, each
        mapping an outcome label to its shot count.
    method : str, optional
        ``"lls"`` / ``"linear_least_squares"`` for raw linear least squares, or
        ``"projected_lls"`` / ``"mle"`` to add the Smolin maximum-likelihood
        projection onto the closest physical state. Anything else raises.

    Returns
    -------
    NDArray[np.complex128]
        Shape ``(3, 3)``. This is an estimate and may be slightly non-PSD,
        since linear least squares does not enforce positivity. It is advised
        to use ``"projected_lls"`` option -> enforce such condition.

    Raises
    ------
    NotImplementedError
        If ``method`` is none of ``"lls"``, ``"linear_least_squares"``,
        ``"projected_lls"``, or ``"mle"``.
    ValueError
        Propagated from ``_counts_to_probs`` on malformed counts.

    References
    ----------
    Thew, R. T., Nemoto, K., White, A. G. & Munro, W. J. (2002). Qudit
    quantum-state tomography. Phys. Rev. A 66, 012303.
    """
    if method not in ("lls", "linear_least_squares", "projected_lls", "mle"):
        raise NotImplementedError(
            f"Method {method!r} is not implemented in this version; only "
            f"'lls' / 'linear_least_squares', 'projected_lls', or 'mle' is available."
        )

    projectors = _mub_projectors()
    p = _counts_to_probs(counts_basis)
    lamd = [
        em.lambda_1(),
        em.lambda_2(),
        em.lambda_3(),
        em.lambda_4(),
        em.lambda_5(),
        em.lambda_6(),
        em.lambda_7(),
        em.lambda_8(),
    ]
    a_mat = np.array(
        [[0.5 * np.real(np.trace(proj @ gen)) for gen in lamd] for proj in projectors]
    )
    y = p - 1.0 / 3.0
    # lls solver using numpy library
    s, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    rho = np.eye(3, dtype=complex) / 3.0 + 0.5 * sum(
        si * gen for si, gen in zip(s, lamd, strict=True)
    )
    # 'mle' is an alias for the Smolin maximum-likelihood projection.
    if method in ("projected_lls", "mle"):
        rho = _project_closest_rho(rho)
    return rho  # type: ignore[no-any-return]


__all__ = ["reconstruct_state", "state_tomography_circuits"]
