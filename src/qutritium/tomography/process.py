# MIT License — Copyright (c) 2023-2026 Son Pham, Tien Nguyen, Bao Bach, Charlie
# See LICENSE.txt for full terms.

"""Single-qutrit process tomography: Choi-matrix reconstruction from MUB data."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import qutritium.circuit.elementary_matrices as em
from qutritium.circuit.qutrit_circuit import QutritCircuit
from qutritium.gates.base import Gate
from qutritium.tomography.bases import mub_bases
from qutritium.tomography.state import _MUBWrapper, reconstruct_state, state_tomography_circuits


def _shift_from_zero(k: int) -> NDArray[np.complex128]:
    """Return a unitary taking from |0> to |k>."""
    if k == 0:
        return np.eye(3, dtype=complex)
    return em.x01() if k == 1 else em.x02()


def process_tomography_circuits(gate: Gate) -> tuple[list[list[QutritCircuit]], list[NDArray]]:
    """Build the circuits for single-qutrit process tomography of ``gate``.

    Same idea as state tomography, one level up: we prepare the 12 MUB states,
    push each one through ``gate``, and do state tomography on every output.
    Run the circuits, keep the counts in order, and hand them to
    ``reconstruct_process``.

    To see what the noise does, run the circuits on a ``DensityMatrixSimulator``
    that carries the noise model on the gate label -- the prep and the
    measurement feel it too.

    Parameters
    ----------
    gate : Gate
        The single-qutrit gate to characterize.

    Returns
    -------
    tuple[list[list[QutritCircuit]], list[NDArray]]
        ``(circuits, input_states)`` -- 12 groups of 4 measurement circuits,
        and the input density matrix each group prepares.

    Raises
    ------
    ValueError
        If ``gate`` acts on more than one qutrit.
    """
    if gate.num_qutrits != 1:
        raise ValueError(f"Process tomography only process single-qutrit gate in this version."
                         f"; got a {gate.num_qutrits} qutrits.")
    circuit_groups: list[list[QutritCircuit]] = []
    input_states: list[NDArray] = []
    for b, basis in enumerate(mub_bases()):
        for k in range(3):
            ket = basis[:, [k]]
            input_states.append(ket @ ket.conj().T)
            prep = QutritCircuit(1, None)
            # Shift the state |0> -> |k> -> B_b |k> -> |v_b^(k)>
            prep_unitary = basis @ _shift_from_zero(k)
            if not np.allclose(prep_unitary, np.eye(3)):
                prep.append(_MUBWrapper(prep_unitary, label=f"prep{b}{k}"), first_qutrit=0)
            prep.append(gate, first_qutrit=0)
            # Measure MUB basis
            circuit_groups.append(state_tomography_circuits(prep))
    return circuit_groups, input_states


def reconstruct_process(counts_per_input: list[list[dict[str, int]]],
                        input_states: list[NDArray]
                        ) -> NDArray[np.complex128]:
    """Reconstruct a channel's Choi matrix from process-tomography counts.

    First we run ``reconstruct_state`` on each group of counts to get the
    measured output state. The channel is linear, so on vectorized density
    matrices it is just one matrix ``M``,

        vec(E(rho)) = M vec(rho).

    Every known input and its reconstructed output give one column equation;
    we stack all of them (``M R_in = R_out``) and solve by least squares for
    ``M``. The Choi matrix then follows straight from its definition,

        J = sum_ij E(|i><j|) (x) |i><j|,

    by feeding ``M`` the nine elementary matrices ``E_ij``.

    Parameters
    ----------
    counts_per_input : list[list[dict[str, int]]]
        For each input state, the four MUB count-dicts in
        ``process_tomography_circuits`` order.
    input_states : list[NDArray]
        The matching input density matrices.

    Returns
    -------
    NDArray[np.complex128]
        The ``(9, 9)`` Hermitian Choi matrix.

    Raises
    ------
    ValueError
        If the two lists disagree in length, or there are fewer than 9 inputs.
    """
    if len(counts_per_input) != len(input_states):
        raise ValueError(
            f"Length mismatch: {len(counts_per_input)} count groups for "
            f"{len(input_states)} input states."
        )
    if len(input_states) < 9:
        raise ValueError("Process tomography needs at least 9 input states.")

    output_groups = [reconstruct_state(counts) for counts in counts_per_input]
    r_in = np.column_stack([np.asarray(rho, dtype=np.complex128).flatten() for rho in input_states])
    r_out = np.column_stack([rho.flatten() for rho in output_groups])
    # Solve for M matrix
    transfer_m, *_ = np.linalg.lstsq(r_in.T, r_out.T, rcond=None)
    transfer = transfer_m.T

    # Choi matrix form
    choi_matrix = np.zeros((9, 9), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            e_ij = np.zeros((3, 3), dtype=np.complex128)
            e_ij[i, j] = 1.0
            out = (transfer @ e_ij.flatten()).reshape(3, 3)
            choi_matrix += np.kron(out, e_ij)
    return (choi_matrix + choi_matrix.conj().T) / 2  # type: ignore[no-any-return]


def choi_to_kraus(choi: NDArray, atol: float = 1e-8) -> list[NDArray[np.complex128]]:
    """Pull the Kraus operators out of a Choi matrix.

    Eigendecompose ``J = sum_l lam_l |u_l><u_l|`` and reshape each eigenvector
    into a Kraus operator ``K_l = sqrt(lam_l) mat(u_l)``. Eigenvalues at or
    below ``atol`` are dropped, so a unitary gate hands back a single Kraus
    operator and a noisy one gives several.

    Parameters
    ----------
    choi : NDArray
        The ``(9, 9)`` Choi matrix from ``reconstruct_process``.
    atol : float, optional
        Eigenvalues at or below this are treated as zero and dropped.

    Returns
    -------
    list[NDArray[np.complex128]]
        The ``(3, 3)`` Kraus operators.

    Raises
    ------
    ValueError
        If ``choi`` is not ``(9, 9)`` or not Hermitian.
    """
    choi = np.asarray(choi, dtype=np.complex128)
    if choi.shape != (9, 9):
        raise ValueError(f"Choi matrix must have shape (9, 9); got {choi.shape}.")
    if not np.allclose(choi, choi.conj().T, atol=1e-8):
        raise ValueError("Choi matrix must be Hermitian.")
    eigvals, eigvecs = np.linalg.eigh(choi)
    kraus = [
        np.sqrt(val) * eigvecs[:, index].reshape(3, 3)
        for index, val in reversed(list(enumerate(eigvals))) if val > atol
    ]
    return kraus


__all__ = ["choi_to_kraus", "process_tomography_circuits", "reconstruct_process"]
