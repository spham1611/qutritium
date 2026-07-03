"""Regression tests for correctness bugs found in the v1.3.x review.

Each class pins a specific bug so it cannot silently return:
  1. metrics silently wrong for unnormalized kets / invalid density matrices
  2. two-qutrit adjacency check falsy-zero hole (second_qutrit == 0)
  3. CSUM/CSUMDag control/target swap when control > target
  4. Instruction.inverse() round-trip for custom (append-path) instructions
"""

from __future__ import annotations

import numpy as np
import pytest

import qutritium.circuit.elementary_matrices as em
from qutritium import QutritCircuit, purity, state_fidelity, von_neumann_entropy
from qutritium.circuit.instruction import Instruction
from qutritium.gates import CNOT3, CSUM, SWAP3, CPhase, CSUMDag, Rx01

_SWAP9 = em.swap3()


# ===================================================================
# 1. Metrics: unnormalized kets + invalid density matrices
# ===================================================================
class TestMetricsNormalization:
    def test_unnormalized_ket_self_fidelity_is_one(self):
        v = np.array([2, 0, 0], dtype=complex)
        assert state_fidelity(v, v) == pytest.approx(1.0)

    def test_unnormalized_ket_purity_is_one(self):
        v = np.array([0, 3, 0], dtype=complex)
        assert purity(v) == pytest.approx(1.0)

    def test_unnormalized_ket_entropy_is_zero(self):
        v = np.array([2, 0, 0], dtype=complex)
        assert von_neumann_entropy(v) == pytest.approx(0.0, abs=1e-12)

    def test_unnormalized_matches_normalized(self):
        v = np.array([1, 1, 1], dtype=complex)  # norm sqrt(3)
        w = v / np.sqrt(3)
        assert state_fidelity(v, w) == pytest.approx(1.0)

    def test_zero_norm_ket_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            state_fidelity(
                np.zeros(3, dtype=complex), np.array([1, 0, 0], dtype=complex)
            )

    def test_non_unit_trace_matrix_raises(self):
        bad = 2 * np.eye(3, dtype=complex) / 3  # trace 2
        with pytest.raises(ValueError, match="unit trace"):
            purity(bad)

    def test_non_hermitian_matrix_raises(self):
        bad = np.array(
            [[1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex
        )  # trace 1, not Hermitian
        with pytest.raises(ValueError, match="Hermitian"):
            purity(bad)


# ===================================================================
# 2. Adjacency check: second_qutrit == 0 must not slip through
# ===================================================================
class TestAdjacencyGuard:
    @pytest.mark.parametrize("first,second", [(2, 0), (0, 2)])
    def test_non_adjacent_rejected_either_order(self, first, second):
        with pytest.raises(ValueError, match="adjacent"):
            Instruction("csum", n_qutrit=3, first_qutrit=first, second_qutrit=second)

    @pytest.mark.parametrize("first,second", [(1, 0), (0, 1)])
    def test_adjacent_accepted(self, first, second):
        Instruction("csum", n_qutrit=3, first_qutrit=first, second_qutrit=second)


# ===================================================================
# 3. Two-qutrit orientation: control > target must SWAP-conjugate
# ===================================================================
def _effect(gate, first, second):
    qc = QutritCircuit(2, None)
    qc.append(gate, first_qutrit=first, second_qutrit=second)
    return qc.operation_set[0].effect_matrix


class TestTwoQutritOrientation:
    @pytest.mark.parametrize(
        "gate,mat",
        [
            (CSUM(), em.csum()),
            (CSUMDag(), em.csum_dag()),
            (CPhase(), em.cphase()),
            (SWAP3(), em.swap3()),
        ],
        ids=["CSUM", "CSUMDag", "CPhase", "SWAP3"],
    )
    def test_forward_is_raw_matrix(self, gate, mat):
        assert np.allclose(_effect(gate, 0, 1), mat)

    @pytest.mark.parametrize(
        "gate,mat",
        [
            (CSUM(), em.csum()),
            (CSUMDag(), em.csum_dag()),
            (CPhase(), em.cphase()),
            (SWAP3(), em.swap3()),
        ],
        ids=["CSUM", "CSUMDag", "CPhase", "SWAP3"],
    )
    def test_reversed_is_swap_conjugated(self, gate, mat):
        assert np.allclose(_effect(gate, 1, 0), _SWAP9 @ mat @ _SWAP9)

    def test_csum_reversed_physics(self):
        # control=q1=1, target=q0=0 -> target maps to (0+1)%3 = 1
        # input |q0=0, q1=1> = index 1; output should put target(q0) -> 1 => |1,1> = index 4
        u = _effect(CSUM(), 1, 0)
        psi_in = np.zeros((9, 1), dtype=complex)
        psi_in[1, 0] = 1.0
        out = u @ psi_in
        assert np.argmax(np.abs(out.flatten())) == 4

    def test_noncustom_cnot_unaffected_both_orders(self):
        for first, second in [(0, 1), (1, 0)]:
            ins = Instruction(
                "CNOT", n_qutrit=2, first_qutrit=first, second_qutrit=second
            )
            assert np.allclose(ins.effect_matrix, em.cnot(control=first, target=second))

    def test_cnot3_reversed_swaps(self):
        # custom CNOT3 uses a fixed control-on-left matrix -> needs the swap
        assert np.allclose(_effect(CNOT3(), 1, 0), _SWAP9 @ em.cnot(0, 1) @ _SWAP9)


# ===================================================================
# 4. Custom instruction inverse round-trips
# ===================================================================
class TestCustomInverseRoundTrip:
    def test_single_qutrit_double_inverse_is_base(self):
        qc = QutritCircuit(1, None)
        qc.append(Rx01(0.7), first_qutrit=0)
        ins = qc.operation_set[0]
        base = Rx01(0.7).matrix()
        assert np.allclose(ins.inverse().gate_matrix, base.conj().T)
        assert np.allclose(ins.inverse().inverse().gate_matrix, base)

    def test_two_qutrit_double_inverse_is_base(self):
        qc = QutritCircuit(2, None)
        qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
        ins = qc.operation_set[0]
        assert np.allclose(ins.inverse().inverse().gate_matrix, em.csum())
