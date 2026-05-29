"""Tests for QutritCircuit introspection methods: gate_count, depth, to_matrix, draw."""
from __future__ import annotations

import numpy as np
import pytest

from qutritium import QASMSimulator, QutritCircuit
from qutritium.gates import CSUM, H3, X01


# ===================================================================
# gate_count
# ===================================================================
class TestGateCount:
    def test_empty_circuit(self):
        assert QutritCircuit(2, None).gate_count() == 0

    def test_counts_gates_only(self):
        qc = QutritCircuit(2, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
        assert qc.gate_count() == 2

    def test_excludes_measurement(self):
        qc = QutritCircuit(2, None)
        qc.append(H3(), first_qutrit=0)
        qc.measure_all()
        assert qc.gate_count() == 1


# ===================================================================
# depth
# ===================================================================
class TestDepth:
    def test_empty_circuit(self):
        assert QutritCircuit(2, None).depth() == 0

    def test_serial_chain_on_one_qutrit(self):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(X01(), first_qutrit=0)
        qc.append(H3(), first_qutrit=0)
        assert qc.depth() == 3

    def test_parallel_gates_different_qutrits(self):
        qc = QutritCircuit(3, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(H3(), first_qutrit=1)
        qc.append(X01(), first_qutrit=2)
        assert qc.depth() == 1

    def test_two_qutrit_gate_occupies_both(self):
        qc = QutritCircuit(2, None)
        qc.append(H3(), first_qutrit=0)  # depth on q0 -> 1
        qc.append(CSUM(), first_qutrit=0, second_qutrit=1)  # both -> 2
        assert qc.depth() == 2

    def test_filter_two_qutrit_only(self):
        qc = QutritCircuit(2, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(H3(), first_qutrit=1)
        assert qc.depth(lambda ins: ins.second_qutrit is not None) == 0


# ===================================================================
# to_matrix
# ===================================================================
class TestToMatrix:
    def test_single_gate_matches_gate_matrix(self):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        assert np.allclose(qc.to_matrix(), H3().matrix(), atol=1e-12)

    def test_empty_circuit_is_identity(self):
        qc = QutritCircuit(2, None)
        assert np.allclose(qc.to_matrix(), np.eye(9), atol=1e-12)

    def test_composition_order(self):
        # to_matrix = (last gate) @ ... @ (first gate)
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(X01(), first_qutrit=0)
        expected = X01().matrix() @ H3().matrix()
        assert np.allclose(qc.to_matrix(), expected, atol=1e-12)

    def test_two_qutrit_shape(self):
        qc = QutritCircuit(2, None)
        qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
        assert qc.to_matrix().shape == (9, 9)

    def test_raises_on_measurement(self):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.measure_all()
        with pytest.raises(RuntimeError):
            qc.to_matrix()

    def test_result_is_unitary(self):
        qc = QutritCircuit(2, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
        u = qc.to_matrix()
        assert np.allclose(u @ u.conj().T, np.eye(9), atol=1e-12)


# ===================================================================
# reset_circuit (clean slate, including measurement flag)
# ===================================================================
class TestResetCircuit:
    def test_clears_operations(self):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.measure_all()
        qc.reset_circuit()
        assert len(qc) == 0

    def test_clears_measurement_flag(self):
        qc = QutritCircuit(1, None)
        qc.measure_all()
        qc.reset_circuit()
        assert qc.measurement_flag is False

    def test_can_rebuild_after_reset(self):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.measure_all()
        qc.reset_circuit()
        # full rebuild: append, re-measure, simulate
        qc.append(X01(), first_qutrit=0)
        qc.measure_all()
        sim = QASMSimulator(qc)
        sim.run(num_shots=100)
        assert sum(sim.get_counts().values()) == 100


# ===================================================================
# draw (returns string, does not print)
# ===================================================================
class TestDraw:
    def test_returns_string(self):
        qc = QutritCircuit(2, None)
        qc.append(H3(), first_qutrit=0)
        out = qc.draw()
        assert isinstance(out, str)
        assert "q0" in out and "q1" in out

    def test_empty_circuit_draws_bare_wires(self):
        qc = QutritCircuit(2, None)
        out = qc.draw()
        assert "q0" in out and "q1" in out

    def test_does_not_print(self, capsys):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.draw()
        captured = capsys.readouterr()
        assert captured.out == ""
