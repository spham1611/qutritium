"""Tests for QutritCircuit introspection methods: gate_count, depth, to_matrix, draw."""
from __future__ import annotations

import numpy as np
import pytest

import qutritium.circuit.elementary_matrices as em
from qutritium import Instruction, QASMSimulator, QutritCircuit, SU3Decomposition
from qutritium.gates import CSUM, H3, Rx01, X01


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


# ===================================================================
# measurement guard (a measurement must remain the final operation)
# ===================================================================
class TestMeasurementGuard:
    def test_append_after_measure_raises(self):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.measure_all()
        with pytest.raises(RuntimeError, match="measure"):
            qc.append(X01(), first_qutrit=0)

    def test_double_measure_raises(self):
        qc = QutritCircuit(1, None)
        qc.measure_all()
        with pytest.raises(RuntimeError):
            qc.measure_all()


class TestFromGate:
    def test_matrix_matches_gate(self):
        ins = Instruction._from_gate(H3(), n_qutrit=1, first_qutrit=0)
        assert ins._is_custom and ins.gate is not None
        assert np.allclose(ins.gate_matrix, H3().matrix())

    def test_append_unchanged(self):
        qc = QutritCircuit(1, None)
        qc.append(Rx01(0.6), first_qutrit=0)
        qc.append(X01(), first_qutrit=0)
        ins = qc.operation_set[0]
        assert ins.type == "Rx01" and list(ins.parameter) == [0.6]
        assert np.allclose(ins.gate_matrix, Rx01(0.6).matrix())


class ToNativeGatePath:
    def test_reconstructs_unitary(self):
        # to_native factors (with the virtual-Z phases) still compose to U
        u = H3().matrix()
        dec = SU3Decomposition(u, qutrit_index=0, n_qutrits=1)
        a = dec.angles
        product = (em.u_d(a.phi6, a.phi5, a.phi4)
                   @ dec.to_native().instructions[2].gate_matrix
                   @ dec.to_native().instructions[1].gate_matrix
                   @ dec.to_native().instructions[0].gate_matrix)
        assert np.allclose(product, u, atol=1e-6)

    def test_instructions_carry_gate_refs(self):
        dec = SU3Decomposition(H3().matrix(), qutrit_index=0, n_qutrits=1)
        native = dec.to_native()
        assert [i.type for i in native.instructions] == ["G01", "G12", "G01"]
        assert all(i.gate is not None and i._is_custom for i in native.instructions)

    def test_matrices_identical_to_string_path(self):
        dec = SU3Decomposition(H3().matrix(), qutrit_index=0, n_qutrits=1)
        a = dec.angles
        native = dec.to_native()
        assert np.allclose(native.instructions[0].gate_matrix, em.g01(a.theta1, a.phi1))
        assert np.allclose(native.instructions[1].gate_matrix, em.g12(a.theta2, a.phi2))
        assert np.allclose(native.instructions[2].gate_matrix, em.g01(a.theta3, a.phi3))
