"""Tests for DensityMatrixSimulator and cross-backend agreement with QASMSimulator.

Organized as:
  1. Cross-backend: DM counts and final state match the statevector sim
  2. Density-matrix invariants: trace 1, Hermitian, diagonal = probabilities
  3. expectation_value
  4. partial_trace (Bell -> I/3 is the key indexing check)
  5. Index/endianness convention (qutrit 0 most significant)
"""
from __future__ import annotations

import numpy as np
import pytest

from qutritium import (
    DensityMatrixSimulator,
    QASMSimulator,
    QutritCircuit,
    state_fidelity,
)
from qutritium.gates import CSUM, H3, Rx01, X01


def _bell_circuit(measured: bool = True) -> QutritCircuit:
    qc = QutritCircuit(2, None)
    qc.append(H3(), first_qutrit=0)
    qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
    if measured:
        qc.measure_all()
    return qc


# Circuits exercised by the cross-backend comparison (all measured).
def _standard_circuits() -> list[QutritCircuit]:
    circuits = []

    qc = QutritCircuit(1, None)
    qc.append(H3(), first_qutrit=0)
    qc.measure_all()
    circuits.append(qc)

    qc = QutritCircuit(1, None)
    qc.append(X01(), first_qutrit=0)
    qc.measure_all()
    circuits.append(qc)

    qc = QutritCircuit(1, None)
    qc.append(Rx01(0.7), first_qutrit=0)
    qc.measure_all()
    circuits.append(qc)

    circuits.append(_bell_circuit())

    qc = QutritCircuit(2, None)
    qc.append(H3(), first_qutrit=0)
    qc.append(H3(), first_qutrit=1)
    qc.measure_all()
    circuits.append(qc)

    return circuits


# ===================================================================
# 1. Cross-backend agreement
# ===================================================================
class TestCrossBackend:
    @pytest.mark.parametrize("qc", _standard_circuits())
    def test_counts_agree_within_sampling(self, qc):
        shots = 20_000
        sv = QASMSimulator(qc)
        dm = DensityMatrixSimulator(qc)
        sv.run(num_shots=shots)
        dm.run(num_shots=shots)
        sv_counts = sv.get_counts()
        dm_counts = dm.get_counts()
        # Same support
        assert set(sv_counts) == set(dm_counts)
        # Frequencies agree within a loose sampling tolerance
        for key in sv_counts:
            assert abs(sv_counts[key] - dm_counts[key]) / shots < 0.05

    @pytest.mark.parametrize("qc", _standard_circuits())
    def test_final_state_fidelity_is_one(self, qc):
        sv = QASMSimulator(qc)
        dm = DensityMatrixSimulator(qc)
        rho_dm = dm.return_final_state()
        rho_sv = sv.density_matrix()
        # Pure (rank-1) states go through two eigendecompositions in
        # state_fidelity, losing ~7 digits; 1e-6 is well within that.
        assert state_fidelity(rho_dm, rho_sv) == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# 2. Density-matrix invariants
# ===================================================================
class TestDensityMatrixInvariants:
    def test_trace_is_one(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        rho = dm.return_final_state()
        assert np.trace(rho).real == pytest.approx(1.0)

    def test_state_is_hermitian(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        rho = dm.return_final_state()
        assert np.allclose(rho, rho.conj().T, atol=1e-12)

    def test_shape_is_dimension_squared(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        rho = dm.return_final_state()
        assert rho.shape == (9, 9)

    def test_probabilities_are_diagonal(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        probs = dm.probabilities()
        rho = dm.return_final_state()
        assert np.allclose(probs, np.real(np.diag(rho)), atol=1e-12)

    def test_probabilities_sum_to_one(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        assert dm.probabilities().sum() == pytest.approx(1.0)


# ===================================================================
# 3. expectation_value
# ===================================================================
class TestExpectationValue:
    def test_z01_on_ground_state(self):
        # |0>: Z01 = diag(1, -1, 1) -> <Z01> = +1
        qc = QutritCircuit(1, None)
        dm = DensityMatrixSimulator(qc)
        z01 = np.diag([1, -1, 1]).astype(complex)
        assert dm.expectation_value(z01) == pytest.approx(1.0)

    def test_z01_after_x01(self):
        # X01|0> = |1> -> <Z01> = -1
        qc = QutritCircuit(1, None)
        qc.append(X01(), first_qutrit=0)
        dm = DensityMatrixSimulator(qc)
        z01 = np.diag([1, -1, 1]).astype(complex)
        assert dm.expectation_value(z01) == pytest.approx(-1.0)

    def test_rejects_wrong_shape(self):
        dm = DensityMatrixSimulator(QutritCircuit(1, None))
        with pytest.raises(ValueError, match="shape"):
            dm.expectation_value(np.eye(2, dtype=complex))

    def test_rejects_non_hermitian(self):
        dm = DensityMatrixSimulator(QutritCircuit(1, None))
        non_herm = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
        with pytest.raises(ValueError, match="Hermitian"):
            dm.expectation_value(non_herm)


# ===================================================================
# 4. partial_trace
# ===================================================================
class TestPartialTrace:
    def test_bell_reduced_is_maximally_mixed(self):
        # Tracing out either qutrit of a maximally entangled pair -> I/3
        dm = DensityMatrixSimulator(_bell_circuit())
        reduced = dm.partial_trace([0])
        assert np.allclose(reduced, np.eye(3) / 3, atol=1e-12)

    def test_bell_reduced_qutrit1(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        reduced = dm.partial_trace([1])
        assert np.allclose(reduced, np.eye(3) / 3, atol=1e-12)

    def test_keep_all_is_full_state(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        full = dm.return_final_state()
        reduced = dm.partial_trace([0, 1])
        assert np.allclose(reduced, full, atol=1e-12)

    def test_product_state_reduced_is_pure(self):
        # |+>|0> with |+> = (|0>+|1>)/sqrt2 on qutrit 0 (via Rx01? no—use H3 truncated)
        # Build |+>_{01} on qutrit 0 by Ry01(pi/2): keeps within {0,1}
        from qutritium.gates import Ry01
        qc = QutritCircuit(2, None)
        qc.append(Ry01(np.pi / 2), first_qutrit=0)  # (|0>+|1>)/sqrt2 on qutrit 0
        dm = DensityMatrixSimulator(qc)
        reduced = dm.partial_trace([0])
        # Pure state -> purity 1
        assert np.real(np.trace(reduced @ reduced)) == pytest.approx(1.0, abs=1e-9)

    def test_rejects_empty(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        with pytest.raises(ValueError, match="empty"):
            dm.partial_trace([])

    def test_rejects_duplicates(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        with pytest.raises(ValueError, match="duplicate"):
            dm.partial_trace([0, 0])

    def test_rejects_out_of_range(self):
        dm = DensityMatrixSimulator(_bell_circuit())
        with pytest.raises(ValueError, match="range"):
            dm.partial_trace([5])


# ===================================================================
# 5. Index / endianness convention
# ===================================================================
class TestIndexConvention:
    def test_qutrit0_is_most_significant_sv(self):
        # X01 on qutrit 0 only: |00> -> |10>, NOT |01>
        qc = QutritCircuit(2, None)
        qc.append(X01(), first_qutrit=0)
        qc.measure_all()
        sim = QASMSimulator(qc)
        sim.run(num_shots=100)
        assert sim.get_counts() == {"10": 100}

    def test_qutrit0_is_most_significant_dm(self):
        qc = QutritCircuit(2, None)
        qc.append(X01(), first_qutrit=0)
        qc.measure_all()
        sim = DensityMatrixSimulator(qc)
        sim.run(num_shots=100)
        assert sim.get_counts() == {"10": 100}
