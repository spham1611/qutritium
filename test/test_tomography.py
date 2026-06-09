"""Tests for qutritium.tomography: MUB bases, circuits, state reconstruction.

Reconstruction is checked two ways: from exact (analytic) Born counts, which
is deterministic and must be near-perfect; and from a sampled simulation,
which must clear the fidelity > 0.95 acceptance threshold.
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
from qutritium.gates import G01, H3, X01
from qutritium.tomography import mub_bases, reconstruct_state, state_tomography_circuits


def _exact_counts(rho: np.ndarray, shots: int = 100_000) -> list[dict[str, int]]:
    """Born-rule counts for each MUB, computed exactly (no sampling)."""
    out: list[dict[str, int]] = []
    for basis in mub_bases():
        d: dict[str, int] = {}
        for k in range(3):
            v = basis[:, [k]]
            p = float(np.real((v.conj().T @ rho @ v)[0, 0]))
            d[str(k)] = round(max(p, 0.0) * shots)
        out.append(d)
    return out


def _prep_rho(gate=None) -> np.ndarray:
    qc = QutritCircuit(1, None)
    if gate is not None:
        qc.append(gate, first_qutrit=0)
    return DensityMatrixSimulator(qc).return_final_state()


# ===================================================================
# 1. MUB bases
# ===================================================================
class TestMUBBases:
    def test_four_bases(self):
        assert len(mub_bases()) == 4

    def test_first_is_computational(self):
        assert np.allclose(mub_bases()[0], np.eye(3))

    @pytest.mark.parametrize("idx", range(4))
    def test_each_is_unitary(self, idx):
        b = mub_bases()[idx]
        assert np.allclose(b @ b.conj().T, np.eye(3), atol=1e-9)

    def test_mutual_unbiasedness(self):
        bases = mub_bases()
        for i in range(4):
            for j in range(i + 1, 4):
                overlaps = np.abs(bases[i].conj().T @ bases[j]) ** 2
                assert np.allclose(overlaps, 1 / 3, atol=1e-9)


# ===================================================================
# 2. Tomography circuits
# ===================================================================
class TestTomographyCircuits:
    def _h3_prep(self) -> QutritCircuit:
        prep = QutritCircuit(1, None)
        prep.append(H3(), first_qutrit=0)
        return prep

    def test_returns_four_measured_circuits(self):
        circuits = state_tomography_circuits(self._h3_prep())
        assert len(circuits) == 4
        assert all(c.measurement_flag for c in circuits)

    def test_computational_basis_has_no_rotation(self):
        circuits = state_tomography_circuits(self._h3_prep())
        assert circuits[0].gate_count() == 1  # prep only (identity rotation skipped)
        assert circuits[1].gate_count() == 2  # prep + B_b^dag

    def test_does_not_mutate_prep(self):
        prep = self._h3_prep()
        state_tomography_circuits(prep)
        assert prep.measurement_flag is False
        assert prep.gate_count() == 1

    def test_rejects_multi_qutrit(self):
        with pytest.raises(ValueError, match="single-qutrit"):
            state_tomography_circuits(QutritCircuit(2, None))

    def test_rejects_measured_prep(self):
        prep = QutritCircuit(1, None)
        prep.measure_all()
        with pytest.raises(ValueError, match="measurement"):
            state_tomography_circuits(prep)


# ===================================================================
# 3. reconstruct_state
# ===================================================================
class TestReconstructState:
    @pytest.mark.parametrize("gate", [None, H3(), X01(), G01(0.7, 0.3)])
    def test_roundtrip_exact_counts(self, gate):
        rho_true = _prep_rho(gate)
        rho_est = reconstruct_state(_exact_counts(rho_true))
        assert state_fidelity(rho_true, rho_est) > 0.999
        assert np.trace(rho_est).real == pytest.approx(1.0, abs=1e-6)

    def test_roundtrip_mixed_state(self):
        rho_true = np.diag([0.5, 0.3, 0.2]).astype(complex)
        rho_est = reconstruct_state(_exact_counts(rho_true))
        assert state_fidelity(rho_true, rho_est) > 0.999

    def test_roundtrip_from_sampled_simulation(self):
        prep = QutritCircuit(1, None)
        prep.append(H3(), first_qutrit=0)
        rho_true = _prep_rho(H3())
        counts = []
        for circ in state_tomography_circuits(prep):
            sim = QASMSimulator(circ)
            sim.run(num_shots=50_000)
            counts.append(sim.get_counts())
        assert state_fidelity(rho_true, reconstruct_state(counts)) > 0.95

    @pytest.mark.parametrize("method", ["lls", "linear_least_squares"])
    def test_valid_methods(self, method):
        reconstruct_state(_exact_counts(_prep_rho(None)), method=method)  # must not raise

    def test_unknown_method_raises(self):
        with pytest.raises(NotImplementedError):
            reconstruct_state(_exact_counts(_prep_rho(None)), method="mle")

    def test_wrong_basis_count_raises(self):
        with pytest.raises(ValueError, match="4 MUBs"):
            reconstruct_state([{"0": 100}])

    def test_zero_counts_raises(self):
        with pytest.raises(ValueError, match="zero"):
            reconstruct_state([{"0": 0, "1": 0, "2": 0}] * 4)
