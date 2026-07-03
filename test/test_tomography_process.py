"""Tests for qutritium.tomography.process and the PSD projection."""

from __future__ import annotations

import numpy as np
import pytest

from qutritium import DensityMatrixSimulator
from qutritium.channels import NoiseModel, depolarizing_channel
from qutritium.gates import CSUM, H3, I3, X01
from qutritium.tomography import (
    choi_to_kraus,
    process_tomography_circuits,
    reconstruct_process,
)

I9 = np.eye(9, dtype=complex)


def _exact_counts(circuit, shots: int = 90_000) -> dict[str, int]:
    probs = DensityMatrixSimulator(circuit).probabilities()
    return {str(k): round(float(p) * shots) for k, p in enumerate(probs)}


def _run_pt(gate, noise_model=None):
    groups, inputs = process_tomography_circuits(gate)
    counts = []
    for group in groups:
        per_basis = []
        for circ in group:
            if noise_model is None:
                per_basis.append(_exact_counts(circ))
            else:
                dm = DensityMatrixSimulator(circ)
                dm.set_noise_model(noise_model)
                probs = dm.probabilities()
                per_basis.append(
                    {str(k): round(float(p) * 90_000) for k, p in enumerate(probs)}
                )
        counts.append(per_basis)
    return reconstruct_process(counts, inputs)


def _choi_of_unitary(u: np.ndarray) -> np.ndarray:
    omega = np.zeros((9, 1), dtype=complex)
    for i in range(3):
        omega[3 * i + i] = 1.0
    v = np.kron(u, np.eye(3)) @ omega
    return v @ v.conj().T


class TestReconstructProcess:
    def test_identity_channel(self):
        choi = _run_pt(I3())
        assert np.allclose(choi, _choi_of_unitary(np.eye(3)), atol=1e-3)
        assert np.trace(choi).real == pytest.approx(3.0, abs=1e-3)

    @pytest.mark.parametrize("gate", [X01(), H3()], ids=["X01", "H3"])
    def test_unitary_channel(self, gate):
        choi = _run_pt(gate)
        assert np.allclose(choi, _choi_of_unitary(gate.matrix()), atol=1e-3)

    def test_depolarizing_channel(self):
        # X01 followed by depolarizing(0.3):
        # J = (1-p) J_X01 + p * I9 / 3
        nm = NoiseModel()
        nm.add_quantum_error(depolarizing_channel(0.3), "X01")
        choi = _run_pt(X01(), noise_model=nm)
        expected = 0.7 * _choi_of_unitary(X01().matrix()) + 0.3 * I9 / 3
        assert np.allclose(choi, expected, atol=1e-3)

    def test_rejects_two_qutrit_gate(self):
        with pytest.raises(ValueError, match="single-qutrit"):
            process_tomography_circuits(CSUM())

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="count groups"):
            reconstruct_process([], [np.eye(3) / 3])


class TestChoiToKraus:
    def test_identity_gives_identity_kraus(self):
        kraus = choi_to_kraus(_choi_of_unitary(np.eye(3)))
        assert len(kraus) == 1
        # unitary freedom is a global phase only for a rank-1 Choi
        assert np.allclose(kraus[0].conj().T @ kraus[0], np.eye(3), atol=1e-9)

    def test_completeness_for_noisy_channel(self):
        nm = NoiseModel()
        nm.add_quantum_error(depolarizing_channel(0.3), "X01")
        choi = _run_pt(X01(), noise_model=nm)
        kraus = choi_to_kraus(choi, atol=1e-6)
        total = sum(k.conj().T @ k for k in kraus)
        assert np.allclose(total, np.eye(3), atol=1e-2)

    def test_kraus_rebuild_channel_action(self):
        nm = NoiseModel()
        nm.add_quantum_error(depolarizing_channel(0.3), "X01")
        choi = _run_pt(X01(), noise_model=nm)
        kraus = choi_to_kraus(choi, atol=1e-6)
        rho_in = np.diag([1.0, 0, 0]).astype(complex)  # |0><0|
        rho_out = sum(k @ rho_in @ k.conj().T for k in kraus)
        x = X01().matrix()
        expected = 0.7 * (x @ rho_in @ x.conj().T) + 0.3 * np.eye(3) / 3
        assert np.allclose(rho_out, expected, atol=1e-2)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            choi_to_kraus(np.eye(3))

    def test_rejects_non_hermitian(self):
        bad = np.zeros((9, 9), dtype=complex)
        bad[0, 1] = 1.0
        with pytest.raises(ValueError, match="Hermitian"):
            choi_to_kraus(bad)
