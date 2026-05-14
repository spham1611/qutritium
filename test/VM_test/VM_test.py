"""
Smoke tests for the v1.0.0 hardware-agnostic core.

Migrated from the v0.0.1 ``test/VM_test/VM_test.py`` (which was a top-level
script using ``src.X`` imports). Restructured as proper pytest tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from qutritium import QASM_Simulator, QutritCircuit, SU3Decomposition


# ---------------------------------------------------------------------------
# Circuit-level tests
# ---------------------------------------------------------------------------
def test_circuit_addition_preserves_operations() -> None:
    """``a + b`` should produce a fresh circuit with concatenated ops."""
    a = QutritCircuit(3, None)
    a.add_gate("hdm", first_qutrit=0)
    a.add_gate("rx01", first_qutrit=0, parameter=[np.pi])

    b = QutritCircuit(3, None)
    b.add_gate("rx01", first_qutrit=0, parameter=[np.pi])
    b.add_gate("hdm", first_qutrit=0)
    b.measure_all()

    c = a + b
    assert len(c) == len(a) + len(b)
    # ``a`` must not have been mutated.
    assert not a.measurement_flag
    assert c.measurement_flag


def test_circuit_addition_rejects_mismatched_qutrit_count() -> None:
    a = QutritCircuit(2, None)
    b = QutritCircuit(3, None)
    with pytest.raises(ValueError):
        _ = a + b


def test_double_measurement_raises() -> None:
    qc = QutritCircuit(1, None)
    qc.measure_all()
    with pytest.raises(RuntimeError):
        qc.measure_all()


# ---------------------------------------------------------------------------
# SU(3) decomposition reconstruction
# ---------------------------------------------------------------------------
def test_uft_decomposition_roundtrip() -> None:
    """The discrete Fourier matrix U_ft should round-trip to fidelity ~1."""
    omega = np.exp(1j * 2 * np.pi / 3)
    u_ft = (1 / np.sqrt(3)) * np.array(
        [[omega, 1.0, np.conj(omega)],
         [1.0, 1.0, 1.0],
         [np.conj(omega), 1.0, omega]],
        dtype=complex,
    )
    dec = SU3Decomposition(u_ft, qutrit_index=0, n_qutrits=1)
    fidelity = np.abs(np.trace(u_ft.conj().T @ dec.reconstruct())) / 3
    assert fidelity == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_random_su3_decomposition_roundtrip(seed: int) -> None:
    """Random Haar-ish unitaries should decompose-then-reconstruct cleanly."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    dec = SU3Decomposition(Q, qutrit_index=0, n_qutrits=1)
    fidelity = np.abs(np.trace(Q.conj().T @ dec.reconstruct())) / 3
    assert fidelity == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def test_two_qutrit_hadamard_cnot_distribution() -> None:
    """H_q0 + CNOT(q1, q0) should produce a uniform mixture of {00, 11, 22}."""
    qc = QutritCircuit(2, None)
    qc.add_gate("hdm", first_qutrit=0)
    qc.add_gate("CNOT", first_qutrit=1, second_qutrit=0)
    qc.measure_all()
    sim = QASM_Simulator(qc)
    sim.run(num_shots=20_000)
    counts = sim.get_counts()
    # Expect roughly equal weight on 00, 11, 22 and ~zero on the rest.
    assert set(counts.keys()) <= {"00", "11", "22"}
    for outcome in ("00", "11", "22"):
        # 5-sigma confidence interval on a multinomial is plenty loose.
        assert 6_000 <= counts[outcome] <= 7_500, counts


def test_simulator_run_without_measurement_raises() -> None:
    qc = QutritCircuit(1, None)
    qc.add_gate("hdm", first_qutrit=0)
    sim = QASM_Simulator(qc)
    with pytest.raises(RuntimeError):
        sim.run(num_shots=100)


def test_simulator_run_with_invalid_shots_raises() -> None:
    qc = QutritCircuit(1, None)
    qc.measure_all()
    sim = QASM_Simulator(qc)
    with pytest.raises(ValueError):
        sim.run(num_shots=0)


def test_density_matrix_is_hermitian_psd() -> None:
    """Pure-state density matrix from the simulator must be Hermitian and PSD."""
    qc = QutritCircuit(1, None)
    qc.add_gate("hdm", first_qutrit=0)
    sim = QASM_Simulator(qc)
    rho = sim.density_matrix()
    assert np.allclose(rho, rho.conj().T)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-10)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)
