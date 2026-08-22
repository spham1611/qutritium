"""
Smoke tests for the v1.0.0 hardware-agnostic core.

Migrated from the v0.0.1 ``test/VM_test/VM_test.py`` (which was a top-level
script using ``src.X`` imports). Restructured as proper pytest tests.
Updated for v1.1.0: uses Gate-based ``append()`` API.
"""

from __future__ import annotations

import numpy as np
import pytest

from qutritium import QutritCircuit, StatevectorSimulator, SU3Decomposition
from qutritium.gates import CSUM, G01, G12, H3, Rx01, Rx12, Ry12


# ---------------------------------------------------------------------------
# Circuit-level tests
# ---------------------------------------------------------------------------
def test_circuit_addition_preserves_operations() -> None:
    """``a + b`` should produce a fresh circuit with concatenated ops."""
    a = QutritCircuit(3, None)
    a.append(H3(), first_qutrit=0)
    a.append(Rx01(np.pi), first_qutrit=0)

    b = QutritCircuit(3, None)
    b.append(Rx01(np.pi), first_qutrit=0)
    b.append(H3(), first_qutrit=0)
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
        [[omega, 1.0, np.conj(omega)], [1.0, 1.0, 1.0], [np.conj(omega), 1.0, omega]],
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
def test_two_qutrit_hadamard_csum_distribution() -> None:
    """H3 + CSUM should produce a uniform mixture of {00, 11, 22}."""
    qc = QutritCircuit(2, None)
    qc.append(H3(), first_qutrit=0)
    qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
    qc.measure_all()
    sim = StatevectorSimulator(qc)
    sim.run(num_shots=20_000)
    counts = sim.get_counts()
    assert set(counts.keys()) <= {"00", "11", "22"}
    for outcome in ("00", "11", "22"):
        assert 6_000 <= counts[outcome] <= 7_500, counts


def test_simulator_run_without_measurement_raises() -> None:
    qc = QutritCircuit(1, None)
    qc.append(H3(), first_qutrit=0)
    sim = StatevectorSimulator(qc)
    with pytest.raises(RuntimeError):
        sim.run(num_shots=100)


def test_simulator_run_with_invalid_shots_raises() -> None:
    qc = QutritCircuit(1, None)
    qc.measure_all()
    sim = StatevectorSimulator(qc)
    with pytest.raises(ValueError):
        sim.run(num_shots=0)


def test_density_matrix_is_hermitian_psd() -> None:
    """Pure-state density matrix from the simulator must be Hermitian and PSD."""
    qc = QutritCircuit(1, None)
    qc.append(H3(), first_qutrit=0)
    sim = StatevectorSimulator(qc)
    rho = sim.density_matrix()
    assert np.allclose(rho, rho.conj().T)
    eigvals = np.linalg.eigvalsh(rho)
    assert np.all(eigvals >= -1e-10)
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# SU(3) decomposition: degenerate (coordinate-singular) inputs
# ---------------------------------------------------------------------------
def _reconstruction_error(target: np.ndarray, rebuilt: np.ndarray) -> float:
    """Largest entrywise error between ``rebuilt`` and ``target``.

    The global phase is unobservable, so it is divided out first.  Unlike
    ``|tr(U^dag R)| / 3`` this is sensitive to a *relative* phase between
    blocks, and it is linear rather than quadratic in the error.
    """
    overlap = np.trace(target.conj().T @ rebuilt)
    phase = float(np.angle(overlap)) if np.absolute(overlap) > 1e-12 else 0.0
    return float(np.max(np.absolute(rebuilt - np.exp(1j * phase) * target)))


def _degenerate_cases() -> list[np.ndarray]:
    """Unitaries that sit on a coordinate singularity of the nine-angle form.

    A pure ``{|1>, |2>}`` rotation has theta1 = theta3 = 0, and ``g01 . g12``
    has U[2,0] = 0; both make one or more of the angle inversions degenerate.
    Haar-random matrices never hit these, so the random round-trip tests above
    cannot see them.
    """
    thetas = [0.05, 0.7, 1.4, np.pi / 2, 2.5, 3.0]
    cases = [g(t).matrix() for t in thetas for g in (Ry12, Rx12)]
    cases += [G12(t, p).matrix() for t in thetas for p in (0.0, 0.9)]
    cases += [G01(t, 0.4).matrix() @ G12(t, 1.3).matrix() for t in thetas]
    return cases


@pytest.mark.parametrize("unitary", _degenerate_cases())
def test_degenerate_decomposition_roundtrip(unitary: np.ndarray) -> None:
    """Subspace gates must round-trip, not pick up a spurious relative phase."""
    dec = SU3Decomposition(unitary, qutrit_index=0, n_qutrits=1)
    assert _reconstruction_error(unitary, dec.reconstruct()) < 1e-9
    assert _reconstruction_error(unitary, dec.to_circuit().to_matrix()) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_random_decomposition_is_exact(seed: int) -> None:
    """Haar-random round-trip, at full double precision rather than ~1e-6."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.diag(r) / np.absolute(np.diag(r)))
    dec = SU3Decomposition(q, qutrit_index=0, n_qutrits=1)
    assert _reconstruction_error(q, dec.reconstruct()) < 1e-12
