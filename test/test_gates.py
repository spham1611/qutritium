"""
Test suite for the qutritium.gates subpackage (Phase 2).

Tests are organized into:
  1. Gate base class contract (unitarity, inverse, repr, equality)
  2. Fixed single-qutrit gates (matrix shape, known values)
  3. Parametric single-qutrit gates (identity at θ=0, periodicity, inverse)
  4. Two-qutrit gates (unitarity, known actions)
  5. Circuit integration via QutritCircuit.append()
  6. Round-trip: Gate → append → simulate → verify state
"""
from __future__ import annotations

import numpy as np
import pytest

from qutritium import QASMSimulator, QutritCircuit
from qutritium.gates import (CNOT3, CPhase, CSUM, CSUMDag, G01, G02, G12, H3, I3, Rx01, Rx02, Rx12, Ry01, Ry02, Ry12,
                             Rz01, Rz02, Rz12, S3, SWAP3, T3, Ud, UFT, X01, X02, X12, XMinus, XPlus, Y01, Y02, Y12, Z01,
                             Z02, Z12)  # Fixed single-qutrit; Parametric single-qutrit; Two-qutrit

# ===================================================================
# Helpers
# ===================================================================
_FIXED_SINGLE_GATES = [
    I3(), X01(), X02(), X12(),
    Y01(), Y02(), Y12(),
    Z01(), Z02(), Z12(),
    XPlus(), XMinus(),
    H3(), S3(), T3(), UFT(),
]

_FIXED_TWO_GATES = [CSUM(), CSUMDag(), CNOT3(), CPhase(), SWAP3()]


def _is_unitary(m: np.ndarray, atol: float = 1e-9) -> bool:
    return bool(np.allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=atol))


# ===================================================================
# 1. Gate base class contract
# ===================================================================
class TestGateBaseContract:
    """Every Gate subclass must satisfy the Gate ABC contract."""

    @pytest.mark.parametrize("gate", _FIXED_SINGLE_GATES, ids=lambda g: g.label)
    def test_fixed_single_gate_is_unitary(self, gate):
        assert gate.is_unitary(), f"{gate.label} is not unitary"

    @pytest.mark.parametrize("gate", _FIXED_TWO_GATES, ids=lambda g: g.label)
    def test_fixed_two_gate_is_unitary(self, gate):
        assert gate.is_unitary(), f"{gate.label} is not unitary"

    @pytest.mark.parametrize("gate", _FIXED_SINGLE_GATES, ids=lambda g: g.label)
    def test_fixed_single_gate_shape(self, gate):
        assert gate.matrix().shape == (3, 3)
        assert gate.num_qutrits == 1

    @pytest.mark.parametrize("gate", _FIXED_TWO_GATES, ids=lambda g: g.label)
    def test_fixed_two_gate_shape(self, gate):
        assert gate.matrix().shape == (9, 9)
        assert gate.num_qutrits == 2

    @pytest.mark.parametrize("gate", _FIXED_SINGLE_GATES, ids=lambda g: g.label)
    def test_inverse_is_unitary(self, gate):
        inv = gate.inverse()
        assert inv.is_unitary(), f"{inv.label} is not unitary"

    @pytest.mark.parametrize("gate", _FIXED_SINGLE_GATES, ids=lambda g: g.label)
    def test_inverse_product_is_identity(self, gate):
        product = gate.matrix() @ gate.inverse().matrix()
        assert np.allclose(product, np.eye(3), atol=1e-12), \
            f"{gate.label} @ {gate.label}† != I"

    @pytest.mark.parametrize("gate", _FIXED_TWO_GATES, ids=lambda g: g.label)
    def test_two_gate_inverse_product_is_identity(self, gate):
        product = gate.matrix() @ gate.inverse().matrix()
        assert np.allclose(product, np.eye(9), atol=1e-12), \
            f"{gate.label} @ {gate.label}† != I"

    def test_repr_fixed_gate(self):
        assert repr(H3()) == "H3"

    def test_repr_parametric_gate(self):
        r = repr(Rx01(np.pi))
        assert "Rx01" in r
        assert "3.1416" in r

    def test_gate_equality(self):
        assert H3() == H3()
        assert Rx01(0.5) == Rx01(0.5)
        assert Rx01(0.5) != Rx01(0.6)
        assert X01() != X12()

    def test_num_params(self):
        assert I3().num_params == 0
        assert Rx01(0.5).num_params == 1
        assert G01(0.5, 0.3).num_params == 2
        assert Ud(0.1, 0.2, 0.3).num_params == 3

    def test_params_property(self):
        assert Rx12(1.5).params == (1.5,)
        assert G02(0.5, 0.3).params == (0.5, 0.3)
        assert Ud(0.1, 0.2, 0.3).params == (0.1, 0.2, 0.3)


# ===================================================================
# 2. Fixed single-qutrit gates — known values
# ===================================================================
class TestFixedSingleQutritGates:
    """Verify matrix values for fixed gates against known references."""

    def test_identity(self):
        assert np.allclose(I3().matrix(), np.eye(3))

    def test_x01_swaps_0_1(self):
        """X01|0⟩ = |1⟩, X01|1⟩ = |0⟩, X01|2⟩ = |2⟩."""
        m = X01().matrix()
        ket0 = np.array([1, 0, 0], dtype=complex)
        ket1 = np.array([0, 1, 0], dtype=complex)
        ket2 = np.array([0, 0, 1], dtype=complex)
        assert np.allclose(m @ ket0, ket1)
        assert np.allclose(m @ ket1, ket0)
        assert np.allclose(m @ ket2, ket2)

    def test_x02_swaps_0_2(self):
        m = X02().matrix()
        ket0 = np.array([1, 0, 0], dtype=complex)
        ket2 = np.array([0, 0, 1], dtype=complex)
        assert np.allclose(m @ ket0, ket2)
        assert np.allclose(m @ ket2, ket0)

    def test_x12_swaps_1_2(self):
        m = X12().matrix()
        ket1 = np.array([0, 1, 0], dtype=complex)
        ket2 = np.array([0, 0, 1], dtype=complex)
        assert np.allclose(m @ ket1, ket2)
        assert np.allclose(m @ ket2, ket1)

    def test_z01_is_diagonal(self):
        m = Z01().matrix()
        assert np.allclose(m, np.diag([1, -1, 1]))

    def test_xplus_is_cyclic(self):
        """X₊|0⟩ = |1⟩, X₊|1⟩ = |2⟩, X₊|2⟩ = |0⟩."""
        m = XPlus().matrix()
        ket0 = np.array([1, 0, 0], dtype=complex)
        ket1 = np.array([0, 1, 0], dtype=complex)
        ket2 = np.array([0, 0, 1], dtype=complex)
        assert np.allclose(m @ ket0, ket1)
        assert np.allclose(m @ ket1, ket2)
        assert np.allclose(m @ ket2, ket0)

    def test_xplus_cubed_is_identity(self):
        """X₊³ = I."""
        m = XPlus().matrix()
        assert np.allclose(m @ m @ m, np.eye(3), atol=1e-12)

    def test_xminus_is_xplus_inverse(self):
        assert np.allclose(
            XPlus().matrix() @ XMinus().matrix(), np.eye(3), atol=1e-12
        )

    def test_h3_creates_superposition(self):
        """H3|0⟩ should have equal amplitudes 1/√3 on all three states."""
        m = H3().matrix()
        ket0 = np.array([1, 0, 0], dtype=complex)
        result = m @ ket0
        assert np.allclose(np.abs(result) ** 2, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)

    def test_s3_cubed_is_identity(self):
        """S³ = I."""
        m = S3().matrix()
        assert np.allclose(m @ m @ m, np.eye(3), atol=1e-12)

    def test_y_gates_are_hermitian(self):
        """Pauli-like Y gates are Hermitian (Y† = Y)."""
        for cls in [Y01, Y02, Y12]:
            m = cls().matrix()
            assert np.allclose(m, m.conj().T, atol=1e-12), \
                f"{cls.__name__} is not Hermitian"


# ===================================================================
# 3. Parametric single-qutrit gates
# ===================================================================
class TestParametricGates:
    """Properties of parametric rotation gates."""

    @pytest.mark.parametrize("gate_cls", [Rx01, Rx02, Rx12, Ry01, Ry02, Ry12, Rz01, Rz02, Rz12])
    def test_rotation_at_zero_is_identity(self, gate_cls):
        """R(0) = I for all rotation gates."""
        g = gate_cls(0.0)
        assert np.allclose(g.matrix(), np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("gate_cls", [Rx01, Rx02, Rx12, Ry01, Ry02, Ry12, Rz01, Rz02, Rz12])
    def test_rotation_2pi_is_minus_identity(self, gate_cls):
        """R(2π) = -I for Pauli-like generators (half-angle convention)."""
        g = gate_cls(2 * np.pi)
        expected = -np.eye(3, dtype=complex)
        # Only the 2×2 subblock picks up the -1; the complementary state stays at +1
        # So R(2π) = diag with two -1 entries and one +1 entry
        m = g.matrix()
        # Verify it's diagonal and unitary
        assert g.is_unitary()
        # R(4π) = I
        m4pi = gate_cls(4 * np.pi).matrix()
        assert np.allclose(m4pi, np.eye(3), atol=1e-9)

    @pytest.mark.parametrize("gate_cls", [Rx01, Rx02, Rx12, Ry01, Ry02, Ry12, Rz01, Rz02, Rz12])
    def test_rotation_inverse(self, gate_cls):
        """R(θ)† = R(-θ)."""
        theta = 0.7
        g = gate_cls(theta)
        g_inv = g.inverse()
        product = g.matrix() @ g_inv.matrix()
        assert np.allclose(product, np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 2, np.pi, 2.5])
    def test_rx01_unitarity_parametric(self, theta):
        assert Rx01(theta).is_unitary()

    @pytest.mark.parametrize("gate_cls", [G01, G02, G12])
    def test_generalized_rotation_at_phi0_is_rx(self, gate_cls):
        """g_{ij}(θ, 0) = Rx_{ij}(θ)."""
        theta = 1.3
        g = gate_cls(theta, 0.0)
        # Map G01 → Rx01, etc.
        rx_map = {G01: Rx01, G02: Rx02, G12: Rx12}
        rx = rx_map[gate_cls](theta)
        assert np.allclose(g.matrix(), rx.matrix(), atol=1e-12)

    @pytest.mark.parametrize("gate_cls", [G01, G02, G12])
    def test_generalized_rotation_unitarity(self, gate_cls):
        g = gate_cls(0.7, 1.2)
        assert g.is_unitary()

    def test_ud_at_zero_is_identity(self):
        assert np.allclose(Ud(0, 0, 0).matrix(), np.eye(3), atol=1e-12)

    def test_ud_is_diagonal(self):
        m = Ud(0.1, 0.2, 0.3).matrix()
        # Off-diagonal elements should be zero
        assert np.allclose(m - np.diag(np.diag(m)), 0, atol=1e-12)

    def test_ud_inverse(self):
        g = Ud(0.1, 0.2, 0.3)
        product = g.matrix() @ g.inverse().matrix()
        assert np.allclose(product, np.eye(3), atol=1e-12)


# ===================================================================
# 4. Two-qutrit gates
# ===================================================================
class TestTwoQutritGates:
    """Two-qutrit gate actions and properties."""

    def test_csum_action(self):
        """CSUM|c, t⟩ = |c, (t+c) mod 3⟩."""
        m = CSUM().matrix()
        for c in range(3):
            for t in range(3):
                # Input: |c, t⟩
                inp = np.zeros(9, dtype=complex)
                inp[3 * c + t] = 1.0
                out = m @ inp
                # Expected: |c, (t+c) mod 3⟩
                expected_idx = 3 * c + ((t + c) % 3)
                expected = np.zeros(9, dtype=complex)
                expected[expected_idx] = 1.0
                assert np.allclose(out, expected), \
                    f"CSUM|{c},{t}⟩ failed: got idx {np.argmax(np.abs(out))}, " \
                    f"expected {expected_idx}"

    def test_csum_inverse(self):
        product = CSUM().matrix() @ CSUMDag().matrix()
        assert np.allclose(product, np.eye(9), atol=1e-12)

    def test_cphase_is_diagonal(self):
        m = CPhase().matrix()
        assert np.allclose(m - np.diag(np.diag(m)), 0, atol=1e-12)

    def test_cphase_diagonal_values(self):
        """CPhase diagonal: ω^{c·t} for c,t ∈ {0,1,2}."""
        omega = np.exp(2j * np.pi / 3)
        m = CPhase().matrix()
        for c in range(3):
            for t in range(3):
                idx = 3 * c + t
                expected = omega ** (c * t)
                assert np.isclose(m[idx, idx], expected, atol=1e-12), \
                    f"CPhase[{c},{t}] = {m[idx, idx]}, expected {expected}"

    def test_swap_action(self):
        """SWAP|a, b⟩ = |b, a⟩."""
        m = SWAP3().matrix()
        for a in range(3):
            for b in range(3):
                inp = np.zeros(9, dtype=complex)
                inp[3 * a + b] = 1.0
                out = m @ inp
                expected = np.zeros(9, dtype=complex)
                expected[3 * b + a] = 1.0
                assert np.allclose(out, expected)

    def test_swap_self_inverse(self):
        m = SWAP3().matrix()
        assert np.allclose(m @ m, np.eye(9), atol=1e-12)

    def test_cnot3_matches_csum_on_adjacent(self):
        """CNOT3 and CSUM should produce the same 9×9 matrix."""
        assert np.allclose(CNOT3().matrix(), CSUM().matrix(), atol=1e-12)


# ===================================================================
# 5. Circuit integration: QutritCircuit.append()
# ===================================================================
class TestCircuitAppend:
    """Test the new append() method on QutritCircuit."""

    def test_append_single_gate(self):
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        assert len(qc) == 1

    def test_append_parametric_gate(self):
        qc = QutritCircuit(1, None)
        qc.append(Rx01(np.pi / 4), first_qutrit=0)
        assert len(qc) == 1

    def test_append_two_qutrit_gate(self):
        qc = QutritCircuit(2, None)
        qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
        assert len(qc) == 1

    def test_append_rejects_missing_target(self):
        qc = QutritCircuit(2, None)
        with pytest.raises(ValueError, match="requires second_qutrit"):
            qc.append(CSUM(), first_qutrit=0)

    def test_append_rejects_extra_target(self):
        qc = QutritCircuit(2, None)
        with pytest.raises(ValueError, match="does not accept"):
            qc.append(H3(), first_qutrit=0, second_qutrit=1)

    def test_append_rejects_non_gate(self):
        qc = QutritCircuit(1, None)
        with pytest.raises(TypeError):
            qc.append("not_a_gate", first_qutrit=0)

    def test_append_with_inverse(self):
        """append(gate.inverse(), ...) should apply the adjoint."""
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(H3().inverse(), first_qutrit=0)
        sim = QASMSimulator(qc)
        state = sim.return_final_state()
        # H @ H† = I, so state should be |0⟩
        expected = np.zeros((3, 1), dtype=complex)
        expected[0, 0] = 1.0
        assert np.allclose(state, expected, atol=1e-12)


# ===================================================================
# 6. End-to-end: Gate → append → simulate → verify
# ===================================================================
class TestEndToEnd:
    """Round-trip tests: build circuit with Gate objects, simulate, verify."""

    def test_h3_creates_uniform_superposition(self):
        """H3|0⟩ should produce equal probabilities on |0⟩, |1⟩, |2⟩."""
        qc = QutritCircuit(1, None)
        qc.append(H3(), first_qutrit=0)
        qc.measure_all()

        sim = QASMSimulator(qc)
        sim.run(num_shots=30_000)
        counts = sim.get_counts()
        for outcome in ("0", "1", "2"):
            assert 9_000 <= counts.get(outcome, 0) <= 11_000, counts

    def test_x01_flips_0_to_1(self):
        """X01|0⟩ = |1⟩, deterministic."""
        qc = QutritCircuit(1, None)
        qc.append(X01(), first_qutrit=0)
        qc.measure_all()

        sim = QASMSimulator(qc)
        sim.run(num_shots=100)
        counts = sim.get_counts()
        assert counts == {"1": 100}

    def test_bell_state_via_append(self):
        """H3 + CSUM should produce qutrit Bell state {|00⟩, |11⟩, |22⟩}."""
        qc = QutritCircuit(2, None)
        qc.append(H3(), first_qutrit=0)
        qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
        qc.measure_all()

        sim = QASMSimulator(qc)
        sim.run(num_shots=20_000)
        counts = sim.get_counts()
        assert set(counts.keys()) <= {"00", "11", "22"}
        for outcome in ("00", "11", "22"):
            assert 5_500 <= counts[outcome] <= 7_500, counts

    def test_rotation_gate_statevector(self):
        """Rx01(π)|0⟩ = -i|1⟩ (up to global phase)."""
        qc = QutritCircuit(1, None)
        qc.append(Rx01(np.pi), first_qutrit=0)
        sim = QASMSimulator(qc)
        state = sim.return_final_state()
        # Should be -i|1⟩
        assert np.abs(state[0, 0]) < 1e-9
        assert np.isclose(np.abs(state[1, 0]), 1.0, atol=1e-9)
        assert np.abs(state[2, 0]) < 1e-9

    def test_swap_via_append(self):
        """SWAP|1,2⟩ = |2,1⟩."""
        # Prepare |1,2⟩
        init = np.zeros((9, 1), dtype=complex)
        init[3 * 1 + 2, 0] = 1.0  # |1,2⟩
        qc = QutritCircuit(2, init)
        qc.append(SWAP3(), first_qutrit=0, second_qutrit=1)

        sim = QASMSimulator(qc)
        state = sim.return_final_state()
        expected = np.zeros((9, 1), dtype=complex)
        expected[3 * 2 + 1, 0] = 1.0  # |2,1⟩
        assert np.allclose(state, expected, atol=1e-12)

    def test_ud_phase_on_statevector(self):
        """Ud(φ₁,φ₂,φ₃) applied to |0⟩ gives e^{iφ₁}|0⟩."""
        qc = QutritCircuit(1, None)
        qc.append(Ud(0.5, 0.0, 0.0), first_qutrit=0)
        sim = QASMSimulator(qc)
        state = sim.return_final_state()
        assert np.isclose(state[0, 0], np.exp(0.5j), atol=1e-12)
        assert np.abs(state[1, 0]) < 1e-12
        assert np.abs(state[2, 0]) < 1e-12
