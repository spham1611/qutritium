"""Tests for the qutritium.metrics subpackage.

Organized by metric:
  1. state_fidelity   (pure-pure, pure-mixed, mixed-mixed, symmetry, bounds)
  2. trace_distance   (identical, orthogonal, symmetry, bounds)
  3. purity           (pure, maximally mixed, bounds)
  4. von_neumann_entropy (pure, maximally mixed, base conversion)
  5. process_fidelity (identity check, X01 vs I, bounds)
  6. average_gate_fidelity (consistency with process_fidelity)
  7. Input validation
"""
from __future__ import annotations

import numpy as np
import pytest

from qutritium.metrics import (
    average_gate_fidelity,
    process_fidelity,
    purity,
    state_fidelity,
    trace_distance,
    von_neumann_entropy,
)

# ===================================================================
# Fixtures: useful test states
# ===================================================================
_KET0 = np.array([1, 0, 0], dtype=complex)
_KET1 = np.array([0, 1, 0], dtype=complex)
_KET2 = np.array([0, 0, 1], dtype=complex)
_MAX_MIXED = np.eye(3, dtype=complex) / 3
_X01 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
_H3 = (1 / np.sqrt(3)) * np.array(
    [[1, 1, 1],
     [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)],
     [1, np.exp(4j * np.pi / 3), np.exp(2j * np.pi / 3)]],
    dtype=complex,
)


def _random_pure_state(d: int = 3, seed: int = 0) -> np.ndarray:
    """Haar-random pure state of dimension d."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _random_density_matrix(d: int = 3, seed: int = 0) -> np.ndarray:
    """Random full-rank density matrix from A A^dag / tr(A A^dag)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


# ===================================================================
# 1. state_fidelity
# ===================================================================
class TestStateFidelity:
    def test_identical_pure_states(self):
        assert state_fidelity(_KET0, _KET0) == pytest.approx(1.0)

    def test_orthogonal_pure_states(self):
        assert state_fidelity(_KET0, _KET1) == pytest.approx(0.0, abs=1e-12)

    def test_pure_state_with_self_as_column_vector(self):
        col = _KET0.reshape(-1, 1)
        assert state_fidelity(col, col) == pytest.approx(1.0)

    def test_pure_vs_density_matrix_form(self):
        """Fidelity should agree whether |psi> is passed as ket or |psi><psi|."""
        rho = np.outer(_KET0, _KET0.conj())
        assert state_fidelity(_KET0, _KET1) == pytest.approx(
            state_fidelity(rho, _KET1), abs=1e-12,
        )

    def test_pure_vs_maximally_mixed(self):
        # F(|0>, I/3) = <0|I/3|0> = 1/3
        assert state_fidelity(_KET0, _MAX_MIXED) == pytest.approx(1.0 / 3.0)

    def test_symmetry(self):
        rho = _random_density_matrix(seed=1)
        sigma = _random_density_matrix(seed=2)
        f_ab = state_fidelity(rho, sigma)
        f_ba = state_fidelity(sigma, rho)
        assert f_ab == pytest.approx(f_ba, abs=1e-10)

    def test_random_self_fidelity_is_one(self):
        for seed in range(5):
            rho = _random_density_matrix(seed=seed)
            assert state_fidelity(rho, rho) == pytest.approx(1.0, abs=1e-9)

    def test_pure_pure_matches_inner_product_squared(self):
        psi = _random_pure_state(seed=3)
        phi = _random_pure_state(seed=4)
        expected = abs(np.vdot(psi, phi)) ** 2
        # Two eigendecompositions through _matrix_sqrt_hermitian on a
        # rank-1 input lose ~7-8 digits relative to the direct
        # |<psi|phi>|^2 formula. Default pytest.approx tolerance (rel=1e-6)
        # is well within the achievable precision.
        assert state_fidelity(psi, phi) == pytest.approx(expected)

    def test_bounded_in_unit_interval(self):
        for seed in range(5):
            rho = _random_density_matrix(seed=seed)
            sigma = _random_density_matrix(seed=seed + 10)
            f = state_fidelity(rho, sigma)
            assert 0.0 <= f <= 1.0 + 1e-12

    def test_mixed_vs_mixed_known_value(self):
        """For two commuting diagonal mixed states, F reduces to the
        classical Bhattacharyya coefficient:
            F(diag(p), diag(q)) = (sum_i sqrt(p_i * q_i))^2.
        """
        rho = np.diag([0.5, 0.3, 0.2]).astype(complex)
        sigma = np.diag([0.4, 0.4, 0.2]).astype(complex)
        expected = (
                           np.sqrt(0.5 * 0.4) + np.sqrt(0.3 * 0.4) + np.sqrt(0.2 * 0.2)
                   ) ** 2
        assert state_fidelity(rho, sigma) == pytest.approx(expected)


# ===================================================================
# 2. trace_distance
# ===================================================================
class TestTraceDistance:
    def test_identical_is_zero(self):
        assert trace_distance(_KET0, _KET0) == pytest.approx(0.0, abs=1e-12)

    def test_orthogonal_pure_states_is_one(self):
        assert trace_distance(_KET0, _KET1) == pytest.approx(1.0)

    def test_symmetry(self):
        rho = _random_density_matrix(seed=5)
        sigma = _random_density_matrix(seed=6)
        assert trace_distance(rho, sigma) == pytest.approx(
            trace_distance(sigma, rho), abs=1e-12,
        )

    def test_non_negative(self):
        for seed in range(5):
            rho = _random_density_matrix(seed=seed)
            sigma = _random_density_matrix(seed=seed + 20)
            assert trace_distance(rho, sigma) >= -1e-12

    def test_pure_vs_maximally_mixed(self):
        # T(|0>, I/3) = (1/2) sum |eig(|0><0| - I/3)| = (1/2)(2/3 + 1/3 + 1/3) = 2/3
        assert trace_distance(_KET0, _MAX_MIXED) == pytest.approx(2.0 / 3.0)


# ===================================================================
# 3. purity
# ===================================================================
class TestPurity:
    @pytest.mark.parametrize("ket", [_KET0, _KET1, _KET2])
    def test_pure_state_purity_is_one(self, ket):
        assert purity(ket) == pytest.approx(1.0, abs=1e-12)

    def test_maximally_mixed_purity_is_one_over_d(self):
        assert purity(_MAX_MIXED) == pytest.approx(1.0 / 3.0)

    def test_purity_in_bounds(self):
        for seed in range(5):
            rho = _random_density_matrix(seed=seed)
            p = purity(rho)
            assert 1.0 / 3.0 - 1e-12 <= p <= 1.0 + 1e-12


# ===================================================================
# 4. von_neumann_entropy
# ===================================================================
class TestVonNeumannEntropy:
    @pytest.mark.parametrize("ket", [_KET0, _KET1, _KET2])
    def test_pure_state_has_zero_entropy(self, ket):
        assert von_neumann_entropy(ket) == pytest.approx(0.0, abs=1e-12)

    def test_maximally_mixed_log2(self):
        # S(I/3) = log_2(3)
        assert von_neumann_entropy(_MAX_MIXED) == pytest.approx(np.log2(3.0))

    def test_maximally_mixed_natural_log(self):
        assert von_neumann_entropy(_MAX_MIXED, base=np.e) == pytest.approx(
            np.log(3.0),
        )

    def test_base_conversion_is_linear(self):
        rho = _random_density_matrix(seed=7)
        s_2 = von_neumann_entropy(rho, base=2.0)
        s_e = von_neumann_entropy(rho, base=np.e)
        # log_2 = ln / ln(2)
        assert s_2 == pytest.approx(s_e / np.log(2.0), abs=1e-10)

    def test_non_negative(self):
        for seed in range(5):
            rho = _random_density_matrix(seed=seed)
            assert von_neumann_entropy(rho) >= -1e-12

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match="base"):
            von_neumann_entropy(_KET0, base=1.0)
        with pytest.raises(ValueError, match="base"):
            von_neumann_entropy(_KET0, base=-2.0)


# ===================================================================
# 5. process_fidelity
# ===================================================================
class TestProcessFidelity:
    def test_self_fidelity_is_one_identity(self):
        eye3 = np.eye(3, dtype=complex)
        assert process_fidelity(eye3, eye3) == pytest.approx(1.0)

    def test_self_fidelity_is_one_x01(self):
        assert process_fidelity(_X01, _X01) == pytest.approx(1.0)

    def test_self_fidelity_is_one_h3(self):
        assert process_fidelity(_H3, _H3) == pytest.approx(1.0)

    def test_identity_vs_x01(self):
        # tr(I^dag X01) = tr(X01) = 0 + 0 + 1 = 1
        # F = |1|^2 / 9 = 1/9
        eye3 = np.eye(3, dtype=complex)
        assert process_fidelity(eye3, _X01) == pytest.approx(1.0 / 9.0)

    def test_global_phase_invariance(self):
        # U and e^{i theta} U should have F = 1
        eye3 = np.eye(3, dtype=complex)
        phased = np.exp(1j * 0.7) * eye3
        assert process_fidelity(eye3, phased) == pytest.approx(1.0)

    def test_symmetry(self):
        f_ab = process_fidelity(_X01, _H3)
        f_ba = process_fidelity(_H3, _X01)
        assert f_ab == pytest.approx(f_ba)

    def test_bounded_in_unit_interval(self):
        f = process_fidelity(_X01, _H3)
        assert 0.0 <= f <= 1.0 + 1e-12


# ===================================================================
# 6. average_gate_fidelity
# ===================================================================
class TestAverageGateFidelity:
    def test_self_is_one(self):
        assert average_gate_fidelity(_X01, _X01) == pytest.approx(1.0)

    def test_consistency_with_process_fidelity(self):
        """F_avg = (d * F_pro + 1) / (d + 1)."""
        d = 3
        f_pro = process_fidelity(_X01, _H3)
        expected = (d * f_pro + 1.0) / (d + 1.0)
        assert average_gate_fidelity(_X01, _H3) == pytest.approx(expected)

    def test_identity_vs_x01(self):
        # F_pro = 1/9, d = 3 -> F_avg = (3/9 + 1) / 4 = (4/3)/4 = 1/3
        eye3 = np.eye(3, dtype=complex)
        assert average_gate_fidelity(eye3, _X01) == pytest.approx(1.0 / 3.0)

    def test_lower_bound_is_one_over_d_plus_one(self):
        """F_avg >= 1/(d+1) when F_pro = 0 (impossible for pure unitaries
        but the formula should respect the floor for any non-negative F_pro)."""
        # Use a Haar-random unitary pair and check the bound
        rng = np.random.default_rng(8)
        u1 = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
        u2 = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
        f_avg = average_gate_fidelity(u1, u2)
        assert f_avg >= 1.0 / 4.0 - 1e-12


# ===================================================================
# 7. Input validation
# ===================================================================
class TestInputValidation:
    def test_state_fidelity_mismatched_dims(self):
        rho_2 = np.eye(2, dtype=complex) / 2
        rho_3 = np.eye(3, dtype=complex) / 3
        with pytest.raises(ValueError, match="mismatched"):
            state_fidelity(rho_2, rho_3)

    def test_state_fidelity_rejects_3d(self):
        bad = np.zeros((3, 3, 3), dtype=complex)
        with pytest.raises(ValueError):
            state_fidelity(bad, _KET0)

    def test_process_fidelity_rejects_non_square(self):
        non_square = np.zeros((3, 4), dtype=complex)
        eye3 = np.eye(3, dtype=complex)
        with pytest.raises(ValueError, match="square"):
            process_fidelity(non_square, eye3)

    def test_process_fidelity_mismatched_dims(self):
        eye2 = np.eye(2, dtype=complex)
        eye3 = np.eye(3, dtype=complex)
        with pytest.raises(ValueError, match="mismatched"):
            process_fidelity(eye2, eye3)

    def test_process_fidelity_rejects_non_unitary_ideal(self):
        non_unitary = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=complex,
        )
        eye3 = np.eye(3, dtype=complex)
        with pytest.raises(ValueError, match="not unitary"):
            process_fidelity(non_unitary, eye3)

    def test_process_fidelity_rejects_non_unitary_actual(self):
        non_unitary = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=complex,
        )
        eye3 = np.eye(3, dtype=complex)
        with pytest.raises(ValueError, match="not unitary"):
            process_fidelity(eye3, non_unitary)
