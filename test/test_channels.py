"""Tests for qutritium.channels.

Organized as:
  1. Channel validation + Kraus action
  2. Preset channels (depolarizing, dephasing, amplitude damping, pauli)
  3. ReadoutError confusion matrix
  4. NoiseModel container + SPAMNoiseModel
"""
from __future__ import annotations

import numpy as np
import pytest

from qutritium.channels import (amplitude_damping_channel, Channel, dephasing_channel, depolarizing_channel, NoiseModel,
                                pauli_channel, ReadoutError, SPAMNoiseModel)

I3 = np.eye(3, dtype=complex)


def _ket_rho(i: int) -> np.ndarray:
    rho = np.zeros((3, 3), dtype=complex)
    rho[i, i] = 1.0
    return rho


# ===================================================================
# 1. Channel
# ===================================================================
class TestChannel:
    def test_identity_channel_valid(self):
        assert len(Channel([I3]).kraus) == 1

    def test_completeness_violation_raises(self):
        with pytest.raises(ValueError, match="completeness"):
            Channel([I3, I3])  # sum K^dag K = 2I

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Channel([np.eye(2, dtype=complex)])

    def test_num_qutrits_below_one_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            Channel([np.eye(1, dtype=complex)], num_qutrits=0)

    def test_kraus_returns_copy(self):
        ch = Channel([I3])
        ch.kraus.append(I3)  # mutate the returned list
        assert len(ch.kraus) == 1  # channel is unaffected

    def test_apply_identity_is_noop(self):
        rho = _ket_rho(1)
        assert np.allclose(Channel([I3]).apply_kraus_op(rho), rho)


# ===================================================================
# 2. Preset channels
# ===================================================================
class TestDepolarizing:
    def test_p1_gives_maximally_mixed(self):
        assert np.allclose(depolarizing_channel(1.0).apply_kraus_op(_ket_rho(0)), I3 / 3)

    def test_p0_is_identity(self):
        rho = _ket_rho(2)
        assert np.allclose(depolarizing_channel(0.0).apply_kraus_op(rho), rho)

    @pytest.mark.parametrize("p", [0.3, 0.7])
    def test_convex_mix_formula(self, p):
        # E(rho) = (1-p) rho + p I/3
        rho = _ket_rho(1)
        expected = (1 - p) * rho + p * I3 / 3
        assert np.allclose(depolarizing_channel(p).apply_kraus_op(rho), expected)

    @pytest.mark.parametrize("p", [-0.1, 1.5])
    def test_out_of_range_raises(self, p):
        with pytest.raises(ValueError):
            depolarizing_channel(p)


class TestDephasing:
    _PLUS = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]], dtype=complex)

    def test_p1_removes_coherences(self):
        out = dephasing_channel(1.0).apply_kraus_op(self._PLUS)
        assert np.allclose(out, np.diag([0.5, 0.5, 0.0]))

    def test_p0_is_identity(self):
        assert np.allclose(dephasing_channel(0.0).apply_kraus_op(self._PLUS), self._PLUS)


class TestAmplitudeDamping:
    def test_single_step_2_to_1(self):
        assert np.allclose(amplitude_damping_channel(1.0).apply_kraus_op(_ket_rho(2)), _ket_rho(1))

    def test_single_step_1_to_0(self):
        assert np.allclose(amplitude_damping_channel(1.0).apply_kraus_op(_ket_rho(1)), _ket_rho(0))

    @pytest.mark.parametrize("gamma", [0.0, 0.4, 1.0])
    def test_ground_state_is_fixed(self, gamma):
        assert np.allclose(amplitude_damping_channel(gamma).apply_kraus_op(_ket_rho(0)), _ket_rho(0))

    def test_steady_state_relaxes_to_ground(self):
        ch = amplitude_damping_channel(1.0)
        rho = _ket_rho(2)
        for _ in range(6):
            rho = ch.apply_kraus_op(rho)
        assert np.allclose(rho, _ket_rho(0))

    def test_gamma21_defaults_to_gamma10(self):
        a = amplitude_damping_channel(0.5)
        b = amplitude_damping_channel(0.5, 0.5)
        assert np.allclose(a.apply_kraus_op(_ket_rho(1)), b.apply_kraus_op(_ket_rho(1)))

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            amplitude_damping_channel(1.5)


class TestPauliChannel:
    def test_identity_only_is_noop(self):
        rho = _ket_rho(1)
        assert np.allclose(pauli_channel({"identity": 1.0}).apply_kraus_op(rho), rho)

    def test_valid_mixture(self):
        assert len(pauli_channel({"identity": 0.5, "x_plus": 0.5}).kraus) == 2

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            pauli_channel({"identity": 1.2, "x_plus": -0.2})

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError):
            pauli_channel({"identity": 0.5})

    def test_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            pauli_channel({"foo": 1.0})


# ===================================================================
# 3. ReadoutError
# ===================================================================
class TestReadoutError:
    def test_identity_apply(self):
        ro = ReadoutError(np.eye(3))
        assert np.allclose(ro.apply(np.array([1.0, 0.0, 0.0])), [1, 0, 0])

    def test_apply_mixes_probabilities(self):
        a = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        assert np.allclose(ReadoutError(a).apply(np.array([1.0, 0.0, 0.0])), [0.9, 0.05, 0.05])

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            ReadoutError(np.ones((2, 3)))

    def test_non_power_of_three_raises(self):
        with pytest.raises(ValueError, match="power of 3"):
            ReadoutError(np.eye(2))

    def test_negative_entry_raises(self):
        bad = np.array([[1.5, 0, 0], [-0.5, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="negative"):
            ReadoutError(bad)

    def test_columns_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            ReadoutError(np.diag([1.0, 1.0, 0.5]))

    def test_from_single_qutrit_one_keeps_shape(self):
        a = np.diag([1.0, 1.0, 1.0])
        assert ReadoutError.from_single_qutrit(a, 1).confusion_matrix.shape == (3, 3)

    def test_from_single_qutrit_two_is_nine(self):
        a = np.diag([1.0, 1.0, 1.0])
        assert ReadoutError.from_single_qutrit(a, 2).confusion_matrix.shape == (9, 9)


# ===================================================================
# 4. NoiseModel + SPAMNoiseModel
# ===================================================================
class TestNoiseModel:
    def test_gate_error_lookup(self):
        nm = NoiseModel()
        ch = depolarizing_channel(0.1)
        nm.add_quantum_error(ch, "X01")
        assert nm.error_for("X01") is ch
        assert nm.error_for("H3") is None

    def test_prep_error_registered(self):
        nm = NoiseModel()
        nm.add_prep_error(depolarizing_channel(0.1))
        assert len(nm.prep_errors) == 1

    def test_readout_registered(self):
        nm = NoiseModel()
        nm.add_readout_error(ReadoutError(np.eye(3)))
        assert nm.readout is not None

    def test_has_kraus_errors(self):
        nm = NoiseModel()
        assert nm.has_kraus_errors is False
        nm.add_readout_error(ReadoutError(np.eye(3)))
        assert nm.has_kraus_errors is False  # readout is not Kraus
        nm.add_quantum_error(depolarizing_channel(0.1), "X01")
        assert nm.has_kraus_errors is True


class TestSPAMNoiseModel:
    def test_has_prep_and_readout(self):
        spam = SPAMNoiseModel(0.05, 0.05, 1)
        assert len(spam.prep_errors) == 1
        assert spam.readout is not None
        assert spam.has_kraus_errors is True

    def test_zero_error_readout_is_identity(self):
        spam = SPAMNoiseModel(0.0, 0.0, 1)
        assert np.allclose(spam.readout.confusion_matrix, np.eye(3))

    @pytest.mark.parametrize("p_prep,p_meas", [(-0.1, 0.0), (0.0, 1.5)])
    def test_out_of_range_raises(self, p_prep, p_meas):
        with pytest.raises(ValueError):
            SPAMNoiseModel(p_prep, p_meas, 1)
