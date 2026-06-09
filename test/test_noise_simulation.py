"""Tests for noise integration on the simulators.

Organized as:
  1. Gate (Kraus) errors on the density-matrix simulator
  2. Prep errors
  3. Readout errors (both backends, deterministic)
  4. set_noise_model guards
"""
from __future__ import annotations

import numpy as np
import pytest

from qutritium import DensityMatrixSimulator, QASMSimulator, QutritCircuit
from qutritium.channels import (depolarizing_channel, NoiseModel, ReadoutError, SPAMNoiseModel)
from qutritium.gates import H3, X01


def _x01_measured() -> QutritCircuit:
    qc = QutritCircuit(1, None)
    qc.append(X01(), first_qutrit=0)
    qc.measure_all()
    return qc


def _h3_measured() -> QutritCircuit:
    qc = QutritCircuit(1, None)
    qc.append(H3(), first_qutrit=0)
    qc.measure_all()
    return qc


# ===================================================================
# 1. Gate (Kraus) errors on the density-matrix simulator
# ===================================================================
class TestGateErrors:
    def test_depolarizing_after_gate_diagonal(self):
        # X01|0> = |1>, then depolarizing(0.3): diag = [0.1, 0.8, 0.1]
        nm = NoiseModel()
        nm.add_quantum_error(depolarizing_channel(0.3), "X01")
        dm = DensityMatrixSimulator(_x01_measured())
        dm.set_noise_model(nm)
        assert np.allclose(dm.probabilities(), [0.1, 0.8, 0.1], atol=1e-9)

    def test_error_only_fires_on_matching_label(self):
        # error registered for X01 but circuit uses H3 -> no effect
        nm = NoiseModel()
        nm.add_quantum_error(depolarizing_channel(0.5), "X01")
        dm = DensityMatrixSimulator(_h3_measured())
        dm.set_noise_model(nm)
        assert np.allclose(dm.probabilities(), [1 / 3, 1 / 3, 1 / 3], atol=1e-9)

    def test_empty_noise_model_matches_noiseless(self):
        baseline = DensityMatrixSimulator(_h3_measured()).probabilities()
        dm = DensityMatrixSimulator(_h3_measured())
        dm.set_noise_model(NoiseModel())
        assert np.allclose(dm.probabilities(), baseline, atol=1e-12)


# ===================================================================
# 2. Prep errors
# ===================================================================
class TestPrepErrors:
    def test_full_depolarizing_prep_gives_uniform(self):
        # fully depolarize before gates -> uniform regardless of the gate
        nm = NoiseModel()
        nm.add_prep_error(depolarizing_channel(1.0))
        dm = DensityMatrixSimulator(_h3_measured())
        dm.set_noise_model(nm)
        assert np.allclose(dm.probabilities(), [1 / 3, 1 / 3, 1 / 3], atol=1e-9)


# ===================================================================
# 3. Readout errors (both backends, deterministic)
# ===================================================================
def _collapse_to_zero() -> ReadoutError:
    # every true outcome reported as 0
    return ReadoutError(np.array([[1.0, 1.0, 1.0], [0, 0, 0], [0, 0, 0]]))


class TestReadoutErrors:
    def test_readout_on_statevector(self):
        nm = NoiseModel()
        nm.add_readout_error(_collapse_to_zero())
        sim = QASMSimulator(_h3_measured())
        sim.set_noise_model(nm)
        sim.run(num_shots=500)
        assert sim.get_counts() == {"0": 500}

    def test_readout_on_density_matrix(self):
        nm = NoiseModel()
        nm.add_readout_error(_collapse_to_zero())
        sim = DensityMatrixSimulator(_h3_measured())
        sim.set_noise_model(nm)
        sim.run(num_shots=500)
        assert sim.get_counts() == {"0": 500}


# ===================================================================
# 4. set_noise_model guards
# ===================================================================
class TestNoiseModelGuards:
    def test_statevector_rejects_kraus(self):
        nm = NoiseModel()
        nm.add_quantum_error(depolarizing_channel(0.1), "X01")
        with pytest.raises(NotImplementedError):
            QASMSimulator(QutritCircuit(1, None)).set_noise_model(nm)

    def test_statevector_accepts_readout_only(self):
        nm = NoiseModel()
        nm.add_readout_error(ReadoutError(np.eye(3)))
        QASMSimulator(QutritCircuit(1, None)).set_noise_model(nm)  # must not raise

    def test_spam_rejected_by_statevector(self):
        with pytest.raises(NotImplementedError):
            QASMSimulator(QutritCircuit(1, None)).set_noise_model(SPAMNoiseModel(0.05, 0.05, 1))

    def test_cannot_set_after_run(self):
        sim = DensityMatrixSimulator(_h3_measured())
        sim.run(num_shots=10)
        with pytest.raises(RuntimeError):
            sim.set_noise_model(NoiseModel())

    def test_can_replace_before_run(self):
        sim = DensityMatrixSimulator(QutritCircuit(1, None))
        sim.set_noise_model(NoiseModel())
        sim.set_noise_model(NoiseModel())  # allowed before any run
