# MIT License — Copyright (c) 2023-2026 Son Pham
# See LICENSE.txt for full terms.

"""Deprecation shim: ``QASMSimulator`` is now an alias for ``StatevectorSimulator``."""

from __future__ import annotations

import numpy as np
import pytest

from qutritium import QASMSimulator, QutritCircuit, StatevectorSimulator
from qutritium.gates import X01


def _circuit() -> QutritCircuit:
    qc = QutritCircuit(1, None)
    qc.append(X01(), first_qutrit=0)
    return qc


def test_qasmsimulator_warns_on_instantiation():
    with pytest.warns(DeprecationWarning, match="StatevectorSimulator"):
        sim = QASMSimulator(_circuit())
    assert isinstance(sim, StatevectorSimulator)


def test_qasmsimulator_matches_statevector():
    with pytest.warns(DeprecationWarning):
        old = QASMSimulator(_circuit()).return_final_state()
    new = StatevectorSimulator(_circuit()).return_final_state()
    assert np.allclose(old, new)


def test_alias_is_shared_across_import_paths():
    from qutritium.simulator import QASMSimulator as from_subpackage

    assert from_subpackage is QASMSimulator
