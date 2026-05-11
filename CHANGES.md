# Changelog

All notable changes to Qutritium are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project follows
[Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-05-07

### Summary

**Major release.** Pivots Qutritium from a Qiskit-pulse / IBM-hardware-coupled
package into a hardware-agnostic Python library for qutrit quantum computing.

The retained core (`src/qutritium/`) now depends only on `numpy` and `scipy`
(plus optional `matplotlib` for plotting). All Qiskit-pulse, `qiskit-ibm-provider`,
and IBM-backend-specific code has been moved to the top-level `legacy/`
directory as a historical record of the work presented at the Munich Quantum
Software Conference 2023; `legacy/` is **not** part of the installable wheel.

A separate `changes_summary.txt` companion file lists just the file paths
grouped by action for quick reference.

### Migration notes

This release is **breaking**. The two changes most likely to affect existing
users:

1. **Import paths.** All imports must change from `src.X` to `qutritium.X`.
   ```python
   # v0.0.x
   from src.quantumcircuit.QC import Qutrit_circuit
   from src.vm_backend.QASM_backend import QASM_Simulator
   # v1.0.0
   from qutritium import Qutrit_circuit, QASM_Simulator
   ```
   The top-level `qutritium` package re-exports the four most-used symbols
   (`Qutrit_circuit`, `Instruction`, `QASM_Simulator`, `SU3_matrices`) so
   that fully-qualified module paths are no longer necessary for typical
   usage.

2. **`Pulse_Wrapper` removed from the package.** The `Pulse_Wrapper` class in
   `src.decomposition.transpilation` (which produced Qiskit-pulse
   `ScheduleBlock` objects for IBM hardware) is **gone** from the importable
   surface. Its source is preserved in
   `legacy/decomposition/transpilation.py.legacy.txt` for reference.

### `rz01` / `rz12` convention change (Option 2A)

The v0.0.1 codebase had two **inconsistent** definitions of `rz01` /
`rz12` in two different modules — `qc_elementary_matrices.py` used the
asymmetric `diag(exp(iφ), 1, 1)` while `qc_utility.single_matrix_form` used
the symmetric `diag(exp(-iφ/2), exp(iφ/2), 1)`. This release standardises on
the **symmetric** form everywhere (textbook convention; matches Bertlmann &
Krammer arXiv:0806.1174, Nielsen & Chuang, and the existing
`single_matrix_form` definition).

The composite `r01` and `r12` rotations were updated in lockstep so that
their *outputs* are **bit-identical** to v0.0.1 (verified to machine
precision over 1000 random `(phi, theta)` samples). This means the SU(3)
decomposition's angle-extraction logic in `Parameter.get_parameters` did
**not** need any changes — its inputs are unchanged. Reconstruction
fidelity test results: `1.0000000000` for the discrete Fourier matrix and
`0.9999999999` worst-case over 20 random SU(3) samples.

Backward compatibility is not available, please use this new version of API

---

### Added

#### Top-level files

| Path                          | Purpose                                                            | Lines |
|-------------------------------|--------------------------------------------------------------------|-------|
| `pyproject.toml` (rewritten)  | PEP 621 metadata, deps, ruff/black/mypy/pytest config, single source of truth replacing `setup.py` + `setup.cfg`. | ~140  |
| `CITATION.cff`                | Citation File Format 1.2.0 — citable metadata + Munich poster + Unitary Fund acknowledgment + reference to v0.0.1 as historical software artefact. | ~60   |
| `README.md` (rewritten)       | Reflects hardware-agnostic v1.0.0 purpose; quick-start; module overview; v0.0.x → v1.0.0 migration guide. | ~110  |
| `CHANGES.md`                  | This file. Replaces the 1-line `CHANGES.txt` stub.                | (this)|
| `changes_summary.txt`         | Plain-text companion to this changelog — file-path-only summary grouped by action (added / modified / moved / deleted). | ~80   |
| `legacy/README.md`            | Explains what's in `legacy/`, why it's preserved, and how to revive it if needed. | ~70   |

#### Package surface

| Path                                              | Was                                  | Now                                                                                                                                                                                                                                                                |
|---------------------------------------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/qutritium/__init__.py`                       | empty file                           | full module docstring + curated `__all__` re-exporting `Qutrit_circuit`, `Instruction`, `QASM_Simulator`, `SU3_matrices`; `__version__ = "1.0.0"`.                                                                                                                  |
| `src/qutritium/quantumcircuit/__init__.py`        | three commented-out wildcard imports | explicit `__all__` re-exporting `GATE_SET`, `Instruction`, `Qutrit_circuit`, plus the four utility functions.                                                                                                                                                      |
| `src/qutritium/vm_backend/__init__.py`            | `from .QASM_backend import *` (wildcard) | explicit `__all__ = ["QASM_Simulator"]`.                                                                                                                                                                                                                          |
| `src/qutritium/decomposition/__init__.py`         | empty file                           | explicit `__all__` re-exporting `Parameter` and `SU3_matrices`.                                                                                                                                                                                                    |
| `src/qutritium/py.typed`                          | (didn't exist)                       | PEP 561 marker indicating the package ships type annotations for downstream type-checkers.                                                                                                                                                                         |

#### Test surface

| Path                                  | Status                                                                                                                                                                                       |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `test/__init__.py`                    | added (pytest test-package marker).                                                                                                                                                          |
| `test/VM_test/__init__.py`            | added.                                                                                                                                                                                       |
| `test/VM_test/VM_test.py`             | rewritten as a proper pytest suite with 12 tests: circuit-addition correctness, double-measurement guard, U_ft + 4 random SU(3) roundtrip checks, H+CNOT distribution, simulator input validation, density-matrix Hermiticity/PSD. **All 12 tests pass** (`python -m pytest test/ -v` ⇒ `12 passed in 0.13s`). |

---

### Modified

#### `src/qutritium/quantumcircuit/qc_elementary_matrices.py`

**Lines:** ~180 → ~340 (most growth is docstrings + named constants; logic
is leaner).

- Added module docstring with explicit references (Bertlmann & Krammer,
  Wikipedia Gell-Mann / generalized Pauli articles).
- Replaced the bare-list `state_0 = [[1],[0],[0]]` constants with private
  `np.ndarray` column vectors (`_STATE_0`, `_STATE_1`, `_STATE_2`).
- Added module-level constants `_PI` and `OMEGA_DEFAULT` to remove magic
  numbers.
- **Math change (Option 2A):** `rz01` / `rz12` switched from the asymmetric
  `diag(exp(iφ), 1, 1)` form to the symmetric `diag(exp(-iφ/2), exp(iφ/2), 1)`
  form. `r01` / `r12` formulae updated in lockstep to preserve composite
  output. See "Migration notes" above.
- Renamed `Identity` → `identity` (PEP 8 snake_case) with `Identity = identity`
  alias for backward compatibility.
- Tightened all type hints to `NDArray[np.complex128]`.
- `cnot`: `Exception` → `ValueError`; cached repeated `state_0 @ state_0.T`
  projector products into local variables.
- Full NumPy-style docstrings on every public function.

#### `src/qutritium/quantumcircuit/qc_utility.py`

**Lines:** ~243 → ~340.

- Replaced wildcard `from src.quantumcircuit.qc_elementary_matrices import *`
  with explicit imports of only what's used (none, post-cleanup; the wildcard
  was unused).
- Added shared `_require_params` validator for parametrized gates; previously
  a missing `parameter` argument produced an opaque `TypeError`/`IndexError`
  deep inside the dispatch.
- Bare `Exception` raises → `ValueError` (input validation), `KeyError`
  (unknown gate name).
- `multi_matrix_form`: explicit `KeyError` for the unhandled non-CNOT case
  (previously the function silently fell through and returned `None`).
- `statevector_to_state`: `tmp = tmp / 3` (true division — relied on
  `int(tmp % 3)` to mask the float drift) → `tmp //= 3` (correct floor
  division). `Exception` → `ValueError` with both shapes in the error.
- `print_statevector`: f-string formatting; no semantic change.
- **`checking_unitary` correctness fix:** rewrote to use `U @ U.conj().T`
  (the actual definition of unitarity, `U U† = I`) rather than `U @ inv(U)`
  (which only checks invertibility). Also replaced the brittle
  `abs(sum(U U⁻¹ - I)) < 1e-5` test (satisfiable by non-unitary
  matrices whose deviations cancel — e.g. `diag(1+ε, 1-ε, 1)`) with
  element-wise `np.allclose`.
- All gate cases now use `if … return` rather than `elif …` chains
  (no semantic change; flatter control flow).

#### `src/qutritium/quantumcircuit/instruction_structure.py`

**Lines:** ~170 → ~220.

- Import path `src.quantumcircuit.qc_utility` → `qutritium.quantumcircuit.qc_utility`.
- `gate_set: list[Union[str, Any]]` → `GATE_SET: frozenset[str]`. Constant
  naming convention; `frozenset` for O(1) membership; `Union[str, Any]` was
  meaningless (`Any` subsumes `str`). `gate_set` preserved as a sorted-list
  alias for backward compatibility.
- Default `second_qutrit_set=0` (a real bug — single-qutrit gates were
  silently being treated as two-qutrit gates with control on q0) →
  `second_qutrit_set=None`.
- Index validation moved to top of `__init__` (`IndexError` with explicit
  message instead of mid-construction `Exception`).
- `np.matrix(...).getH()` (deprecated NumPy `matrix` class) →
  `np.asarray(...).conj().T` on plain `ndarray`. Behaviour identical.
- `_verify_gate`: `Exception` → `ValueError` with sorted list of supported
  gates in the message.
- Stale commented-out import removed.
- All print statements converted to f-strings.

#### `src/qutritium/quantumcircuit/QC.py`

**Lines:** ~174 → ~310.

- Import paths updated.
- **Bug fix in `operation_set` setter:** in v0.0.1, the `@operation_set.setter`
  *extended* the internal list (`self._operation_set.extend(op)`), even
  though Python's setter contract is replacement. This made
  `circuit.operation_set = [ins]` silently append. Renamed the append-helper
  to a private `_extend_operation_set` method (single internal caller
  updated); the public setter now correctly *replaces* the list.
- **Bug fix in `__add__`:** v0.0.1's `__add__` *mutated* `self` and returned
  it, violating `a + b should leave a and b unchanged`. Rewrote to build and
  return a fresh `Qutrit_circuit`. Now passes the
  `test_circuit_addition_preserves_operations` regression test.
- **Bug fix in initial state dtype:** v0.0.1's default initial state was
  `np.array([[0]*dim]).T` — an integer array. The first complex matrix
  application then silently coerced. Now explicitly `dtype=complex`.
- `Exception` raises → `ValueError` (shape errors) and `RuntimeError`
  (state-machine violations like double-measurement).
- Added container protocol: `__len__`, `__iter__`, `__repr__`. None existed
  in v0.0.1, making circuits inspectable only via `print` or
  `circuit.operation_set`.
- Tightened all type hints; added `_Operation` type alias.

#### `src/qutritium/vm_backend/QASM_backend.py`

**Lines:** ~191 → ~270.

- Import paths updated.
- **Critical bug fix in `add_SPAM_noise`:** v0.0.1 used
  `self._operation_set.insert(__index=0, __object=...)` — invalid Python
  (the `list.insert` method does not accept dunder kwargs). Would raise
  `TypeError` the first time the SPAM-noise codepath was exercised. Fixed
  to `insert(0, ...)`. Also: gate names `'x+'`, `'x-'`, `'I'` (none of which
  are valid `GATE_SET` entries) → `'x_plus'`, `'x_minus'`, `'Identity'`.
- **Critical bug fix in `density_matrix`:** v0.0.1 returned `state @ state.T`
  (transpose, not conjugate transpose) — only correct for real statevectors.
  Fixed to `state @ state.conj().T`, the proper pure-state outer product.
  `test_density_matrix_is_hermitian_psd` regression-tests this.
- `add_SPAM_noise`: input range validation; `error_type != "Pauli_error"`
  now raises `NotImplementedError` instead of silently `pass`ing.
- `run`: validation moved to top (was at the bottom, which meant invalid
  inputs only triggered errors *after* potentially expensive simulation).
- `run`: shot sampling vectorised via `rng.choice(..., size=num_shots)`
  rather than a Python loop.
- `get_counts`: `dict((x, list.count(x)) for x in set(list))` (O(n²)) →
  `Counter` (O(n)).
- All `np.random` legacy calls → `np.random.default_rng()` Generator API.
- `plot`: lazy-imported matplotlib so the simulator itself doesn't drag in
  matplotlib at module load. Now uses `fig, ax = plt.subplots()` + `ax.set_*`
  rather than implicit `plt.*` state, returns the `Figure` for testability,
  no longer auto-`plt.show()`.
- `Exception` → `ValueError` / `RuntimeError` / `NotImplementedError`
  throughout.
- Tightened type hints; statevector copy on construction so the simulator
  doesn't share mutable state with the circuit.

#### `src/qutritium/decomposition/transpilation.py`

**Lines:** ~403 → ~280.

- **Removed the `Pulse_Wrapper` class entirely** (lines 223–403 of the v0.0.1
  file, ~180 lines). Removed dependent imports: `qiskit_ibm_provider.IBMBackend`,
  `qiskit.pulse.schedule.ScheduleBlock`, `src.pulse.{Pulse01, Pulse12}`,
  `src.pulse_creation.{Shift_phase, Set_frequency, GateSchedule}`, and
  `matplotlib.pyplot`. The class is preserved verbatim in
  `legacy/decomposition/transpilation.py.legacy.txt`.
- Import paths `src.X` → `qutritium.X`.
- `Parameter`, the "static class with one classmethod" anti-pattern, was
  promoted to a module-level function `get_parameters`. The `Parameter` class
  is kept as a backward-compat shim that delegates.
- `getattr(self.parameters, 'phi6')` (lookup-by-string-name) →
  `self.parameters.phi6` (direct attribute access). Faster, IDE-completable.
- **Latent bug fix in `get_parameters`:** the middle branch (`|U[2,2]| ≈ 0`)
  could fall through all three sub-branches without ever assigning
  `phi_4 / phi_5 / phi_6`, raising `UnboundLocalError`. Initialised to `0.0`
  defaults at the top of the branch.
- `__str__` and `__repr__` were duplicated character-for-character; now
  `__repr__` defers to `__str__`.
- `assert su3.shape[0] == 3` → `if su3.shape != (3, 3): raise ValueError(...)`.
  Asserts can be disabled with `python -O` and should not be used for input
  validation.
- `native_list`: extracted phase computation into named locals; replaced the
  deeply-nested literal list with explicit construction.
- Added a `DecompositionAngles` named tuple as the return type for
  `get_parameters` (replaces the runtime `namedtuple('params', '...')`).
- Module docstring with the canonical decomposition formula in LaTeX.

---

### Moved (to `legacy/`)

The full list — preserved verbatim. None of these files are part of the
installable v1.0.0 package.

#### `src/` modules → `legacy/`

| From                                         | Reason                                                |
|----------------------------------------------|-------------------------------------------------------|
| `src/pulse.py`                               | Pulse-model abstraction; entirely Qiskit-pulse-based. |
| `src/pulse_creation.py`                      | Shift_phase, Set_frequency, GateSchedule helpers.     |
| `src/algo_implementation.py`                 | Pulse-based algorithm prototypes.                     |
| `src/analyzer.py`                            | Pulse-result analysis.                                |
| `src/utility.py`                             | Pulse-flavoured helpers (different from quantumcircuit/qc_utility.py). |
| `src/simple_backend_log.py`                  | IBM backend logging.                                  |
| `src/constant.py`                            | Mostly Qiskit-channel constants.                      |
| `src/Qiskit_schedule.png`                    | Generated by `Pulse_Wrapper.print_qiskit_schedule()`. |
| `src/backend/`                               | `backend_ibm.py` + IBM backend test notebook.         |
| `src/exceptions/`                            | `pulse_exception.py` (only.)                          |
| `src/calibration/`                           | Entire pulse-calibration suite — discriminator, drag, fine_tune, rough_rabi, transmission_reflection, utility. |
| `src/characterization/`                      | control_tomography_qudit + ramsey notebooks.          |
| `src/tomography/`                            | `Qutrit_tomo` was incomplete in v0.0.1 (`execute_tomography()` was a stub). Moved pending a hardware-agnostic rewrite. |
| `src/benchmarking/`                          | Empty stub package.                                   |
| `src/clifford/`                              | Empty stub package.                                   |

#### Top-level files → `legacy/`

| From                                | Reason                                                    |
|-------------------------------------|-----------------------------------------------------------|
| `main.py`                           | Empty PyCharm template (unused).                          |
| `README_covlant.md`                 | Earlier README draft (Covalent integration plan).         |
| `0 1 2 discrimination.svg`          | Discrimination plot from the Munich poster.               |
| `presentation.pdf`                  | The Munich poster itself.                                 |
| `paper_references/`                 | Two PDF papers cited by the project.                      |
| `output/`                           | Pre-generated discrimination figures.                     |
| `project_images/`                   | README banner asset.                                      |

#### Scripts → `legacy/scripts/`

| From                                           | Reason                                              |
|------------------------------------------------|-----------------------------------------------------|
| `scripts/QC_decomposition_script.py`           | Used `Pulse_Wrapper`.                               |
| `scripts/VZ-proof.ipynb`                       | Pulse-based VZ-gate demonstration notebook.         |
| `scripts/decomposition_permutation_gate.ipynb` | Pulse-based permutation-gate decomposition notebook. |
| `scripts/package_tutorial.ipynb`               | Pulse-based tutorial.                               |
| `scripts/internal_use_only/`                   | calibration.ipynb, read_data.ipynb, read_quito.ipynb, constant.py, utility.py — Backend-specific, internal use. |

#### Tests → `legacy/test/`

| From                            | Reason                                            |
|---------------------------------|---------------------------------------------------|
| `test/Decomp_test/decompose_test.py` | Used `Pulse_Wrapper` + `IBMProvider`. Replaced by pytest tests in `test/VM_test/VM_test.py`. |
| `test/calibration_test/`        | Pulse calibration tests.                          |
| `test/pulse_model_test/`        | Pulse model tests.                                |
| `test/qiskit_api_test/`         | Tests requiring an IBM Quantum account.           |

---

### Removed

| Path                | Reason                                                          |
|---------------------|-----------------------------------------------------------------|
| `setup.py`          | Replaced by PEP 621 metadata in `pyproject.toml`.               |
| `setup.cfg`         | Replaced by PEP 621 metadata in `pyproject.toml`.               |
| `requirements.txt`  | The 117-line file pinned the entire jupyter ecosystem; deps are now in `pyproject.toml` (`numpy`, `scipy` only for the runtime; `[dev]`/`[plot]` extras for everything else). |
| `mkdocs.yml`        | Removed in v1.0.0. README + NumPy-style docstrings are the documentation for the package's current scope (~6 public symbols); a full mkdocs site is overkill and a maintenance trap. The decision can be revisited if the API surface grows substantially in a future release. The v0.0.1 `docs/` markdown stubs are preserved at `legacy/docs/` for historical reference. |
| `CHANGES.txt`       | Replaced by this `CHANGES.md`.                                  |
| `site/`             | mkdocs build output checked into the repo (regenerable; should not be in source control). |
| `.idea/`            | PyCharm IDE configuration directory (project-specific, should not be in source control). |
| `tmp/`              | Empty directory.                                                |
| `experiments/`      | Empty directory.                                                |

---

### Verified

The full Phase 1 refactor was verified end-to-end:

```
$ pip install -e .
$ python -m pytest test/ -v
============================== 12 passed in 0.13s ==============================
```

Key correctness checks:

- **SU(3) reconstruction fidelity** for the discrete Fourier matrix `U_ft`:
  `1.0000000000` (machine precision).
- **Random SU(3) reconstruction fidelity** over 4 parametrised seeds (and an
  earlier 20-sample sweep): worst case `0.9999999999`.
- **Two-qutrit Hadamard + CNOT** distribution: weight on `{|00>, |11>, |22>}`
  matches expected uniform-third within 5σ multinomial confidence.
- **`r01` / `r12` outputs**: bit-identical to v0.0.1 (max diff 3.34e-16) over
  1000 random `(phi, theta)` samples, confirming the Option 2A symmetric
  `rz` change does not perturb the SU(3) decomposition's inputs.

---

## [0.0.1] — 2023-03-04

Initial Munich Quantum Software Conference 2023 release. Qiskit-pulse-based
qutrit calibration and decomposition for IBM Quantum hardware. See
`legacy/` for the full source.
