# Changelog

All notable changes to Qutritium are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/) ·
Versioning: [Semantic Versioning](https://semver.org/)

---

## [1.1.0] — 2026-05-15

### Added

- **`qutritium.gates` subpackage** — first-class Gate objects replacing string-based dispatch.
  - `Gate` ABC with `.matrix()`, `.label`, `.num_qutrits`, `.params`, `.inverse()`, `.is_unitary()`.
  - 16 fixed single-qutrit gates: `I3`, `X01/X02/X12`, `Y01/Y02/Y12`, `Z01/Z02/Z12`, `XPlus/XMinus`, `H3`, `S3`, `T3`, `UFT`.
  - 13 parametric single-qutrit gates: `Rx01/Rx02/Rx12`, `Ry01/Ry02/Ry12`, `Rz01/Rz02/Rz12`, `G01/G02/G12`, `Ud`.
  - 5 two-qutrit gates: `CSUM`, `CSUMDag`, `CNOT3`, `CPhase`, `SWAP3`.
- **`QutritCircuit.append(gate, first_qutrit, ...)`** — the single method for adding gates to circuits. Accepts `Gate`
  instances from `qutritium.gates`.
- Removed legacy `add_gate()` and `add_customized_gate()` string-dispatch methods; all callers (decomposition, tests,
  scripts) migrated to `append()`.
- **`elementary_matrices`**: `csum()`, `csum_dag()`, `cphase()`, `swap3()` — two-qutrit matrix functions (consistency with single-qutrit pattern).
- **`test/test_gates.py`** — 157 new tests (169 total with Phase 1).
- `Gate` exported from top-level `qutritium` package.

### Changed

- `two_qutrit.py` gate classes now delegate to `elementary_matrices` instead of computing matrices inline.
- Fixed PEP 8 indent issues in `cnot()` and overlong docstring lines in `elementary_matrices.py`.

---

## [1.0.0] — 2026-05-07

### Summary

**Major release.** Pivots Qutritium from a Qiskit-pulse / IBM-hardware-coupled
package into a hardware-agnostic Python library for qutrit quantum computing.

Runtime dependencies: `numpy`, `scipy` only. All Qiskit-pulse and IBM-backend
code moved to `legacy/` (not installable). See `changes_summary.txt` for the
full file-by-file diff.

### Breaking changes

1. Import paths: `src.X` → `qutritium.X`.
2. `Pulse_Wrapper` removed from package (preserved in `legacy/`).
3. `rz01`/`rz12` standardised to symmetric form `diag(exp(-iφ/2), exp(iφ/2), 1)`.
   Composite `r01`/`r12` outputs are bit-identical to v0.0.1.

### Highlights

- PEP 621 `pyproject.toml` replacing `setup.py`/`setup.cfg`/`requirements.txt`.
- `CITATION.cff` for machine-readable citation metadata.
- Bug fixes: `operation_set` setter, `__add__` mutation, `density_matrix` conjugate transpose, `add_SPAM_noise` kwargs, `checking_unitary` correctness.
- 12-test pytest suite covering SU(3) decomposition, circuit ops, simulator.

---

## [0.0.1] — 2023-03-04

Initial release. Qiskit-pulse-based qutrit calibration and decomposition for
IBM Quantum hardware. Presented at Munich Quantum Software Conference 2023.
See `legacy/` for the full source.
