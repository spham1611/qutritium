# Changelog

## [1.2.0] — 2026-05-17

Phase 3 polish and release engineering.

Added GitHub Actions CI (`test.yml`) for Python 3.10/3.11/3.12/3.13.
Added MkDocs documentation site with API reference, tutorials, and legacy page.
Added `examples/tutorial.ipynb` — end-to-end Bell state tutorial notebook.
Added `MANIFEST.in` for clean sdist packaging.
Updated `pyproject.toml` with docs dependencies, Documentation URL.
Added `py.typed` marker for PEP 561 typed package support.

168 tests passing across `test/VM_test.py` and `test/test_gates.py`.

## [1.1.0] — 2026-05-15

Added `qutritium.gates` subpackage with first-class Gate objects. 29 single-qutrit
gates (fixed + parametric) and 5 two-qutrit gates, all with `.matrix()`, `.inverse()`,
`.is_unitary()`. Replaced the old string-dispatch `add_gate()` with `QutritCircuit.append(gate, ...)`.

New two-qutrit matrix functions in `elementary_matrices`: `csum()`, `csum_dag()`,
`cphase()`, `swap3()`.

Comprehensive gate test suite in `test/test_gates.py` (~50 test functions, parametrized
across all gate families).

## [1.0.0] — 2026-05-07

Major release. Pivoted from Qiskit-pulse / IBM-coupled package to a
hardware-agnostic library. Dependencies: numpy, scipy only.

All Qiskit-pulse and IBM-backend code moved to `legacy/` (not installable).

Breaking changes:

- Import paths changed from `src.X` to `qutritium.X`
- `Pulse_Wrapper` removed from installable package
- `rz01`/`rz12` standardised to symmetric form `diag(exp(-iφ/2), exp(iφ/2), 1)`

Added PEP 621 `pyproject.toml`, `CITATION.cff`, bug fixes (operation_set setter,
`__add__` mutation, density_matrix conjugate, SPAM noise kwargs, unitarity check).
12-test pytest suite.

## [0.0.1] — 2023-03-04

Initial release. Qiskit-pulse-based qutrit calibration for IBM Quantum hardware.
Presented at Munich Quantum Software Conference 2023.
