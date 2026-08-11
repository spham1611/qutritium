# Changelog

## [Unreleased]

`Simulator.run()` accepts an optional `seed` argument
(`run(num_shots, seed=17)`) that makes shot sampling reproducible on both
simulators.

## [1.5.2] — 2026-07-04

Add JOSS paper with Zenodo archive enable

## [1.5.1] — 2026-07-03

Renamed `QASMSimulator` to `StatevectorSimulator` — the class evolves a
statevector, and there is no QASM layer (OpenQASM was dropped from the roadmap).
`QASMSimulator` stays as a deprecated alias that warns on instantiation and will
be removed in v2.0; switch imports to `StatevectorSimulator`. No breaking
changes — old code still runs.

`NoiseModel.add_quantum_error`/`add_prep_error` now reject multi-qutrit channels
and negative qutrit indices with a clear `ValueError` instead of crashing later
inside NumPy.

## [1.5.0] — 2026-06-17

Added single-qutrit **process tomography** to `qutritium.tomography`:
`process_tomography_circuits` builds the 12 MUB input states and their MUB
measurement circuits, `reconstruct_process` inverts the resulting counts into a
Choi matrix by linear least squares, and `choi_to_kraus` reads Kraus operators
off the Choi spectrum. `reconstruct_state` gains a `"projected_lls"` method that
projects the linear-least-squares estimate onto the closest physical density matrix
(Smolin, Gambetta & Smith, 2012).

Added `examples/process_tomography.ipynb`. The test suite is now 380 tests.

Dropped the unused `scipy` runtime dependency — `qutritium` now needs only NumPy.

OpenQASM 3 import/export is deferred to v2.0 (since dropped from the roadmap
entirely). Purely additive — no breaking changes.

## [1.4.0] — 2026-06-09

Added `qutritium.channels` — Kraus-operator noise channels (`Channel`,
`depolarizing_channel`, `dephasing_channel`, `amplitude_damping_channel`,
`pauli_channel`), a classical `ReadoutError`, and the `NoiseModel` /
`SPAMNoiseModel` containers. Noise attaches to a simulator with
`Simulator.set_noise_model`: the `DensityMatrixSimulator` applies Kraus gate and
prep channels, both backends apply readout error, and the `QASMSimulator`
(renamed `StatevectorSimulator` in v1.5.1) rejects Kraus noise.

Added `qutritium.tomography` — single-qutrit mutually-unbiased-basis state
tomography (`mub_bases`, `state_tomography_circuits`, `reconstruct_state` via
linear least squares) plus density-matrix visualization (`plot_density_matrix`,
`plot_tomography_comparison`).

Added `examples/noise_and_tomography.ipynb`. The test suite is now 370 tests.

Fixed wrong-result bugs that shipped in v1.3.0: metrics now normalize kets and
validate density matrices; the two-qutrit adjacency check no longer admits
`second_qutrit == 0`; `effect_matrix` orients fixed two-qutrit gates correctly
when control > target; `Instruction.inverse()` round-trips for custom gates; and
`run()` clips tiny-negative probabilities before sampling.

Purely additive — no breaking changes.

## [1.3.0] — 2026-05-29

Added `qutritium.metrics` (`state_fidelity`, `trace_distance`, `purity`,
`von_neumann_entropy`, `process_fidelity`, `average_gate_fidelity`), the
`Simulator` ABC, and `DensityMatrixSimulator` (with `expectation_value` and
`partial_trace`). Added `QutritCircuit.depth()`, `gate_count()`, `to_matrix()`,
and a public `CPhaseDag` gate. `draw()` now returns its string instead of
printing.

Breaking: removed `is_dagger=` from `QutritCircuit.append()` — pass
`gate.inverse()` instead.

## [1.2.1] — 2026-05-21

Fixed the maintainer email address in README.

## [1.2.0] — 2026-05-17

Added GitHub Actions CI (`.github/workflows/test.yml`) for Python 3.10/3.11/3.12/3.13.
Added MkDocs documentation site (`mkdocs.yml` + `docs/`) with API reference for gates,
circuits, simulator, decomposition; getting-started guides; Bell state tutorial; legacy
code documentation page; and changelog.

Added `examples/tutorial.ipynb` — end-to-end notebook covering single-qutrit gates,
Bell state preparation, measurement sampling, density matrix, parametric rotations,
and SU(3) decomposition. Runs in <10 seconds.

Added `MANIFEST.in` for clean sdist/wheel packaging (excludes legacy/, test/, docs/).
Added `docs` optional dependency group to `pyproject.toml`.
Added `Documentation` URL to `[project.urls]`.

Tests passing across `test/VM_test.py` and `test/test_gates.py`.

## [1.1.0] — 2026-05-15

Added `qutritium.gates` subpackage with first-class Gate objects. 29 single-qutrit
gates (fixed + parametric) and 5 two-qutrit gates, all with `.matrix()`, `.inverse()`,
`.is_unitary()`. Replaced the old string-dispatch `add_gate()` with `QutritCircuit.append(gate, ...)`.

New two-qutrit matrix functions in `elementary_matrices`: `csum()`, `csum_dag()`,
`cphase()`, `swap3()`.

Gate test suite in `test/test_gates.py`

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
