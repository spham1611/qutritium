# Changelog

## [1.5.0] - 2026-06-17

### Added

- `qutritium.tomography`: single-qutrit process tomography —
  `process_tomography_circuits` (12 MUB input states, each measured in all four
  MUBs), `reconstruct_process` (Choi-matrix reconstruction by linear inversion),
  and `choi_to_kraus` (Kraus operators from the Choi spectrum).
- `reconstruct_state(method="projected_lls")`: projects the linear-inversion
  estimate onto the closest physical (positive, unit-trace) density matrix via
  the Smolin–Gambetta–Smith (2012) algorithm.
- `examples/process_tomography.ipynb` tutorial.

### Breaking

- None. v1.5.0 is purely additive.

### Deferred

- OpenQASM 3 import/export moved to v2.0.

## [1.4.0] - 2026-06-09

### Added

- `qutritium.channels`: `Channel` (Kraus CPTP map), preset channels
  (`depolarizing_channel`, `dephasing_channel`, `amplitude_damping_channel`,
  `pauli_channel`), `ReadoutError` confusion matrix, `NoiseModel`, and
  `SPAMNoiseModel`.
- `Simulator.set_noise_model`: attach noise to a simulator. The
  `DensityMatrixSimulator` applies Kraus gate/prep channels interleaved with
  the unitary evolution; both backends apply readout error at sampling time.
  `QASMSimulator` rejects Kraus noise (`NotImplementedError`).
- `qutritium.tomography`: single-qutrit MUB state tomography — `mub_bases`,
  `state_tomography_circuits`, `reconstruct_state` (linear inversion), plus
  `plot_density_matrix` and `plot_tomography_comparison` (matplotlib).

### Fixed

These wrong-result bugs shipped in v1.3.0 and are corrected here:

- `metrics`: kets are normalized and density matrices validated (Hermitian,
  unit trace) before use; `state_fidelity` / `purity` / `von_neumann_entropy`
  no longer return wrong values for unnormalized or invalid inputs.
- `instruction`: the two-qutrit adjacency check no longer lets
  `second_qutrit == 0` slip through; `effect_matrix` SWAP-conjugates
  fixed-orientation two-qutrit gates (CSUM / CSUMDag / CPhase / SWAP) when
  `control > target`; `Instruction.inverse()` round-trips for custom
  instructions.
- `simulator`: `run()` clips tiny-negative probabilities before sampling;
  `RunTimeError` -> `RuntimeError`.
- `elementary_matrices`: added the missing `__all__`.

### Breaking

- None. v1.4.0 is purely additive. (The pre-1.0 `add_SPAM_noise` method does
  not exist in the hardware-agnostic package; use `SPAMNoiseModel` with
  `set_noise_model` instead.)

## [1.3.0] - 2026-05-29

### Added

- qutritium.metrics: state_fidelity, trace_distance, purity,
  von_neumann_entropy, process_fidelity, average_gate_fidelity.
- Simulator ABC
- QASMSimulator inherits from Simulator .
- DensityMatrixSimulator with rho -> U rho U^dag evolution,
  expectation_value(), and partial_trace().
- QutritCircuit.depth(), gate_count(), to_matrix().
- Public CPhaseDag gate.

### Changed

- Instruction.effect_matrix is now lazy (cached_property).
- QutritCircuit.draw() no longer prints as a side effect.

### Fixed

- statevector_to_state drops numerical-noise coefficients.

### Breaking

- Removed is_dagger= from QutritCircuit.append(); use gate.inverse().

## [1.2.1] — 2026-05-21

Fix maintainer email address in README.

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

Gate test suite in `test/test_gates.py` (~50 test functions, parametrized
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
