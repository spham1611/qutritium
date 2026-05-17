# Legacy Code

The `legacy/` directory at the repository root preserves the original Qutritium
codebase from 2023–2024.

## What it contains

The original Qiskit Pulse / IBM Quantum-based calibration code:

- **`calibration/`** — Rabi, Ramsey, DRAG, fine-tune, transmission/reflection
  frequency calibration, 0/1/2 state discrimination
- **`pulse.py`**, **`pulse_creation.py`** — Pulse model classes and schedule
  construction via `qiskit.pulse.build()`
- **`backend/`** — Custom `IBMProvider` wrapper
- **`algo_implementation.py`** — Demo algorithms running on IBM hardware
- **`tests/`** — Original tests for calibration, pulse, and Qiskit API

## Why it is archived

Two external changes made this code unrunnable:

1. **Qiskit Pulse was removed** from the Qiskit SDK in February 2025
2. **IBM removed pulse-level access** from cloud QPUs on 3 February 2025

## Status

- **Not installed** by `pip install qutritium`
- **Not importable** — imports reference deleted Qiskit modules
- **Preserved** as historical record of the Munich Quantum Software Conference
  2023 poster work

For full details, see [`legacy/README.md`](https://github.com/spham1611/qutritium/blob/main/legacy/README.md).
