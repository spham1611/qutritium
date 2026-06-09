# Qutritium

**A hardware-agnostic Python library for qutrit quantum computing.**

Qutritium provides qutrit (three-level quantum system) gate definitions, circuit
construction, statevector and density-matrix simulation, metrics, and SU(3)
decomposition. It runs entirely in software — no quantum hardware or cloud
account required.

## Features

- **34 qutrit gates** — 29 single-qutrit (fixed + parametric) and 5 two-qutrit gates
- **Circuit model** — build and compose qutrit circuits of arbitrary width;
  introspect with `depth()`, `gate_count()`, `to_matrix()`
- **Two simulators** — exact statevector with Born-rule sampling, and a
  density-matrix backend (expectation values, partial trace) for mixed states
- **Metrics** — state/process fidelity, trace distance, purity, von Neumann entropy
- **Noise modeling** — Kraus channels (depolarizing, dephasing, amplitude damping,
  Pauli), a simulator-level `NoiseModel`, and classical readout error
- **State tomography** — mutually-unbiased-basis circuits with linear-inversion
  reconstruction, plus density-matrix visualization
- **SU(3) decomposition** — factor any 3×3 unitary into native rotations

## Quick Install

```bash
pip install qutritium
```

## Quick Example

```python
from qutritium import QutritCircuit, QASMSimulator
from qutritium.gates import H3, CSUM

# Build a 2-qutrit Bell state circuit
qc = QutritCircuit(2, None)
qc.append(H3(), first_qutrit=0)
qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
qc.measure_all()

# Simulate
sim = QASMSimulator(qc)
sim.run(num_shots=1000)
print(sim.get_counts())
# {'00': ~333, '11': ~333, '22': ~333}
```

## How it fits together

```text
  Gate                  qutritium.gates - X01, H3, CSUM, Rx01, ...
    |                   a unitary; has .matrix() / .inverse()
    |  qc.append(gate, qutrit)
    v
  QutritCircuit         ordered operations (+ measure_all)
    |                   - each append wraps the gate as an Instruction
    |                     (gate + target qutrit(s); lazy 3^n x 3^n effect_matrix)
    |                   - introspect: .draw() .depth() .gate_count() .to_matrix()
    |  hand the circuit to a simulator
    v
  Simulator             QASMSimulator (statevector)
    |                   DensityMatrixSimulator (rho - mixed states, noise)
    |                   - optional: .set_noise_model(NoiseModel(...))
    v
  results               .get_counts()  .probabilities()  .return_final_state()
    |
    +--> tomography.reconstruct_state   counts -> reconstructed rho
    +--> metrics                        state_fidelity, purity, entropy, ...

  SU3Decomposition(U) --> QutritCircuit   decompose any 3x3 unitary into
                                          native gates, then run it
```

## Supported by

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

## Links

- [GitHub Repository](https://github.com/spham1611/qutritium)
- [Tutorial Notebook](https://github.com/spham1611/qutritium/blob/main/examples/tutorial.ipynb)
- [Changelog](changelog.md)
