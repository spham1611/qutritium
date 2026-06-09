<div align="center">

[![Tests](https://github.com/spham1611/qutritium/actions/workflows/test.yml/badge.svg)](https://github.com/spham1611/qutritium/actions)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow.svg?style=for-the-badge)](https://www.python.org/)

# Qutritium

**A hardware-agnostic Python library for qutrit quantum computing.**

*Build, simulate, and decompose three-level quantum circuits — with metrics, noise modeling, and state tomography.*

</div>

---

## Installation

```bash
pip install qutritium            # release
pip install -e ".[dev]"          # editable + dev tools
```

## Quick Start

```python
from qutritium import QutritCircuit, QASMSimulator
from qutritium.gates import H3, CSUM
import numpy as np

# Qutrit Bell state: H3 + CSUM → (|00⟩ + |11⟩ + |22⟩) / √3
qc = QutritCircuit(2, None)
qc.append(H3(), first_qutrit=0)
qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
qc.measure_all()

sim = QASMSimulator(qc)
sim.run(num_shots=10_000)
print(sim.get_counts())   # {'00': ~3333, '11': ~3333, '22': ~3333}
```

Decompose an arbitrary SU(3) unitary:

```python
from qutritium import SU3Decomposition
import numpy as np

U = ...  # any 3×3 unitary
dec = SU3Decomposition(U, qutrit_index=0, n_qutrits=1)
print(dec.angles)  # nine decomposition angles
print(dec.reconstruct())  # ≈ U to machine precision
```

Add noise on the simulator and reconstruct a state with tomography:

```python
from qutritium import QutritCircuit, DensityMatrixSimulator
from qutritium.channels import NoiseModel, depolarizing_channel
from qutritium.gates import X01

qc = QutritCircuit(1, None)
qc.append(X01(), first_qutrit=0)
qc.measure_all()

nm = NoiseModel()
nm.add_quantum_error(depolarizing_channel(0.1), "X01")  # noise lives on the sim, not the circuit
dm = DensityMatrixSimulator(qc)
dm.set_noise_model(nm)
dm.run(num_shots=2000)
print(dm.get_counts())
```

## Gate Library

### Single-qutrit gates

| Category    | Gates                                                                  |
|-------------|------------------------------------------------------------------------|
| Pauli-X     | `X01`, `X02`, `X12`                                                    |
| Pauli-Y     | `Y01`, `Y02`, `Y12`                                                    |
| Pauli-Z     | `Z01`, `Z02`, `Z12`                                                    |
| Shifts      | `XPlus` (cyclic), `XMinus` (inverse)                                   |
| Discrete    | `H3` (Hadamard/DFT), `S3`, `T3`, `UFT`, `I3`                           |
| Rotations   | `Rx01`, `Rx02`, `Rx12`, `Ry01`, `Ry02`, `Ry12`, `Rz01`, `Rz02`, `Rz12` |
| Generalized | `G01(θ,φ)`, `G02(θ,φ)`, `G12(θ,φ)` — native trapped-ion gate           |
| Diagonal    | `Ud(φ₁,φ₂,φ₃)` — virtual-Z in hardware                                 |

### Two-qutrit gates

| Gate      | Action                                          |
|-----------|-------------------------------------------------|
| `CSUM`    | \|c,t⟩ → \|c, (t+c) mod 3⟩                      |
| `CSUMDag` | \|c,t⟩ → \|c, (t−c) mod 3⟩ (CSUM inverse)       |
| `CPhase`  | \|c,t⟩ → ω^{c·t} \|c,t⟩                         |
| `SWAP3`   | \|a,b⟩ → \|b,a⟩                                 |
| `CNOT3`   | Legacy v0.0.1 CNOT (= CSUM on adjacent qutrits) |

All gates inherit from `Gate` and provide `.matrix()`, `.inverse()`, `.is_unitary()`, `.label`, `.params`.

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

## Package Structure

```
src/qutritium/
├── gates/               # Gate objects
│   ├── base.py          #   Gate ABC + _DaggerGate
│   ├── single_qutrit.py #   29 single-qutrit gates
│   └── two_qutrit.py    #   5 two-qutrit gates
├── circuit/             # Circuit infrastructure
│   ├── elementary_matrices.py  # Raw 3×3 / 9×9 unitaries
│   ├── instruction.py          # Instruction + GATE_SET
│   ├── qutrit_circuit.py       # QutritCircuit container
│   └── utils.py                # Statevector utilities
├── simulator/           # QASMSimulator (statevector) + DensityMatrixSimulator
├── channels/            # Noise channels, NoiseModel, ReadoutError, SPAM
├── metrics/             # Fidelity, trace distance, purity, entropy
├── tomography/          # MUB state tomography + visualization
└── decomposition/       # SU(3) → native rotations
```

Supporting files at repo root:

```
.github/workflows/       # CI (test.yml, docs.yml)
docs/                    # MkDocs source → spham1611.github.io/qutritium
examples/                # Bell-state and noise+tomography tutorial notebooks
test/                    # pytest suite
legacy/                  # v0.0.x Qiskit-pulse code (archived, not installed)
```

## Documentation

Full docs: **<https://spham1611.github.io/qutritium/>**

Tutorial notebooks: [`examples/tutorial.ipynb`](examples/tutorial.ipynb) (core) and
[`examples/noise_and_tomography.ipynb`](examples/noise_and_tomography.ipynb) (noise + tomography)

## History

Qutritium was originally built for calibrating qutrits on IBM superconducting hardware,
presented at the **Munich Quantum Software Conference 2023** and funded by a
**Unitary Fund** microgrant. The v1.0.0 release pivoted to a hardware-agnostic
library; the original pulse code is preserved under `legacy/`.

## Authors

- **[Son Pham](https://github.com/spham1611)** — Duke University · sph40@duke.edu
- **[Tien Nguyen](https://github.com/ngdnhtien)** — École Polytechnique, France · tienphys@gmail.com
- **[Bao Bach](https://github.com/bachbao)** — University of Delaware, USA · bachgiabao12@gmail.com
- **[Charlie (abdomsisn)](https://github.com/abdomsisn)** — Duke University · abdomsisn.haobei@gmail.com

## License

[MIT](LICENSE.txt)