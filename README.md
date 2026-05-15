<div align="center">

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow.svg?style=for-the-badge)](https://www.python.org/)

# Qutritium

**A hardware-agnostic Python library for qutrit quantum computing.**

*Build, simulate, and decompose three-level quantum circuits.*

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
from qutritium.gates import H3, CSUM, Rx01
import numpy as np

# Qutrit Bell state: H3 + CSUM → (|00⟩ + |11⟩ + |22⟩) / √3
qc = QutritCircuit(2, None)
qc.append(H3(), qutrit=0)
qc.append(CSUM(), qutrit=0, target_qutrit=1)
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
print(dec.parameters)      # extracted angles
print(dec.reconstruct())   # ≈ U to machine precision
```

## Gate Library

### Single-qutrit gates

| Category | Gates |
|----------|-------|
| Pauli-X  | `X01`, `X02`, `X12` |
| Pauli-Y  | `Y01`, `Y02`, `Y12` |
| Pauli-Z  | `Z01`, `Z02`, `Z12` |
| Shifts   | `XPlus` (cyclic), `XMinus` (inverse) |
| Discrete | `H3` (Hadamard/DFT), `S3`, `T3`, `UFT`, `I3` |
| Rotations | `Rx01`, `Rx02`, `Rx12`, `Ry01`, `Ry02`, `Ry12`, `Rz01`, `Rz02`, `Rz12` |
| Generalized | `G01(θ,φ)`, `G02(θ,φ)`, `G12(θ,φ)` — native trapped-ion gate |
| Diagonal | `Ud(φ₁,φ₂,φ₃)` — virtual-Z in hardware |

### Two-qutrit gates

| Gate | Action |
|------|--------|
| `CSUM` | \|c,t⟩ → \|c, (t+c) mod 3⟩ |
| `CPhase` | \|c,t⟩ → ω^{c·t} \|c,t⟩ |
| `SWAP3` | \|a,b⟩ → \|b,a⟩ |
| `CNOT3` | Legacy v0.0.1 CNOT (= CSUM on adjacent qutrits) |

All gates inherit from `Gate` and provide `.matrix()`, `.inverse()`, `.is_unitary()`, `.label`, `.params`.

## Package Structure

```
qutritium/
├── gates/               # Gate objects (Phase 2)
│   ├── base.py          #   Gate ABC + _DaggerGate
│   ├── single_qutrit.py #   29 single-qutrit gates
│   └── two_qutrit.py    #   5 two-qutrit gates
├── circuit/             # Circuit infrastructure
│   ├── elementary_matrices.py  # Raw 3×3 / 9×9 unitaries
│   ├── instruction.py          # Instruction + GATE_SET
│   ├── qutrit_circuit.py       # QutritCircuit container
│   └── utils.py                # Statevector utilities
├── simulator/           # QASMSimulator (statevector)
├── decomposition/       # SU(3) → native rotations
└── legacy/              # v0.0.x Qiskit-pulse code (not installable)
```

## History

Qutritium was originally built for calibrating qutrits on IBM superconducting hardware,
presented at the **Munich Quantum Software Conference 2023** and funded by a
**Unitary Fund** microgrant. The v1.0.0 release pivoted to a hardware-agnostic
library; the original pulse code is preserved under `legacy/`.

## Authors

- **[Son Pham](https://github.com/spham1611)** — Duke University · sph40@duke.edu
- **[Tien Nguyen](https://github.com/ngdnhtien)** — École Polytechnique, France
- **[Bao Bach](https://github.com/bachbao)** — University of Delaware, USA
- **[Charlie (abdomsisn)](https://github.com/abdomsisn)** — Duke University · abdomsisn.haobei@gmail.com

## License

[MIT](LICENSE.txt)
