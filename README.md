[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

# Qutritium

Hardware-agnostic Python library for qutrit (3-level) quantum computing.
Build circuits, simulate them, decompose arbitrary SU(3) unitaries.

Funded by a [Unitary Fund](https://unitary.fund) microgrant. Originally presented
at the Munich Quantum Software Conference (October 2023).

## Install

```bash
pip install qutritium
# or for development:
pip install -e ".[dev]"
```

## Usage

```python
from qutritium import QutritCircuit, QASMSimulator
from qutritium.gates import H3, CSUM
import numpy as np

# Qutrit Bell state: H3 on qutrit 0, then CSUM
qc = QutritCircuit(2, None)
qc.append(H3(), first_qutrit=0)
qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
qc.measure_all()

sim = QASMSimulator(qc)
sim.run(num_shots=10_000)
print(sim.get_counts())   # {'00': ~3333, '11': ~3333, '22': ~3333}
```

SU(3) decomposition:

```python
from qutritium import SU3Decomposition

U = ...  # any 3x3 unitary
dec = SU3Decomposition(U, qutrit_index=0, n_qutrits=1)
print(dec.angles)
print(dec.reconstruct())  # should match U
```

## Gates

Single-qutrit: `X01`, `X02`, `X12`, `Y01`..., `Z01`..., `XPlus`, `XMinus`,
`H3`, `S3`, `T3`, `UFT`, `I3`, rotation gates `Rx01(θ)`, `Ry01(θ)`, `Rz01(φ)` etc.,
generalized rotations `G01(θ,φ)`, `G02(θ,φ)`, `G12(θ,φ)`, diagonal `Ud(φ₁,φ₂,φ₃)`.

Two-qutrit: `CSUM`, `CPhase`, `SWAP3`, `CNOT3` (legacy).

All gates inherit from `Gate` and have `.matrix()`, `.inverse()`, `.is_unitary()`,
`.label`, `.params`.

## Layout

```
qutritium/
├── gates/           # Gate objects (base.py, single_qutrit.py, two_qutrit.py)
├── circuit/         # Circuit container, instruction, elementary 3x3 matrices
├── simulator/       # Statevector simulator
├── decomposition/   # SU(3) → native rotations
└── legacy/          # Original v0.0.x Qiskit-pulse code (archived, not installable)
```

## Background

Qutritium started as a Qiskit-pulse package for calibrating qutrits on IBM
superconducting hardware. The v1.0.0 release dropped the IBM dependency and
became hardware-agnostic; the original pulse code is kept under `legacy/` for
reference.

## Authors

- [Son Pham](https://github.com/spham1611) — Duke University
- [Tien Nguyen](https://github.com/ngdnhtien) — École Polytechnique
- [Bao Bach](https://github.com/bachbao) — University of Delaware
- [Charlie (abdomsisn)](https://github.com/abdomsisn) — Duke University

## License

MIT — see [LICENSE.txt](LICENSE.txt)
