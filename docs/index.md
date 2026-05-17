# Qutritium

**A hardware-agnostic Python library for qutrit quantum computing.**

Qutritium provides qutrit (three-level quantum system) gate definitions, circuit
construction, statevector simulation, and SU(3) decomposition. It runs entirely
in software — no quantum hardware or cloud account required.

## Features

- **34 qutrit gates** — 29 single-qutrit (fixed + parametric) and 5 two-qutrit gates
- **Circuit model** — build and compose qutrit circuits of arbitrary width
- **Statevector simulator** — exact simulation with Born-rule sampling
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

## Supported by

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

## Links

- [GitHub Repository](https://github.com/spham1611/qutritium)
- [Tutorial Notebook](https://github.com/spham1611/qutritium/blob/main/examples/tutorial.ipynb)
- [Changelog](changelog.md)
