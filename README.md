<div align="center">

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)

# Qutritium

**A hardware-agnostic Python library for qutrit (3-level) quantum computing.**

</div>

---

## What is Qutritium?

Qutritium is a Python package for working with **qutrit** quantum systems — three-level
generalisations of the familiar qubit. It provides:

- A clean object model for qutrit gates, instructions, and circuits.
- A numerically exact statevector **simulator** (`QASM_Simulator`) — no hardware
  backend required.
- A complete **SU(3) decomposition** of arbitrary 3×3 unitaries into native
  subspace rotations (`r01`, `r12`, diagonal phase).
- Reference implementations of the standard qutrit gate set: subspace Pauli
  analogues (`x01`, `x12`, ...), parametrised rotations (`rx01`, `ry12`, `rz01`, ...),
  the qutrit Hadamard / discrete Fourier gate (`hdm`, `u_ft`), the qutrit `CNOT`,
  and arbitrary diagonal phase gates (`u_d`).

This is the **v1.0.0** release. It depends only on `numpy` and `scipy`. The
v0.0.x releases shipped Qiskit-pulse-based calibration routines for IBM
quantum hardware; that code is preserved under [`legacy/`](./legacy/) as a
historical record of the work presented at the Munich Quantum Software
Conference 2023.

## Installation

```bash
pip install qutritium
```

For development:

```bash
git clone https://github.com/spham1611/qutritium.git
cd qutritium
pip install -e ".[dev]"
```

Optional extras:

- `[plot]` — matplotlib (only needed for `QASM_Simulator.plot()`)
- `[dev]` — pytest, ruff, black, mypy

## Quick start

```python
import numpy as np
from qutritium import Qutrit_circuit, QASM_Simulator, SU3_matrices

# Build a 2-qutrit Bell-like state with a Hadamard + CNOT.
qc = Qutrit_circuit(2, initial_state=None)
qc.add_gate("hdm", first_qutrit_set=0)
qc.add_gate("CNOT", first_qutrit_set=1, second_qutrit_set=0)
qc.measure_all()

sim = QASM_Simulator(qc)
sim.run(num_shots=10_000)
print(sim.get_counts())
# -> {'00': ~3333, '11': ~3333, '22': ~3333}
```

Decompose an arbitrary SU(3) into native rotations:

```python
omega = np.exp(1j * 2 * np.pi / 3)
U_ft = (1 / np.sqrt(3)) * np.array([
    [omega, 1.0, np.conj(omega)],
    [1.0,    1.0, 1.0           ],
    [np.conj(omega), 1.0, omega ],
], dtype=complex)

dec = SU3_matrices(U_ft, qutrit_index=0, n_qutrits=1)
print(dec.parameters)        # extracted theta_i, phi_j angles
print(dec.reconstruct())     # numerically equal to U_ft
```

## Module overview

| Module                                | Purpose                                                              |
|---------------------------------------|----------------------------------------------------------------------|
| `qutritium.quantumcircuit.QC`         | `Qutrit_circuit` -- the user-facing circuit container.               |
| `qutritium.quantumcircuit.instruction_structure` | `Instruction` -- a single gate application + the `GATE_SET` registry. |
| `qutritium.quantumcircuit.qc_elementary_matrices` | The library of 3×3 unitary primitives.                              |
| `qutritium.quantumcircuit.qc_utility` | Gate-name -> matrix dispatch and statevector utilities.              |
| `qutritium.vm_backend.QASM_backend`   | `QASM_Simulator` -- exact statevector simulator with optional SPAM noise. |
| `qutritium.decomposition.transpilation` | `SU3_matrices`, `Parameter`, `get_parameters` -- SU(3) decomposition. |

## Migration from v0.0.x

v1.0.0 is a **breaking** release. Imports change from `src.X` to `qutritium.X`:

```python
# v0.0.x
from src.quantumcircuit.QC import Qutrit_circuit
from src.vm_backend.QASM_backend import QASM_Simulator
from src.decomposition.transpilation import SU3_matrices, Pulse_Wrapper  # removed!

# v1.0.0
from qutritium import Qutrit_circuit, QASM_Simulator, SU3_matrices
# Pulse_Wrapper has been moved to legacy/ (no longer importable).
```

The `Pulse_Wrapper` class — which converted decomposed circuits into
Qiskit-pulse `ScheduleBlock`s for IBM hardware — is **removed** from the
installable package. Its source is preserved in `legacy/` for reference.

The `rz01` / `rz12` matrix definitions in `qc_elementary_matrices` were
standardised to the symmetric (textbook) form `diag(exp(-iφ/2), exp(iφ/2), 1)`,
matching `qc_utility.single_matrix_form`. The composite `r01` / `r12`
rotations consumed by the SU(3) decomposition are **bit-identical** to v0.0.1
(verified to machine precision over 1000 random samples), so existing
decomposition workflows produce the same matrices to machine precision.

See `CHANGES.md` for the full diff.

## Citation

If you use Qutritium in your research, please cite it via the `CITATION.cff`
file (GitHub will render a "Cite this repository" widget on the repo page).
The original work was presented as a poster at the Munich Quantum Software
Conference 2023 and supported by a Unitary Fund microgrant.

## Authors

- **[Son Pham](https://github.com/spham1611)** — Duke University, USA
- **[Tien Nguyen](https://github.com/ngdnhtien)** — École Polytechnique, France
- **[Bao Bach](https://github.com/bachbao)** — University of Delaware, USA

## License

[MIT License](LICENSE.txt)

## Documentation

This README plus the docstrings in the source are the documentation. Every public class
and function carries a NumPy-style docstring; in an interactive session, `help(Qutrit_circuit)`
or `?Qutrit_circuit` (in IPython/Jupyter) gives you the full API for any symbol. See
[`CHANGES.md`](CHANGES.md) for the v1.0.0 refactor notes and migration guide.
