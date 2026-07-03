# Quick Start

## Concepts

Qutritium models quantum computation with **qutrits** — three-level quantum systems
with computational basis states $|0\rangle$, $|1\rangle$, $|2\rangle$.

The workflow is: **build a circuit** → **simulate** → **read results**.

## Building a circuit

```python
from qutritium import QutritCircuit
from qutritium.gates import H3, X01, CSUM

# Create a 2-qutrit circuit (initialised to |00⟩)
qc = QutritCircuit(n_qutrit=2, initial_state=None)

# Apply gates
qc.append(H3(), first_qutrit=0)             # Hadamard on qutrit 0
qc.append(CSUM(), first_qutrit=0, second_qutrit=1)  # Entangle
qc.measure_all()
```

## Simulating

```python
from qutritium import StatevectorSimulator

sim = StatevectorSimulator(qc)
sim.run(num_shots=3000)
print(sim.get_counts())
# {'00': ~1000, '11': ~1000, '22': ~1000}
```

## Accessing the statevector

```python
qc2 = QutritCircuit(1, None)
qc2.append(H3(), first_qutrit=0)

sim2 = StatevectorSimulator(qc2)
state = sim2.return_final_state()
print(state)  # 3×1 column vector with amplitudes 1/√3
```

## Gate library overview

**Fixed single-qutrit gates:** `I3`, `X01`, `X02`, `X12`, `Y01`, `Y02`, `Y12`,
`Z01`, `Z02`, `Z12`, `XPlus`, `XMinus`, `H3`, `S3`, `T3`, `UFT`

**Parametric single-qutrit gates:** `Rx01(θ)`, `Rx02(θ)`, `Rx12(θ)`,
`Ry01(θ)`, `Ry02(θ)`, `Ry12(θ)`, `Rz01(φ)`, `Rz02(φ)`, `Rz12(φ)`,
`G01(θ, φ)`, `G02(θ, φ)`, `G12(θ, φ)`, `Ud(φ₁, φ₂, φ₃)`

**Two-qutrit gates:** `CSUM`, `CSUMDag`, `CNOT3`, `CPhase`, `CPhaseDag`, `SWAP3`

## SU(3) decomposition

Any 3×3 unitary can be decomposed into native rotations:

```python
import numpy as np
from qutritium import SU3Decomposition

# Random unitary
rng = np.random.default_rng(42)
A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
Q, _ = np.linalg.qr(A)

dec = SU3Decomposition(Q, qutrit_index=0, n_qutrits=1)
reconstructed = dec.reconstruct()
fidelity = np.abs(np.trace(Q.conj().T @ reconstructed)) / 3
print(f"Fidelity: {fidelity:.10f}")  # ~1.0000000000
```
