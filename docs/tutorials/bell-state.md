# Tutorial: Qutrit Bell State

This tutorial walks through building and simulating a qutrit Bell state — the
maximally entangled state $\frac{1}{\sqrt{3}}(|00\rangle + |11\rangle + |22\rangle)$.

See also [`examples/tutorial.ipynb`](https://github.com/spham1611/qutritium/blob/main/examples/tutorial.ipynb)
for the full interactive notebook.

## 1. Build the circuit

The qutrit Bell state is prepared by applying a Hadamard (H3) to the first
qutrit, then entangling with a CSUM gate:

```python
from qutritium import QutritCircuit
from qutritium.gates import H3, CSUM

qc = QutritCircuit(n_qutrit=2, initial_state=None)
qc.append(H3(), first_qutrit=0)
qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
```

## 2. Inspect the statevector

```python
from qutritium import StatevectorSimulator

sim = StatevectorSimulator(qc)
state = sim.return_final_state()

from qutritium.circuit.utils import print_statevector
print_statevector(state, n_qutrit=2)
```

Expected output — three equal-amplitude components:

```
State:
(0.577+0j) |00>
(0.577+0.j) |11>
(0.577+0.j) |22>
```

## 3. Simulate measurements

```python
qc.measure_all()
sim = StatevectorSimulator(qc)
sim.run(num_shots=3000)
counts = sim.get_counts()
print(counts)
# {'00': ~1000, '11': ~1000, '22': ~1000}
```

## 4. Verify entanglement via density matrix

```python
import numpy as np

sim2 = StatevectorSimulator(QutritCircuit(2, None))
# Rebuild without measurement for density matrix
qc2 = QutritCircuit(2, None)
qc2.append(H3(), first_qutrit=0)
qc2.append(CSUM(), first_qutrit=0, second_qutrit=1)
sim2 = StatevectorSimulator(qc2)
rho = sim2.density_matrix()

# The partial trace over qutrit B should give a maximally mixed state
rho_A = np.zeros((3, 3), dtype=complex)
for b in range(3):
    proj_b = np.zeros((3, 1), dtype=complex)
    proj_b[b, 0] = 1.0
    bra_b = np.kron(np.eye(3), proj_b.T)
    ket_b = np.kron(np.eye(3), proj_b)
    rho_A += bra_b @ rho @ ket_b

print("Reduced density matrix (should be I/3):")
print(np.round(rho_A, 4))
```

## 5. U(3) decomposition

Decompose the Hadamard into native rotations:

```python
from qutritium import SU3Decomposition
from qutritium.gates import H3

dec = SU3Decomposition(H3().matrix(), qutrit_index=0, n_qutrits=1)
fidelity = np.abs(np.trace(H3().matrix().conj().T @ dec.reconstruct())) / 3
print(f"Decomposition fidelity: {fidelity:.10f}")
```
