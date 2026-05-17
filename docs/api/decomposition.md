# Decomposition API Reference

## SU3Decomposition

Decompose any 3×3 unitary into a canonical product of native qutrit rotations:

$$U = U_d(\phi_6, \phi_5, \phi_4) \cdot r_{01}(\phi_3, \theta_3) \cdot r_{12}(\phi_2, \theta_2) \cdot r_{01}(\phi_1, \theta_1)$$

This is the qutrit analogue of the qubit ZYZ Euler decomposition. Based on
Vitanov, *Phys. Rev. A* 85, 032331 (2012).

```python
from qutritium import SU3Decomposition
```

### Constructor

```python
SU3Decomposition(su3: NDArray, qutrit_index: int, n_qutrits: int)
```

- `su3` — 3×3 unitary matrix
- `qutrit_index` — which qutrit this decomposition targets
- `n_qutrits` — total qutrit count in the register

### Methods

**`reconstruct() → NDArray`**

Reconstruct the unitary from the decomposed factors. Should match `su3` to
numerical precision.

**`to_native() → NativeDecomposition`**

Return virtual-Z phases and a list of native `g01`/`g12` `Instruction` objects.

**`to_circuit() → QutritCircuit`**

Build a `QutritCircuit` from the decomposition using `G01`, `G12`, `Ud` gates.

### Properties

- `.angles` — `DecompositionAngles` named tuple with all 9 angles
- `.su3` — the input unitary

### Example

```python
import numpy as np
from qutritium import SU3Decomposition

# Generate a random unitary
rng = np.random.default_rng(42)
A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
Q, _ = np.linalg.qr(A)

# Decompose
dec = SU3Decomposition(Q, qutrit_index=0, n_qutrits=1)

# Verify fidelity
fidelity = np.abs(np.trace(Q.conj().T @ dec.reconstruct())) / 3
print(f"Fidelity: {fidelity:.10f}")  # 1.0000000000

# Get a circuit
qc = dec.to_circuit()
print(qc)  # QutritCircuit(n_qutrit=1, ops=4)
```
