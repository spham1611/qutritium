# Simulator API Reference

Two simulators share a common base (`Simulator`):

- **`QASMSimulator`** — exact statevector evolution. Fast; pure states only.
- **`DensityMatrixSimulator`** — density-matrix evolution
  $\rho \to U\rho U^\dagger$. Heavier (memory $\sim 9^{n}$) but represents
  mixed states — the backend for noise (v1.4) and tomography.

```python
from qutritium import QASMSimulator, DensityMatrixSimulator
```

Both consume a `QutritCircuit` and expose the same measurement API
(`run`, `get_counts`, `result`, `probabilities`, `plot`).

---

## Shared API (`Simulator` base)

**`run(num_shots: int = 1024)`**

Evolve the circuit and sample `num_shots` computational-basis outcomes.
Requires a measurement (`measure_all()`); raises `RuntimeError` otherwise and
`ValueError` for non-positive `num_shots`.

**`probabilities() → NDArray`**

Born-rule probability vector over the $3^n$ computational basis states
(runs the simulation if needed). Inspect the distribution without sampling.

**`get_counts() → dict[str, int]`**

Histogram of sampled outcomes (base-3 ket labels, qutrit 0 leftmost). Call
`run()` first.

**`result() → list[str]`**

Raw ordered list of sampled outcomes.

**`plot(plot_type="histogram") → Figure`**

Plot the counts. Types: `"histogram"`, `"line"`, `"dot"`. Requires
matplotlib (`pip install qutritium[plot]`).

---

## QASMSimulator

Statevector backend. In addition to the shared API:

**`return_final_state() → NDArray`**

Final statevector, shape $(3^n, 1)$ (runs simulation if needed; no
measurement required).

**`density_matrix() → NDArray`**

Pure-state density matrix $|\psi\rangle\langle\psi|$.

**`add_SPAM_noise(p_prep, p_meas, error_type="Pauli_error")`**

State-preparation and measurement Pauli noise. *(Will be superseded by the
v1.4 `NoiseModel` framework.)*

```python
from qutritium import QutritCircuit, QASMSimulator
from qutritium.gates import H3

qc = QutritCircuit(1, None)
qc.append(H3(), first_qutrit=0)
qc.measure_all()

sim = QASMSimulator(qc)
sim.run(num_shots=3000)
print(sim.get_counts())        # {'0': ~1000, '1': ~1000, '2': ~1000}
```

---

## DensityMatrixSimulator

Density-matrix backend. Evolves $\rho \to U\rho U^\dagger$ for each gate.
Use for mixed states or when you need expectation values / reduced states.

**`return_final_state() → NDArray`**

Final density matrix, shape $(3^n, 3^n)$.

**`expectation_value(observable) → float`**

$\langle O \rangle = \mathrm{tr}(\rho O)$ for a Hermitian observable of shape
$(3^n, 3^n)$. Validates shape and Hermiticity.

**`partial_trace(keep_indices) → NDArray`**

Reduced density matrix on the qutrits in `keep_indices`, tracing out the rest:

$$
\rho_A = \mathrm{tr}_B(\rho).
$$

Output is indexed in ascending qutrit order, shape $(3^k, 3^k)$ with
$k = \texttt{len(keep\_indices)}$. Raises `ValueError` for empty, duplicate,
or out-of-range indices.

```python
import numpy as np
from qutritium import QutritCircuit, DensityMatrixSimulator
from qutritium.gates import H3, CSUM

# Maximally entangled qutrit pair
qc = QutritCircuit(2, None)
qc.append(H3(), first_qutrit=0)
qc.append(CSUM(), first_qutrit=0, second_qutrit=1)

sim = DensityMatrixSimulator(qc)
rho = sim.return_final_state()                # 9x9, trace 1

# Tracing out qutrit 1 leaves the maximally mixed state -> entanglement
reduced = sim.partial_trace([0])
print(np.allclose(reduced, np.eye(3) / 3))    # True

# Expectation value of a diagonal observable
obs = np.diag([1, 0, -1] * 3).astype(complex)
print(sim.expectation_value(obs))
```

### Cross-checking the two backends

A noiseless circuit gives the same measurement statistics on both simulators,
and `state_fidelity` between the two final states is 1:

```python
from qutritium import state_fidelity
sv, dm = QASMSimulator(qc), DensityMatrixSimulator(qc)
print(state_fidelity(dm.return_final_state(), sv.density_matrix()))  # ≈ 1
```

## Notes

- Statevector memory scales as $3^n$; density-matrix as $9^n$. Practical
  limits are roughly $n \le 10$ (statevector) and $n \le 6$ (density matrix).
- Noise channels and readout error arrive in v1.4 via a `NoiseModel` set on
  the density-matrix simulator.
