# Simulator API Reference

## QASMSimulator

```python
from qutritium import QASMSimulator
```

### Constructor

```python
QASMSimulator(qc: QutritCircuit)
```

### Methods

**`run(num_shots: int = 1024)`**

Simulate the circuit and sample `num_shots` measurement outcomes. Requires the
circuit to have a measurement (`measure_all()`).

**`get_counts() → dict[str, int]`**

Return a histogram of measurement outcomes. Must call `run()` first.

**`return_final_state() → NDArray`**

Return the final statevector (runs simulation if needed). Does not require measurement.

**`density_matrix() → NDArray`**

Return the pure-state density matrix $|\psi\rangle\langle\psi|$.

**`result() → list[str]`**

Return the raw list of sampled outcomes.

**`plot(plot_type: str = "histogram") → Figure`**

Plot measurement counts. Types: `"histogram"`, `"line"`, `"dot"`.
Requires `matplotlib` (`pip install qutritium[plot]`).

**`add_SPAM_noise(p_prep: float, p_meas: float, error_type: str = "Pauli_error")`**

Add state-preparation and measurement (SPAM) noise.

### Example

```python
from qutritium import QutritCircuit, QASMSimulator
from qutritium.gates import H3

qc = QutritCircuit(1, None)
qc.append(H3(), first_qutrit=0)
qc.measure_all()

sim = QASMSimulator(qc)
sim.run(num_shots=3000)

print(sim.get_counts())   # {'0': ~1000, '1': ~1000, '2': ~1000}
print(sim.density_matrix())  # 3×3 density matrix
```
