# Circuits API Reference

## QutritCircuit

```python
from qutritium import QutritCircuit
```

### Constructor

```python
QutritCircuit(n_qutrit: int, initial_state: NDArray | None)
```

- `n_qutrit` — number of qutrits (≥ 1)
- `initial_state` — column vector of shape `(3**n_qutrit, 1)`, or `None` for $|0\cdots0\rangle$

### Methods

**`append(gate, first_qutrit, second_qutrit=None)`**

Add a `Gate` instance to the circuit. For the adjoint, pass `gate.inverse()`.

- `gate` — a `Gate` object from `qutritium.gates`
- `first_qutrit` — target qutrit index (0-based)
- `second_qutrit` — required for two-qutrit gates

**`measure_all()`**

Add a measurement to all qutrits. Can only be called once per circuit.

**`reset_circuit()`**

Remove all operations (measurement flag persists).

**`draw()`**

Print a text summary of the circuit.

### Properties

- `.n_qutrit` — number of qutrits
- `.operation_set` — list of operations
- `.measurement_flag` — whether measurement has been added

### Operators

- `len(qc)` — number of operations
- `qc1 + qc2` — concatenate circuits (left must not have measurement)
- `iter(qc)` — iterate over operations

## Instruction

Low-level representation of a gate applied to specific qutrit(s). Normally
created internally by `QutritCircuit.append()`.

```python
from qutritium.circuit import Instruction
```

### Properties

- `.effect_matrix` — full-register matrix ($3^n \times 3^n$)
- `.type` — gate name string
- `.gate` — reference to the `Gate` object (if created via `append()`)
