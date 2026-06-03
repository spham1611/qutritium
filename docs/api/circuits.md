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

Raises `RuntimeError` if called after `measure_all()` — the measurement must
remain the final operation, so build all gates first.

**`measure_all()`**

Add a measurement to all qutrits. Can only be called once per circuit (raises
`RuntimeError` on a second call).

**`reset_circuit()`**

Clear all operations **and** the measurement, returning a clean slate — the
circuit can be rebuilt and re-measured afterward (same `n_qutrit` and
`initial_state`).

**`gate_count() → int`**

Number of gate operations, excluding the measurement.

**`depth(filter_function=lambda _: True) → int`**

Circuit depth — the longest path through the per-qutrit timeline. Gates on
disjoint qutrits run in parallel (depth 1); a two-qutrit gate occupies both
its qutrits in one step. `filter_function` selects which instructions count
(e.g. `lambda ins: ins.second_qutrit is not None` for two-qutrit depth).

**`to_matrix() → NDArray`**

Collapse the whole circuit to a single $3^n \times 3^n$ unitary (gates
multiplied in time order, most recent on the left). Raises `RuntimeError` if
the circuit contains a measurement.

**`draw() → str`**

Return a text diagram of the circuit as a string (does **not** print —
wrap in `print(qc.draw())` to display).

### Properties

- `.n_qutrit` — number of qutrits
- `.operation_set` — list of operations (setter validates the single-trailing-measurement invariant)
- `.measurement_flag` — whether a measurement has been added

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
