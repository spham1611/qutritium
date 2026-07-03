# Decomposition API Reference

`SU3Decomposition` factors any single-qutrit unitary into a fixed sequence of
**native two-level rotations** plus a diagonal phase — the qutrit analogue of
the qubit ZYZ Euler decomposition.

```python
from qutritium import SU3Decomposition
```

---

## Overview

This is what lets you take an arbitrary $3\times3$ unitary and run it on
hardware whose native operations are subspace rotations (e.g. trapped-ion
$g_{01}$ / $g_{12}$ drives) plus virtual-Z phases.

The factorization (Vitanov, *Phys. Rev. A* **85**, 032331, 2012) is:

$$
U = U_d(\phi_6, \phi_5, \phi_4)\; r_{01}(\phi_3, \theta_3)\; r_{12}(\phi_2, \theta_2)\; r_{01}(\phi_1, \theta_1)
$$

Read right to left (first operation applied last in matrix order): a rotation in
the $\{|0\rangle, |1\rangle\}$ subspace, then $\{|1\rangle, |2\rangle\}$, then
$\{|0\rangle, |1\rangle\}$ again, finished by a diagonal phase.

### The building blocks

A **composite two-level rotation** in subspace $\{i, j\}$ is

$$
r_{ij}(\phi, \theta) = R^{z}_{ij}(\phi)\, R^{x}_{ij}(\theta)\, R^{z}_{ij}(-\phi),
$$

i.e. an $x$-rotation by polar angle $\theta$ sandwiched between $\pm\phi$
$z$-rotations that set the azimuthal axis. The **diagonal phase** is

$$
U_d(\phi_6, \phi_5, \phi_4) = \mathrm{diag}\!\left(e^{i\phi_6},\, e^{i\phi_5},\, e^{i\phi_4}\right).
$$

### The nine angles

| Symbol | Role |
|--------|------|
| $\theta_1, \theta_2, \theta_3$ | Polar (rotation) angles of the three two-level rotations |
| $\phi_1, \phi_2, \phi_3$ | Azimuthal phases of those rotations |
| $\phi_4, \phi_5, \phi_6$ | Diagonal phases on $\|2\rangle, \|1\rangle, \|0\rangle$ |

Eight of these are independent (matching the 8 real parameters of $SU(3)$);
the ninth carries the global phase, so the decomposition reproduces any
$U \in U(3)$.

---

## SU3Decomposition

### Constructor

```python
SU3Decomposition(su3: NDArray, qutrit_index: int, n_qutrits: int)
```

- `su3` — the $3\times3$ unitary to decompose. Validated to be unitary
  ($U U^\dagger = I$) within `atol=1e-8`; raises `ValueError` otherwise.
- `qutrit_index` — which qutrit in the register this unitary acts on (used by
  `to_native` / `to_circuit`).
- `n_qutrits` — total qutrit count in the target register.

The nine angles are extracted at construction and cached on `.angles`.

### Methods

**`reconstruct() → NDArray`**

Multiply the four decomposed factors back together. Equals `su3` up to
floating-point error — useful as a self-check:

$$
\texttt{reconstruct()} = U_d\, r_{01}(\phi_3,\theta_3)\, r_{12}(\phi_2,\theta_2)\, r_{01}(\phi_1,\theta_1) \approx U.
$$

**`to_native() → NativeDecomposition`**

Return a `(phases, instructions)` named tuple for a hardware-native gate set:

- `phases` — a length-2 array `[phase01, phase12]` of **virtual-Z** angles,
  derived from the diagonal factor:

  $$
  \phi_{01} = \phi_6 - \phi_5, \qquad \phi_{12} = \phi_5 - \phi_4.
  $$

  On hardware these are applied "for free" by shifting the phase reference of
  subsequent drives — zero gate duration.
- `instructions` — three `Instruction` objects in order `g01, g12, g01`,
  carrying the polar/azimuthal angle pairs.

**`to_circuit() → QutritCircuit`**

Same factor sequence as `to_native`, but emitted as `G01`, `G12`, `Ud` gate
objects appended to a fresh `QutritCircuit` you can simulate directly.

### Properties

- `.angles` — `DecompositionAngles` named tuple (`theta1, theta2, theta3,
  phi1, …, phi6`).
- `.su3` — the input unitary.

### Helper named tuples

- `DecompositionAngles` — the nine extracted angles.
- `NativeDecomposition` — `(phases, instructions)` returned by `to_native`.

---

## Examples

```python
import numpy as np
from qutritium import SU3Decomposition
from qutritium.gates import H3

# Haar-ish random unitary via QR
rng = np.random.default_rng(42)
A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
Q, _ = np.linalg.qr(A)

dec = SU3Decomposition(Q, qutrit_index=0, n_qutrits=1)
fid = np.abs(np.trace(Q.conj().T @ dec.reconstruct())) / 3
print(f"Fidelity: {fid:.10f}")   # 1.0000000000

# Recover a known gate's angles
dec2 = SU3Decomposition(H3().matrix(), qutrit_index=0, n_qutrits=1)
print(dec2.angles)                                     # the nine angles that build H3
print(np.allclose(dec2.reconstruct(), H3().matrix()))  # True

# Compile to a native circuit
qc = dec2.to_circuit()
print(qc)                  # QutritCircuit(n_qutrit=1, ops=4)  -> G01, G12, G01, Ud

native = dec2.to_native()
print(native.phases)       # [phase01, phase12]  virtual-Z angles
```

## Notes

- The decomposition is **exact**, not approximate — `reconstruct()` matches the
  input to machine precision for any valid unitary.
- Angle extraction branches on the magnitude of $U_{22}$ to stay numerically
  stable at the poles (where some angles become degenerate). You don't need to
  handle these cases — they're internal.
- Two-qutrit ($SU(9)$) decomposition is **not** in this release; `su3` must be
  $3\times3$.
