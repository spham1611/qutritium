# Metrics API Reference

State- and process-level comparison functions. All are plain functions —
import from `qutritium` (or `qutritium.metrics`):

```python
from qutritium import (
    state_fidelity, trace_distance, purity, von_neumann_entropy,
    process_fidelity, average_gate_fidelity,
)
```

State functions accept either a density matrix of shape `(d, d)` or a
pure-state ket of shape `(d,)` / `(d, 1)` — kets are promoted to
$|\psi\rangle\langle\psi|$ internally.

---

## State metrics

**`state_fidelity(rho, sigma) → float`**

$$
F(\rho, \sigma) = \left(\mathrm{tr}\sqrt{\sqrt{\rho}\,\sigma\,\sqrt{\rho}}\right)^2.
$$

Reduces to $|\langle\psi|\phi\rangle|^2$ for pure states. Symmetric, in
$[0, 1]$; equals 1 iff $\rho = \sigma$.

**`trace_distance(rho, sigma) → float`**

$$
T(\rho, \sigma) = \tfrac{1}{2}\,\lVert \rho - \sigma \rVert_1.
$$

In $[0, 1]$; 0 iff $\rho = \sigma$, 1 iff their supports are orthogonal.

**`purity(rho) → float`**

$$
P(\rho) = \mathrm{tr}(\rho^2).
$$

In $[1/d, 1]$; equals 1 for pure states and $1/d$ for the maximally mixed
state $I/d$. Useful as a quick "how mixed is this?" check (e.g. after a
noise channel).

**`von_neumann_entropy(rho, base=2.0) → float`**

$$
S(\rho) = -\,\mathrm{tr}(\rho \log \rho).
$$

`base=2` gives bits (default), `base=np.e` gives nats. 0 for pure states,
$\log_{\text{base}} d$ for the maximally mixed state. Numerical-zero
eigenvalues are dropped before the log.

---

## Process metrics

Both take two unitaries of shape `(d, d)`; inputs are validated to be unitary
(`atol=1e-8`).

**`process_fidelity(u_ideal, u_actual) → float`**

$$
F_{\text{pro}} = \frac{|\mathrm{tr}(U_{\text{ideal}}^\dagger U_{\text{actual}})|^2}{d^2}.
$$

In $[0, 1]$; equals 1 iff the two unitaries are equal up to a global phase.

**`average_gate_fidelity(u_ideal, u_actual) → float`**

$$
F_{\text{avg}} = \frac{d\,F_{\text{pro}} + 1}{d + 1}.
$$

The Haar-average of `state_fidelity` between $U_{\text{ideal}}|\psi\rangle$
and $U_{\text{actual}}|\psi\rangle$ over input states — the figure of merit
quoted in experimental gate-characterization work.

---

## Examples

```python
import numpy as np
from qutritium.gates import X01
from qutritium import (
    state_fidelity, trace_distance, purity, von_neumann_entropy,
    process_fidelity, average_gate_fidelity,
)

psi = np.array([1, 0, 0], dtype=complex)
mixed = np.eye(3) / 3

state_fidelity(psi, psi)      # 1.0
state_fidelity(psi, mixed)    # 1/3  (pure vs maximally mixed)
trace_distance(psi, mixed)    # 2/3
purity(mixed)                 # 1/3
von_neumann_entropy(mixed)    # log2(3) ≈ 1.585

u = X01().matrix()
process_fidelity(u, u)                            # 1.0
average_gate_fidelity(np.eye(3, dtype=complex), u)  # 1/3
```

## Notes

- Fidelities return real floats; tiny imaginary parts from numerical noise
  are discarded.
- For pure (rank-1) inputs, `state_fidelity` loses a few digits of precision
  through its two matrix square roots — compare with a tolerance like
  `1e-6`, not `1e-12`.
- Channel-vs-channel (Choi-based) process fidelity is not in this release; the
  process metrics take unitaries only.
