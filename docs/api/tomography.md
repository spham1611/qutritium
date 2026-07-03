# Tomography API Reference

Single-qutrit state **and** process tomography with mutually unbiased bases
(MUBs), plus a couple of helpers for plotting the result.

```python
from qutritium.tomography import (
    mub_bases, state_tomography_circuits, reconstruct_state,
    process_tomography_circuits, reconstruct_process, choi_to_kraus,
    plot_density_matrix, plot_tomography_comparison,
)
```

---

## State tomography

**`mub_bases()`**

Gives you the four mutually unbiased bases for a qutrit, each as a
change-of-basis unitary. `bases[0]` is the computational basis; the other three
are the Fourier-type bases

$$
|v_b^{(k)}\rangle = \frac{1}{\sqrt{3}}\sum_{j} \omega^{\,b j^2 + k j}\,|j\rangle,
\qquad \omega = e^{2\pi i/3}.
$$

MUBs are the natural set for tomography: they're informationally complete and
spread the measurements out as evenly as possible. To measure in basis $b$ you
rotate the state by `bases[b].conj().T` and then measure in the computational
basis.

**`state_tomography_circuits(prep)`**

Takes your single-qutrit prep circuit (no measurement on it) and hands back the
four circuits to actually run — a copy of `prep`, the basis rotation
$B_b^\dagger$, then `measure_all`. Run them and keep the counts **in the order
you got them**; that's the order `reconstruct_state` expects. It raises if
`prep` has more than one qutrit or already has a measurement.

**`reconstruct_state(counts_basis, method="lls")`**

Takes those four count-dicts and inverts them back into $\rho$.

The idea: if we write $\rho$ in the Gell-Mann basis,
$\rho = I/3 + \tfrac12\sum_i s_i \lambda_i$, then every measured probability is
linear in the Bloch vector $s$, so recovering the state is just a least-squares
solve $A s = y$, with $A_{m,i} = \tfrac12\mathrm{Tr}(P_m\lambda_i)$ and
$y_m = p_m - 1/3$.

One caveat: linear least squares doesn't know $\rho$ has to be positive, so with
few shots you can get a slightly unphysical estimate (a small negative
eigenvalue). Pass `method="projected_lls"` to project that estimate onto the
closest physical density matrix — guaranteed positive and unit-trace. The
available methods are `"lls"` / `"linear_least_squares"` (raw, unprojected),
`"projected_lls"`, or its alias `"mle"` (the same projection); anything else
raises.

---

## Process tomography

Same idea as state tomography, one level up: instead of reconstructing a *state*
you reconstruct a *channel* — what a gate actually does, noise included. The
recipe prepares an informationally complete set of inputs, sends each through the
gate, and does state tomography on every output.

**`process_tomography_circuits(gate)`**

Returns `(circuits, input_states)`. It uses the 12 MUB states
$|v_b^{(k)}\rangle\langle v_b^{(k)}|$ as inputs, applies `gate` to each, and
wraps every output in the four state-tomography circuits — so you get 12 groups
of 4 circuits to run, plus the list of input density matrices the least-squares
solve needs. It raises on a multi-qutrit gate.

**`reconstruct_process(counts_per_input, input_states)`**

Turns those counts into the channel's **Choi matrix** $J$ — a $9\times 9$
Hermitian matrix that fully describes the channel. The channel is linear, so on
vectorized density matrices it is a single matrix $M$ with
$\mathrm{vec}(\mathcal{E}(\rho)) = M\,\mathrm{vec}(\rho)$. Each known input
$\rho_\text{in}$ and its tomographically reconstructed output $\rho_\text{out}$
give one column equation; stacking all 12 and solving by least squares recovers
$M$, and the Choi matrix follows from its definition,

$$
J = \sum_{i,j} \mathcal{E}\!\left(|i\rangle\langle j|\right)\otimes
|i\rangle\langle j|.
$$

It needs at least 9 inputs (the MUB set gives 12) and raises if the two lists
disagree in length.

**`choi_to_kraus(choi, atol=1e-8)`**

Eigendecomposes $J = \sum_l \lambda_l |u_l\rangle\langle u_l|$ and returns the
Kraus operators $K_l = \sqrt{\lambda_l}\,\mathrm{mat}(u_l)$, dropping
eigenvalues at or below `atol`. A unitary gate gives back a single Kraus
operator; a noisy one gives several. It raises if `choi` is not $9\times 9$ or
not Hermitian.

---

## Plotting

These need matplotlib (`pip install qutritium[plot]`), and each hands back a
`matplotlib.figure.Figure` so you can save or tweak it.

**`plot_density_matrix(rho, style="city", title=...)`**

$\mathrm{Re}(\rho)$ and $\mathrm{Im}(\rho)$ as 3D bars (`"city"`) or a Hinton
diagram (`"hinton"`).

**`plot_tomography_comparison(rho_ideal, rho_estimated, fidelity=None)`**

The ideal and reconstructed states side by side, which is usually the figure
you actually want.

---

## Examples

### State tomography

```python
from qutritium import QutritCircuit, StatevectorSimulator, DensityMatrixSimulator, state_fidelity
from qutritium.gates import H3
from qutritium.tomography import state_tomography_circuits, reconstruct_state

prep = QutritCircuit(1, None)
prep.append(H3(), first_qutrit=0)

counts = []
for circ in state_tomography_circuits(prep):
    sim = StatevectorSimulator(circ)
    sim.run(num_shots=20_000)
    counts.append(sim.get_counts())            # keep the basis order

rho_est = reconstruct_state(counts)
rho_true = DensityMatrixSimulator(prep).return_final_state()
print(state_fidelity(rho_true, rho_est))       # > 0.95
```

### Process tomography

```python
import numpy as np
from qutritium import DensityMatrixSimulator
from qutritium.channels import NoiseModel, depolarizing_channel
from qutritium.gates import X01
from qutritium.tomography import (
    process_tomography_circuits, reconstruct_process, choi_to_kraus,
)

nm = NoiseModel()
nm.add_quantum_error(depolarizing_channel(0.2), "X01")  # X01 with 20% depolarizing

groups, inputs = process_tomography_circuits(X01())
counts = []
for group in groups:
    per_basis = []
    for circ in group:
        dm = DensityMatrixSimulator(circ)
        dm.set_noise_model(nm)
        probs = dm.probabilities()
        per_basis.append({str(k): round(p * 100_000) for k, p in enumerate(probs)})
    counts.append(per_basis)

choi = reconstruct_process(counts, inputs)  # (9, 9) Choi matrix
kraus = choi_to_kraus(choi, atol=1e-6)  # Kraus operators
print(len(kraus), "Kraus operators")
total = sum(k.conj().T @ k for k in kraus)
print("trace preserving:", np.allclose(total, np.eye(3), atol=1e-2))
```

### Plotting

```python
from qutritium.tomography import plot_tomography_comparison

fig = plot_tomography_comparison(rho_true, rho_est, fidelity=state_fidelity(rho_true, rho_est))
fig.savefig("state_tomography.png")
```

## References

- Ivanović (1981), *J. Phys. A* **14**, 3241; Wootters & Fields (1989),
  *Ann. Phys.* **191**, 363 — the MUB construction and its optimality.
- Thew, Nemoto, White & Munro (2002), *Phys. Rev. A* **66**, 012303 — the qudit
  least-squares reconstruction recipe used here.
- Smolin, Gambetta & Smith (2012), *Phys. Rev. Lett.* **108**, 070502 — the
  projection onto the closest physical state used by `"projected_lls"`.
- Choi (1975), *Linear Algebra Appl.* **10**, 285; Jamiołkowski (1972), *Rep.
  Math. Phys.* **3**, 275 — the Choi matrix / channel-state duality behind
  process tomography.
