# Tomography

Single-qutrit state tomography with mutually unbiased bases (MUBs), plus a couple
of helpers for plotting the result.

```python
from qutritium.tomography import (
    mub_bases, state_tomography_circuits, reconstruct_state,
    plot_density_matrix, plot_tomography_comparison,
)
```

---

## mub_bases

**`mub_bases()`** gives you the four mutually unbiased bases for a qutrit, each as
a change-of-basis unitary. `bases[0]` is the computational basis; the other three
are the Fourier-type bases

$$
|v_b^{(k)}\rangle = \frac{1}{\sqrt{3}}\sum_{j} \omega^{\,b j^2 + k j}\,|j\rangle,
\qquad \omega = e^{2\pi i/3}.
$$

MUBs are the natural set for tomography: they're informationally complete and
spread the measurements out as evenly as possible (Wootters & Fields, 1989). To
measure in basis $b$ you rotate the state by `bases[b].conj().T` and then measure
in the computational basis.

---

## state_tomography_circuits

**`state_tomography_circuits(prep)`** takes your single-qutrit prep circuit (no
measurement on it) and hands back the four circuits to actually run — a copy of
`prep`, the basis rotation $B_b^\dagger$, then `measure_all`. Run them and keep
the counts **in the order you got them**; that's the order `reconstruct_state`
expects. It raises if `prep` has more than one qutrit or already has a
measurement.

---

## reconstruct_state

**`reconstruct_state(counts_basis, method="lls")`** takes those four count-dicts
and inverts them back into $\rho$.

The idea: if we write $\rho$ in the Gell-Mann basis,
$\rho = I/3 + \tfrac12\sum_i s_i \lambda_i$, then every measured probability is
linear in the Bloch vector $s$, so recovering the state is just a least-squares
solve $A s = y$, with $A_{m,i} = \tfrac12\mathrm{Tr}(P_m\lambda_i)$ and
$y_m = p_m - 1/3$ (Thew et al., 2002).

One caveat: linear inversion doesn't know $\rho$ has to be positive, so with few
shots you can get a slightly unphysical estimate (a small negative eigenvalue).
That's expected — the constrained (MLE) version is on the v1.5 list. Only `"lls"`
/ `"linear_least_squares"` is wired up for now; anything else raises.

```python
from qutritium import QutritCircuit, QASMSimulator, DensityMatrixSimulator, state_fidelity
from qutritium.gates import H3
from qutritium.tomography import state_tomography_circuits, reconstruct_state

prep = QutritCircuit(1, None)
prep.append(H3(), first_qutrit=0)

counts = []
for circ in state_tomography_circuits(prep):
    sim = QASMSimulator(circ)
    sim.run(num_shots=20_000)
    counts.append(sim.get_counts())            # keep the basis order

rho_est = reconstruct_state(counts)
rho_true = DensityMatrixSimulator(prep).return_final_state()
print(state_fidelity(rho_true, rho_est))       # > 0.95
```

---

## Plotting

These need matplotlib (`pip install qutritium[plot]`), and each hands back a
`matplotlib.figure.Figure` so you can save or tweak it.

- **`plot_density_matrix(rho, style="city", title=...)`** — $\mathrm{Re}(\rho)$
  and $\mathrm{Im}(\rho)$ as 3D bars (`"city"`) or a Hinton diagram (`"hinton"`).
- **`plot_tomography_comparison(rho_ideal, rho_estimated, fidelity=None)`** — the
  ideal and reconstructed states side by side, which is usually the figure you
  actually want.

```python
from qutritium.tomography import plot_tomography_comparison
fig = plot_tomography_comparison(rho_true, rho_est, fidelity=state_fidelity(rho_true, rho_est))
fig.savefig("tomography.png")
```

## References

- Ivanović (1981), *J. Phys. A* **14**, 3241; Wootters & Fields (1989),
  *Ann. Phys.* **191**, 363 — where the MUB construction and its optimality come
  from.
- Thew, Nemoto, White & Munro (2002), *Phys. Rev. A* **66**, 012303 — the qudit
  linear-inversion recipe used here.
