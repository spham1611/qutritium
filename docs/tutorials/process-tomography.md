# Tutorial: Process Tomography

State tomography asks "what state did I prepare?" — process tomography asks
"what does my gate actually *do*, noise included?" We'll characterize a noisy
qutrit Hadamard and read its Kraus operators back out of the data.

See also [
`examples/process_tomography.ipynb`](https://github.com/spham1611/qutritium/blob/main/examples/process_tomography.ipynb)
for the full interactive notebook.

## 1. The recipe

A channel is fully determined by what it does to an informationally complete
set of inputs. So: prepare the 12 MUB states, push each one through the gate,
and run state tomography on every output. `process_tomography_circuits` builds
all of that for you — 12 groups of 4 measurement circuits, plus the list of
input density matrices the reconstruction needs:

```python
from qutritium import DensityMatrixSimulator
from qutritium.channels import NoiseModel, depolarizing_channel
from qutritium.gates import H3
from qutritium.tomography import (
    process_tomography_circuits, reconstruct_process, choi_to_kraus,
)

nm = NoiseModel()
nm.add_quantum_error(depolarizing_channel(0.15), "H3")   # the noise we'll try to detect

groups, inputs = process_tomography_circuits(H3())
```

## 2. Run the circuits

Run every circuit and keep the counts grouped and ordered — 12 groups of 4,
matching the order `process_tomography_circuits` gave you:

```python
counts = []
for group in groups:
    per_basis = []
    for circ in group:
        dm = DensityMatrixSimulator(circ)
        dm.set_noise_model(nm)
        dm.run(num_shots=4000)
        per_basis.append(dm.get_counts())
    counts.append(per_basis)
```

## 3. Reconstruct the channel

`reconstruct_process` inverts the counts into the channel's $9 \times 9$ Choi
matrix, and `choi_to_kraus` reads Kraus operators off its spectrum:

```python
import numpy as np

choi = reconstruct_process(counts, inputs)
kraus = choi_to_kraus(choi, atol=1e-3)
print(len(kraus))                     # 9

weights = sorted((float(np.trace(k.conj().T @ k).real) for k in kraus), reverse=True)
print([round(w, 2) for w in weights]) # [2.61, 0.1, 0.07, ...] — one dominant + eight small
```

That spectrum is the noise made visible: a perfect unitary gives back a
*single* Kraus operator with weight 3, while our depolarized `H3` shows one
dominant operator (weight $3(1 - 8p/9) = 2.6$ for $p = 0.15$) plus eight small
depolarizing operators of weight $\approx p/3$.

## 4. Sanity checks

The true channel is trace-preserving, so the recovered Kraus set should
satisfy $\sum_\ell K_\ell^\dagger K_\ell \approx I$ up to shot noise (linear
least squares doesn't enforce it exactly), and the Choi trace should be
$d = 3$:

```python
total = sum(k.conj().T @ k for k in kraus)
print(np.allclose(total, np.eye(3), atol=0.05))   # True
print(np.trace(choi).real)                        # ~3.0
```

More shots tighten everything; with too few, small Choi eigenvalues go
slightly negative and `choi_to_kraus` drops them (that's what `atol` is for).
