# Channels & Noise API Reference

Noise here is just a set of Kraus operators (a `Channel`), and a `NoiseModel` is
the bag of channels you hand to a simulator. Anything built from Kraus operators
needs the `DensityMatrixSimulator`; classical readout error is the one exception,
and works on either backend.

```python
from qutritium.channels import (
    Channel, NoiseModel, SPAMNoiseModel, ReadoutError,
    depolarizing_channel, dephasing_channel,
    amplitude_damping_channel, pauli_channel,
)
```

---

## Channel

A `Channel` holds a Completely Positive Trace-Preserving (CPTP) map as its Kraus operators $\{K_i\}$, which have to
satisfy completeness, $\sum_i K_i^\dagger K_i = I$ — the constructor checks that
for you:

$$
\mathcal{E}(\rho) = \sum_i K_i \rho K_i^\dagger.
$$

**`Channel(kraus_operators, num_qutrits=1)`**

Build one from a list of Kraus operators.

**`Channel.kraus`**

The Kraus list (you get a copy).

**`Channel.apply_kraus_op(rho)`**

Apply the map to a density matrix.

---

## Preset channels

These are the ones you'll reach for most. All single-qutrit ($d = 3$).

**`depolarizing_channel(p)`**

With probability `p`, throw the state away and replace it with the maximally
mixed state: $(1-p)\rho + p\,I/3$.

**`dephasing_channel(p)`**

Kill the off-diagonal coherences but leave the populations alone:
$(1-p)\rho + p\,\mathrm{diag}(\rho)$.

**`amplitude_damping_channel(gamma_10, gamma_21=None)`**

Relaxation down the ladder, $|1\rangle\to|0\rangle$ at rate $\gamma_{10}$ and
$|2\rangle\to|1\rangle$ at $\gamma_{21}$ (one rung per application; `gamma_21`
defaults to `gamma_10`). Apply it enough times and any state ends up in
$|0\rangle$.

**`pauli_channel(probabilities)`**

A random mix of Weyl unitaries, $\sum_g p_g\,U_g\rho U_g^\dagger$. Pass a dict
from operator name (`"identity"`, `"x_plus"`, `"z"`, `"x01"`, …) to its
probability; they have to be non-negative and add up to 1.

---

## ReadoutError

Measurement error is classical, so we model it as a confusion matrix $A$ instead
of a quantum channel: $A_{j,i}$ is the chance of reading $j$ when the true outcome
was $i$. We just apply it to the probability vector,
$p_\text{meas} = A\,p_\text{ideal}$, which is why it works on both simulators.

**`ReadoutError(confusion_matrix)`**

`A` has to be square, $3^n$-dimensional, non-negative, and column-stochastic.

**`ReadoutError.from_single_qutrit(a, n_qutrit)`**

Tensor a per-qutrit $3\times3$ matrix up to the full register.

---

## NoiseModel

The `NoiseModel` is the container you attach to a simulator with
`set_noise_model`. It carries three kinds of error:

**`add_quantum_error(channel, gate_label)`**

Add a Kraus channel after every gate with that label. Density-matrix sim only.

**`add_prep_error(channel, qutrit=None)`**

Add a Kraus channel once, before any gate. Density-matrix sim only.

**`add_readout_error(readout)`**

Add a readout confusion matrix. Applied in `run()` on both simulators.

**`NoiseModel.has_kraus_errors`**

Whether any gate or prep channel is set — that's how the `StatevectorSimulator`
knows to turn a model down.

---

## SPAMNoiseModel

If all you want is a bit of state-prep and measurement error, `SPAMNoiseModel`
builds both halves for you.

**`SPAMNoiseModel(p_prep, p_meas, n_qutrit=1)`**

Gives every qutrit a symmetric prep `pauli_channel` (weight `p_prep/2` each on
`x_plus`/`x_minus`) plus a symmetric readout matrix with off-diagonal mass
`p_meas`. It carries a Kraus prep channel, so it's density-matrix only.

---

## Examples

### Applying a channel directly

```python
import numpy as np
from qutritium.channels import amplitude_damping_channel, depolarizing_channel

rho = np.diag([0, 0, 1]).astype(complex)  # |2><2|
amplitude_damping_channel(1.0).apply_kraus_op(rho)  # -> |1><1|
depolarizing_channel(1.0).apply_kraus_op(rho)  # -> I/3
```

### Attaching a NoiseModel to a simulator

```python
from qutritium import QutritCircuit, DensityMatrixSimulator
from qutritium.channels import NoiseModel, depolarizing_channel
from qutritium.gates import X01

qc = QutritCircuit(1, None)
qc.append(X01(), first_qutrit=0)
qc.measure_all()

nm = NoiseModel()
nm.add_quantum_error(depolarizing_channel(0.3), "X01")
dm = DensityMatrixSimulator(qc)
dm.set_noise_model(nm)
print(dm.probabilities())  # [0.1, 0.8, 0.1]
```

### SPAM noise in one call

```python
from qutritium.channels import SPAMNoiseModel

dm2 = DensityMatrixSimulator(qc)  # fresh simulator -- set_noise_model must run before run()/probabilities()
dm2.set_noise_model(SPAMNoiseModel(p_prep=0.02, p_meas=0.03))
```

## Notes

- Everything here is single-qutrit for now; a model just applies its channel to
  each target qutrit one at a time.
- We use Kraus operators, not Lindblad, and the noise sits on the simulator
  rather than the circuit. So you can take one circuit and run it noiseless or
  noisy just by giving fresh simulators different models.

## References

- Gottesman, D. (1998). arXiv:quant-ph/9802007 — Weyl operators $W_{ab} = X^a Z^b$ used by `pauli_channel`
- Schwinger, J. (1960). *PNAS* 46, 570 — unitary operator bases
