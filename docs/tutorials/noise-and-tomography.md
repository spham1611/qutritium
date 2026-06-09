# Noise & State Tomography

We'll take a circuit, run it with some noise on the density-matrix simulator, and
then turn around and reconstruct a prepared state from measurement counts.

## 1. Noise lives on the simulator

The circuit stays pure-unitary — noise goes on the *simulator*. You build a
`NoiseModel` and attach it with `set_noise_model`. `add_quantum_error` says
"apply this channel after every gate with this label," so here a 10% depolarizing
channel follows each `X01`:

```python
from qutritium import QutritCircuit, DensityMatrixSimulator
from qutritium.channels import NoiseModel, depolarizing_channel
from qutritium.gates import X01

qc = QutritCircuit(1, None)
qc.append(X01(), first_qutrit=0)          # |0> -> |1>
qc.measure_all()

nm = NoiseModel()
nm.add_quantum_error(depolarizing_channel(0.1), "X01")

dm = DensityMatrixSimulator(qc)
dm.set_noise_model(nm)
dm.run(num_shots=5000)
print(dm.get_counts())          # mostly '1', with a little leaking into '0' and '2'
```

Three things to remember:

- Kraus channels (gate and prep errors) only run on the `DensityMatrixSimulator`.
  Hand one to the statevector `QASMSimulator` and it'll refuse with a
  `NotImplementedError`.
- Attach the model *before* you call `run()`. The simulation gets cached, so a
  model added afterward raises — build a fresh simulator if you need to swap it.
- Readout error is classical, so it works on both simulators.

Want prep + measurement error together without thinking about it? That's
`SPAMNoiseModel`:

```python
from qutritium.channels import SPAMNoiseModel
dm2 = DensityMatrixSimulator(qc)
dm2.set_noise_model(SPAMNoiseModel(p_prep=0.02, p_meas=0.03))
```

## 2. State tomography

Say you've prepared some state and want to know what it actually is. Measure it in
all four MUBs and invert. Start from a *prep* circuit with no measurement:

```python
from qutritium import QASMSimulator, state_fidelity
from qutritium.gates import H3
from qutritium.tomography import state_tomography_circuits, reconstruct_state

prep = QutritCircuit(1, None)
prep.append(H3(), first_qutrit=0)         # the state we want to reconstruct

counts = []
for circ in state_tomography_circuits(prep):
    sim = QASMSimulator(circ)
    sim.run(num_shots=20_000)
    counts.append(sim.get_counts())       # keep them in order!

rho_est = reconstruct_state(counts)
rho_true = DensityMatrixSimulator(prep).return_final_state()
print(state_fidelity(rho_true, rho_est))  # > 0.95
```

`state_tomography_circuits` hands the circuits back in `mub_bases` order, and
`reconstruct_state` wants the counts in that same order — so just don't shuffle
them.

## 3. Look at it

```python
from qutritium.tomography import plot_tomography_comparison
fig = plot_tomography_comparison(
    rho_true, rho_est, fidelity=state_fidelity(rho_true, rho_est),
)
fig.savefig("tomography.png")
```

You get the ideal and reconstructed density matrices side by side as 3D bars.
More shots, better reconstruction; with only a few, linear inversion can hand back
a slightly unphysical $\rho$ (a small negative eigenvalue) — MLE will tighten that
up in v1.5.
