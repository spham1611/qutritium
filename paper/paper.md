---
title: 'Qutritium: A hardware-agnostic Python library for qutrit quantum computing'
tags:
  - Python
  - quantum computing
  - qutrit
  - qudit
  - quantum simulation
  - U(3) decomposition
authors:
  - name: Son Pham
    orcid: 0009-0007-8139-6797
    corresponding: true
    affiliation: 1
affiliations:
  - name: Duke University, USA
    index: 1
date: 3 July 2026
bibliography: paper.bib
---

# Summary

`qutritium` is a hardware-agnostic Python library for qutrit (three-level, $d = 3$)
quantum computing: it works at the gate level and does not depend on any specific
hardware platform or its pulse-level details. It provides qutrit gates: rotation and generalized rotation gates, the
qutrit Hadamard gate, and other single- and
two-qutrit gates such as controlled-SUM (CSUM). The package provides a circuit container and two simulators: a
statevector simulator and a density-matrix
simulator for mixed states.

Beyond these, `qutritium` decomposes arbitrary single-qutrit unitaries
(elements of U(3)) into native subspace rotations,
exact up to global phase, based on the SU(3) synthesis of @vitanov2012.
It also supports standard quantum information tools: state- and process-level metrics (state and process fidelity,
average gate
fidelity, trace distance, purity, and von Neumann entropy) and Kraus-operator noise channels
(depolarizing, dephasing, amplitude damping, and Pauli), together with a simulator
noise model and a classical readout-error model.

For state tomography, the density matrix is reconstructed by linear least
squares [@thew2002], with an optional maximum-likelihood projection onto the
closest physical state [@smolin2012]. For process tomography, the channel's
Choi matrix is reconstructed by linear least squares and made Hermitian; it is
not projected onto the completely positive, trace-preserving (CPTP) set. Kraus
operators from the positive part of its spectrum give a completely positive
(but not necessarily trace-preserving) channel estimate.

![**Architecture of the `qutritium` package.** Gates are composed into a
`QutritCircuit` or produced by `SU3Decomposition`. The circuit is then executed
by the statevector simulator (`StatevectorSimulator`) or the density-matrix
simulator (`DensityMatrixSimulator`). Depending on whether the result is a
statevector or a density matrix, it is fed to the tomography and metrics
layers.\label{fig:workflow}](figures/workflow.pdf){ width=95% }

# Statement of need

Qudit ($d > 2$) quantum information processing is an active field,
motivated by hardware demonstrations across quantum
platforms such as trapped ions [@ringbauer2022], superconducting circuits [@goss2022],
and integrated photonics [@chi2022], and by the advantages for algorithms
and error correction afforded by a larger Hilbert space [@wang2020].
The qutrit ($d = 3$) is the smallest qudit and a natural first step beyond
the qubit. However, most quantum software ecosystems are qubit-centric;
researchers and experimenters who want to prototype qutrit circuits, decompose
single-qutrit unitaries, or reconstruct channels have no dedicated,
dependency-light tool to do so.

`qutritium` fills this gap. It is a single-qutrit-focused package (with
two-qutrit support) whose only required runtime dependency is NumPy
[@harris2020] (matplotlib is optional, for visualization). It brings together
a qutrit gate set, U(3)-to-native-gate decomposition that is exact up to a
global phase, statevector and density-matrix simulation with a configurable
noise model, metrics, and process and state tomography. The package primarily
targets researchers and learners who need to prototype, learn, and implement
qutrit algorithms without a hardware compiler or a large-scale platform. With
automated testing, PyPI installation, and documentation
(<https://spham1611.github.io/qutritium>), this package lowers the barrier to
qutrit simulation.

# State of the field

Several quantum software packages cover parts of the qudit and qutrit workflow.
None combines a qutrit gate set, decomposition, simulation, and
tomography in one lightweight library.

Qiskit [@qiskit] has no native qutrit type. Users must encode everything by hand. There is no established qutrit gate
set,
decomposition, or tomography. Cirq [@cirq] ships a qudit object, `Qid`, that can
represent a qutrit. There is no qutrit library behind it, so users supply the gate
matrices themselves. QuTiP [@johansson2012] is
excellent for the dynamics of open quantum systems, but it has no dedicated
qutrit layer. QuickQudits [@quickqudits] simulates noisy Clifford and Weyl qudit
circuits with stabilizer tableaus. The tableau formalism handles only Clifford
operations. Arbitrary (non-Clifford) single-qutrit unitaries and their
decomposition are outside what it is designed for. MQT Qudits [@mato2024] is a
full framework for qudit circuits of mixed dimension, with a hardware compiler. Its focus is compilation and scalable
simulation, not tomography or
quantum information metrics.

`qutritium` complements these tools rather than competing with them. Its scope
is deliberately small: single qutrits, plus two-qutrit gates. Within that scope
it is complete. Adding the same features to a broader framework would mean
reworking that framework around a single dimension, or accepting a heavy
dependency stack. For teaching and prototyping, a small package that only needs
NumPy is a better fit. The value of `qutritium` is the combination: exact
decomposition from U(3) to native gates, a configurable noise model built from
Kraus operators, and mutually unbiased basis state and process tomography
[@wootters1989], all behind one small and readable API.

# Software design

`qutritium` is a pipeline with three stages: build, execute, and analyze
(\autoref{fig:workflow}). Users compose gates into a `QutritCircuit`, or generate them
with `SU3Decomposition`. They run the circuit on the statevector
simulator (`StatevectorSimulator`) or on the density matrix simulator
(`DensityMatrixSimulator`), which can apply a noise model. The resulting state goes to the tomography and metrics
layers. Each stage is a small object that works on its own. The package works end to end or as individual
building blocks.

Three design choices shape the package. First, NumPy is the only required
runtime dependency. Every gate, channel, and reconstruction is plain dense
linear algebra. This keeps the code easy to read and audit, which matters for a
teaching and reference tool. Results are reproducible without a heavyweight
backend. Second, decomposition works on U(3), which has nine real parameters,
rather than SU(3), which has eight. The global phase is therefore carried
explicitly. The native realization uses G01 and G12 rotations plus virtual Z
rotations and is exact up to that phase, following the SU(3) synthesis of
@vitanov2012. Third, the density matrix simulator stores the full
$3^{n}\times 3^{n}$ operator. Memory therefore grows as $9^{n}$ and gate
application as $27^{n}$. This is deliberate. The simulator is exact and handles
mixed states, but it targets a few qutrits, not large registers.
Synthesis for many qutrits is better served by broader frameworks such as MQT
Qudits [@mato2024].

Noise is handled through Kraus channels. The depolarizing, dephasing, and Pauli
channels are built as mixtures of unitary Weyl operators [@schwinger1960;
@gottesman1998]. An amplitude damping channel and a classical readout error
model are also included. Both coherent and incoherent errors can be simulated. The tomography layer makes one more
choice. State
reconstruction uses linear least squares in the generalized Gell-Mann basis
[@bertlmann2008]. An optional maximum likelihood step projects the estimate onto
the closest physical state [@smolin2012]. Process reconstruction returns a
Hermitian Choi matrix by linear least squares and extracts Kraus operators from
its positive part. The resulting channel is completely positive but not forced
to be trace preserving. The package documents this instead of silently
projecting onto the CPTP set. The estimate stays faithful to the data, and the
user decides whether to enforce physicality.

# Research impact

`qutritium` reproduces known single-qutrit results to quantified accuracy.
\autoref{fig:state_tomography} shows state tomography of a noisy mixed state in
mutually unbiased bases: it recovers both populations and coherences, with
state fidelity $F = 0.983 \pm 0.015$ (seed 17). \autoref{fig:process_tomography}
shows process tomography of a noisy qutrit Hadamard, reconstructed with process
fidelity $F = 0.993 \pm 0.002$ (seed 40). Both benchmarks are generated by a
local script with fixed seeds.

![**Single-qutrit state tomography of a noisy mixed state.** A state is prepared from $|0\rangle$ by rotations
`G01(1.8, 0.7)` and
`G12(1.2, 2.0)`, then passed through a depolarizing channel ($p = 0.15$) via the density-matrix simulator's noise model. This gives a mixed state $\rho$ with purity $\mathrm{Tr}(\rho^2) = 0.82$ (diagonal populations $0.38, 0.41, 0.22$; entries here and in the panels are rounded to two decimals). It is reconstructed by mutually unbiased basis (MUB) tomography: 300 shots in each of the four bases, linear least squares in the generalized Gell-Mann basis, and the Smolin maximum-likelihood projection onto the closest physical state. Heatmaps show the real (top) and imaginary (bottom) parts of the ideal $\rho$ (
**a**, **d**), the reconstruction $\hat{\rho}$ (**b**, **e**), and the difference $\rho - \hat{\rho}$ (**c**, **f
**) on a shared color scale. The reconstruction recovers both populations (diagonal) and coherences (off-diagonal), with purity $0.81$ and residuals $|\rho - \hat{\rho}| \le 0.08$. State fidelity in the squared (Jozsa) convention is $F(\rho,\hat{\rho}) = \left[\mathrm{Tr}\sqrt{\sqrt{\rho}\,\hat{\rho}\,\sqrt{\rho}}\right]^2 = 0.983 \pm 0.015$ (mean $\pm$ s.d., 50 shot noise realizations; realization nearest the mean shown, seed 17).\label{fig:state_tomography}](figures/state_tomography.pdf){
width=85% }

![**Single-qutrit process tomography of a noisy gate.** The qutrit Hadamard
`H3` followed by a depolarizing channel ($p = 0.15$) is characterized: 12 mutually unbiased basis input states are sent through the channel on the density-matrix simulator and reconstructed by state tomography (4000 shots per circuit), and the $9 \times 9$ Choi matrix is recovered by linear least squares (
`reconstruct_process`). Panels (**a**, **b
**) show the magnitude of the ideal and reconstructed Choi matrices (indexed by the two-qutrit basis $|ij\rangle$); panel (
**c**) shows their absolute difference on a separate expanded color scale (residuals $\le 0.016$). Panel (**d
**) shows the recovered Kraus spectrum (the Choi eigenvalues returned by
`choi_to_kraus`), ideal versus reconstructed: one dominant gate operator and eight small depolarizing operators. The true channel is trace-preserving (numerically, $\sum_\ell K_\ell^\dagger K_\ell = I$ to $10^{-15}$); the reconstructed Kraus set, taken from the Choi matrix, is completely positive but not constrained to be trace-preserving. Process fidelity is $F = 0.993 \pm 0.002$ (squared state fidelity between the normalized Choi states $J/d$; mean $\pm$ s.d., 50 shot noise realizations; realization nearest the mean shown, seed 40).\label{fig:process_tomography}](figures/process_tomography.pdf){
width=90% }

The project also has a public track record. It started as a Qiskit Pulse toolkit
for qutrit calibration, now kept in `legacy/`. It was presented at the
Munich Quantum Software Conference in 2023 and was supported by a Unitary Fund
microgrant. It was later rewritten as the hardware-agnostic library described
here, and development has been public since 2023. The package is on PyPI,
versioned, tested across Python 3.10 to 3.14, and documented.

# Acknowledgements

The author thanks Tien Nguyen and Bao Bach for their contributions to the
earlier Qiskit Pulse version of `qutritium`, now kept in `legacy/`, and
Charlie He for his insight on qutrit physics. The original toolkit was
supported by a Unitary Fund microgrant.

# AI usage disclosure

Generative AI tools were used during the development of `qutritium`. Anthropic Claude
(Claude Opus 4.6-4.8 and Claude Fable 5 via CLI) and Cursor assisted with code review, code testing and debugging,
drafting and editing documentation, and maintaining the test suite.
The scientific and software design — including the qutrit gates, circuits, simulators,
decomposition, noise model, tomography, and other essential API functions — was conceived by the author.
All AI-assisted code and text were reviewed, tested, and validated by the author, who
takes full responsibility for their accuracy, originality, and licensing.

# References