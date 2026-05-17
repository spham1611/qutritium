# Gates API Reference

All gates are imported from `qutritium.gates`:

```python
from qutritium.gates import H3, X01, Rx01, CSUM
```

Every gate inherits from `qutritium.gates.Gate` and provides:

- `.matrix()` → `NDArray[np.complex128]` — the unitary matrix (3×3 or 9×9)
- `.label` → `str` — gate name
- `.num_qutrits` → `int` — 1 or 2
- `.params` → `tuple[float, ...]` — gate parameters (empty for fixed gates)
- `.inverse()` → `Gate` — the conjugate transpose gate
- `.is_unitary()` → `bool` — verify unitarity

## Single-qutrit fixed gates

All matrices are 3×3 unitary.

### Subspace Pauli-X

| Gate    | Action                                  | Matrix                              |
|---------|-----------------------------------------|-------------------------------------|
| `X01()` | $\|0\rangle \leftrightarrow \|1\rangle$ | $\lambda_1 + \|2\rangle\langle 2\|$ |
| `X02()` | $\|0\rangle \leftrightarrow \|2\rangle$ | $\lambda_4 + \|1\rangle\langle 1\|$ |
| `X12()` | $\|1\rangle \leftrightarrow \|2\rangle$ | $\lambda_6 + \|0\rangle\langle 0\|$ |

### Subspace Pauli-Y

| Gate    | Generator                           |
|---------|-------------------------------------|
| `Y01()` | $\lambda_2 + \|2\rangle\langle 2\|$ |
| `Y02()` | $\lambda_5 + \|1\rangle\langle 1\|$ |
| `Y12()` | $\lambda_7 + \|0\rangle\langle 0\|$ |

### Subspace Pauli-Z

| Gate    | Matrix                    |
|---------|---------------------------|
| `Z01()` | $\mathrm{diag}(1, -1, 1)$ |
| `Z02()` | $\mathrm{diag}(1, 1, -1)$ |
| `Z12()` | $\mathrm{diag}(1, 1, -1)$ |

### Other fixed gates

| Gate       | Description                                                       |
|------------|-------------------------------------------------------------------|
| `I3()`     | 3×3 identity                                                      |
| `XPlus()`  | Cyclic shift $\|i\rangle \to \|i{+}1 \bmod 3\rangle$. $X_+^3 = I$ |
| `XMinus()` | Inverse cyclic shift. $(X_+)^\dagger$                             |
| `H3()`     | Qutrit Hadamard / DFT $F_3/\sqrt{3}$                              |
| `S3()`     | $\mathrm{diag}(1, 1, \omega)$. $S^3 = I$                          |
| `T3()`     | $\mathrm{diag}(1, \omega^{1/3}, \omega^{-1/3})$. $T^9 = I$        |
| `UFT()`    | Fourier-related gate $U_{FT}$                                     |

## Single-qutrit parametric gates

### Subspace rotations

Convention: $R_{\text{axis},ij}(\theta) = \exp(-i\theta/2 \cdot \sigma_{\text{axis},ij})$.

| Gate      | Parameters | Subspace                     |
|-----------|------------|------------------------------|
| `Rx01(θ)` | 1          | {$\|0\rangle$, $\|1\rangle$} |
| `Rx02(θ)` | 1          | {$\|0\rangle$, $\|2\rangle$} |
| `Rx12(θ)` | 1          | {$\|1\rangle$, $\|2\rangle$} |
| `Ry01(θ)` | 1          | {$\|0\rangle$, $\|1\rangle$} |
| `Ry02(θ)` | 1          | {$\|0\rangle$, $\|2\rangle$} |
| `Ry12(θ)` | 1          | {$\|1\rangle$, $\|2\rangle$} |
| `Rz01(φ)` | 1          | {$\|0\rangle$, $\|1\rangle$} |
| `Rz02(φ)` | 1          | {$\|0\rangle$, $\|2\rangle$} |
| `Rz12(φ)` | 1          | {$\|1\rangle$, $\|2\rangle$} |

### Generalized rotations

$g_{ij}(\theta, \phi) = \exp\bigl(-i\tfrac{\theta}{2}(\cos\phi\, X_{ij} + \sin\phi\, Y_{ij})\bigr)$

Native gate in trapped-ion implementations (Ringbauer et al., *Nat. Phys.* 18, 1053, 2022).

| Gate        | Parameters |
|-------------|------------|
| `G01(θ, φ)` | 2          |
| `G02(θ, φ)` | 2          |
| `G12(θ, φ)` | 2          |

### Diagonal phase gate

| Gate             | Parameters | Matrix                                                 |
|------------------|------------|--------------------------------------------------------|
| `Ud(φ₁, φ₂, φ₃)` | 3          | $\mathrm{diag}(e^{i\phi_1}, e^{i\phi_2}, e^{i\phi_3})$ |

## Two-qutrit gates

All matrices are 9×9 unitary.

| Gate        | Action                                          | Reference          |
|-------------|-------------------------------------------------|--------------------|
| `CSUM()`    | $\|c, t\rangle \to \|c, (t{+}c) \bmod 3\rangle$ | Wang et al. (2020) |
| `CSUMDag()` | $\|c, t\rangle \to \|c, (t{-}c) \bmod 3\rangle$ | CSUM inverse       |
| `CPhase()`  | $\|c, t\rangle \to \omega^{ct}\|c, t\rangle$    | Qutrit CZ analogue |
| `CNOT3()`   | Legacy CNOT from v0.0.1                         | Equivalent to CSUM |
| `SWAP3()`   | $\|a, b\rangle \to \|b, a\rangle$               | Self-inverse       |

## References

- Bertlmann, R. A. & Krammer, P. (2008). arXiv:0806.1174 — Gell-Mann matrices $\lambda_1$–$\lambda_8$
- Ringbauer, M. et al. (2022). *Nat. Phys.* 18, 1053 — Generalized rotations, T3 gate
- Wang, Y. et al. (2020). *Front. Phys.* 8, 589504 — CSUM, qutrit gate compilation
- Vitanov, N. V. (2012). *Phys. Rev. A* 85, 032331 — SU(3) decomposition
