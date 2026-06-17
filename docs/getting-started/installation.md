# Installation

## From PyPI (when available)

```bash
pip install qutritium
```

## From source (development)

```bash
git clone https://github.com/spham1611/qutritium.git
cd qutritium
pip install -e ".[dev,plot]"
```

## Requirements

- Python ≥ 3.10
- numpy ≥ 2.0

Optional:

- `matplotlib ≥ 3.9` for plotting (install with `pip install qutritium[plot]`)
- `pytest`, `ruff`, `mypy` for development (install with `pip install qutritium[dev]`)

## Verify installation

```python
import qutritium
print(qutritium.__version__)
```
