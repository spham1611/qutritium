# Contributing to Qutritium

Thanks for taking the time to contribute. Qutritium is a hardware-agnostic
Python library for qutrit quantum computing — bug reports, feature ideas, and
pull requests are all welcome.

## Reporting issues

Please open an issue on the
[GitHub issue tracker](https://github.com/spham1611/qutritium/issues). A useful
bug report includes:

- the qutritium version
  (`python -c "import qutritium; print(qutritium.__version__)"`) and your
  Python / NumPy / SciPy versions,
- a minimal snippet that reproduces the problem,
- what you expected versus what happened, with the full traceback.

## Questions and support

For usage questions, open an issue with the `question` label (or start a
[GitHub Discussion](https://github.com/spham1611/qutritium/discussions) if the
repository has them enabled). You can also reach the maintainer at
`sph40@duke.edu`.

## Development setup

```bash
git clone https://github.com/spham1611/qutritium
cd qutritium
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the checks

All of these should be green before you open a PR:

```bash
pytest test/                 # test suite
ruff check src test          # lint
black --check src test       # formatting
mypy src/qutritium           # type check (non-strict)
```

## Code conventions

- Python 3.10+, with type hints and `from __future__ import annotations`.
- Public computational functions take NumPy-style docstrings; reference code
  identifiers with double backticks (e.g. ``QutritCircuit``). Keep inline
  comments rare.
- Every module ends with an `__all__`, and every new file carries the MIT
  license header.
- Use `numpy.typing.NDArray` for array annotations.
- Keep runtime dependencies minimal — currently only NumPy and SciPy. A new
  runtime dependency needs maintainer sign-off; optional tooling goes in an
  extras group (`[plot]`, `[dev]`, `[docs]`).
- Hardware-specific code (pulse-level control, vendor backends) is intentionally
  out of scope for the core library.

## Pull requests

1. Fork and branch off `main`.
2. Add or update tests for any behavior change, and keep the suite green.
3. Run the checks above.
4. Add a changelog entry (`docs/changelog.md` and `CHANGES.md`) under the next
   version.
5. Open the PR with a clear description of the change and its motivation.

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE.txt).
