# Releasing qutritium

This project publishes to PyPI via GitHub Actions Trusted Publishing
([`.github/workflows/release.yml`](.github/workflows/release.yml)). No API
token is kept in the repo or in CI secrets.

## One-time PyPI setup

The `qutritium` PyPI project already exists (currently at `0.0.1`, owned by
`spham1611`). Before the first automated release, attach the GitHub workflow as
a Trusted Publisher:

1. Sign in at <https://pypi.org/manage/account/publishing/>.
2. Pick the `qutritium` project under **Add a new publisher for an existing
   project** and fill in:
    - **Owner:** `spham1611`
    - **Repository name:** `qutritium`
    - **Workflow filename:** `release.yml`
    - **Environment name:** `pypi`
3. Save. The first tagged release after this step completes the binding.

(Same flow on TestPyPI if you ever want a dry-run upload first — point the
workflow's PyPI action at `https://test.pypi.org/legacy/`.)

## Cutting a release

1. **Bump the version** in [`pyproject.toml`](pyproject.toml) (`[project].version`),
   [`src/qutritium/__init__.py`](src/qutritium/__init__.py) (`__version__`),
   and [`CITATION.cff`](CITATION.cff) (`version:` and `date-released:`).
   All three must match. The workflow fails if the tag disagrees with
   `pyproject.toml`.
2. **Update [`CHANGES.md`](CHANGES.md)** with a new section for the version.
3. Commit on `main`, push, and wait for green CI:
   ```bash
   git commit -am "Release vX.Y.Z"
   git push
   ```
4. **Create a GitHub Release in the web UI** — do NOT tag from the command
   line. Releases → "Draft a new release" → type `vX.Y.Z` and pick
   **"Create new tag: vX.Y.Z on publish"**, target `main`. Publishing the
   Release creates the tag (which triggers the `Release` workflow → PyPI)
   *and* fires the Release event that triggers Zenodo archiving. A bare
   `git tag` + `git push --tags` publishes to PyPI but **skips Zenodo** —
   never release that way.
5. The `Release` workflow on GitHub Actions will:
    - verify the tag matches `pyproject.toml`'s version,
    - build sdist + wheel,
    - run `twine check`,
   - upload to PyPI via OIDC (Trusted Publishing) after the `pypi`
     environment deployment is approved.
6. Verify all three outcomes:
   - the `Release` workflow run is green,
   - <https://pypi.org/project/qutritium/> shows the new version,
   - a new Zenodo version record appears under concept DOI
     [10.5281/zenodo.21201868](https://doi.org/10.5281/zenodo.21201868)
     with author = Son Pham only.

## Local dry-run

To produce artifacts locally without uploading:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Inspect the sdist contents to confirm `MANIFEST.in` is doing its job:

```bash
python -c "import tarfile; [print(m.name) for m in tarfile.open('dist/qutritium-X.Y.Z.tar.gz').getmembers()]"
```

The sdist should contain only `src/qutritium/`, top-level metadata files
(`README.md`, `LICENSE.txt`, `CITATION.cff`, `CHANGES.md`, `pyproject.toml`,
`MANIFEST.in`), and `PKG-INFO`. It should NOT contain `legacy/`, `test/`,
`docs/`, `examples/`, or `.github/`.

## Rollback / yank

If a broken release reaches PyPI, **yank** it (does not delete; flags the
release so `pip install` skips it but pinned installs still work):

```bash
# via the web UI: https://pypi.org/manage/project/qutritium/release/X.Y.Z/
```

Then cut a patch release with the fix.
