# Reproducibility guide

## Supported environment

- Python 3.10, 3.11, or 3.12
- NumPy 1.24 or newer
- SciPy 1.10 or newer
- Matplotlib 3.7 or newer

## Clean installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Verification commands

```bash
pytest
ruff check .
python -m compileall -q src examples
python -m build
```

## Recreate the final audit notebook

```bash
jupyter execute notebooks/10_release_reproducibility_audit.ipynb
```

Alternatively, open the notebook in JupyterLab and run all cells in order.

## Recreate the release manifest

```bash
python scripts/build_release_manifest.py
```

The manifest records the package version and SHA-256 hashes of release-critical
synthetic benchmark files.

## Numerical tolerances

Floating-point results can differ slightly across BLAS implementations and
operating systems. Automated tests use physically meaningful absolute or
relative tolerances rather than exact decimal-string equality.
