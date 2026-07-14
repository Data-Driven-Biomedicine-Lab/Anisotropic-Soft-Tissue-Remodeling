# Contributing

Contributions that improve numerical correctness, test coverage,
documentation, reproducibility, or scientific transparency are welcome.

## Development setup

```bash
git clone https://github.com/Data-Driven-Biomedicine-Lab/Anisotropic-Soft-Tissue-Remodeling.git
cd Anisotropic-Soft-Tissue-Remodeling
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,notebook]"
```

## Before opening a pull request

Run:

```bash
pytest
ruff check .
python -m compileall -q src examples
```

Changes to constitutive equations should include at least one independent
numerical derivative check. Changes to remodeling laws should include tests of
bounds, invariance, and limiting cases. New generated data must be synthetic,
reproducible, and documented with an explicit random seed.

## Scientific claims

Pull requests must distinguish software verification and synthetic validation
from experimental validation. Unsupported biological or clinical claims should
not be added to code, documentation, or release notes.
