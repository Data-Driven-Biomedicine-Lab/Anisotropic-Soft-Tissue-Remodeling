PYTHON ?= python

.PHONY: install test lint compile build audit manifest

install:
	$(PYTHON) -m pip install -e ".[dev,notebook]"

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

compile:
	$(PYTHON) -m compileall -q src examples

build:
	$(PYTHON) -m build

manifest:
	$(PYTHON) scripts/build_release_manifest.py

audit:
	jupyter execute notebooks/10_release_reproducibility_audit.ipynb
