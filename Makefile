PY := uv run

.PHONY: install lint test figures notes all clean

install:
	uv sync

lint:
	$(PY) ruff check src tests scripts
	$(PY) ruff format --check src tests scripts
	$(PY) mypy src

test:
	$(PY) pytest -q

figures:
	$(PY) python scripts/make_figures.py --out figures/

notes:
	cd notes && latexmk -pdf -interaction=nonstopmode product.tex && latexmk -c product.tex

all: lint test figures notes

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
