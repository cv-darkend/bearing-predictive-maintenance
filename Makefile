.PHONY: setup data indicators manifest marimo notebook notebooks format lint typecheck test check

NOTEBOOK_NAMES := $(basename $(notdir $(wildcard notebooks/*.py)))

setup:
	uv sync

data:
	uv run bearing-download

indicators:
	uv run bearing-indicators

manifest:
	uv run bearing-manifest

marimo:
	uv run marimo edit notebooks

notebook: marimo

notebooks:
	@mkdir -p build/notebooks
	@for nb in $(NOTEBOOK_NAMES); do \
		echo "Exporting $$nb..."; \
		uv run marimo export ipynb notebooks/$$nb.py -o build/notebooks/$$nb.ipynb --include-outputs --sort=top-down || exit 1; \
		uv run marimo export html notebooks/$$nb.py -o build/notebooks/$$nb.html || exit 1; \
	done

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test
