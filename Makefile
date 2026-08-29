.PHONY: help test lint dist clean

help:
	@echo "test   Run the unit tests"
	@echo "lint   Run ruff"
	@echo "dist   Build the installable plugin zip into dist/"
	@echo "clean  Remove build output and caches"

test:
	pytest -q tests/

lint:
	ruff check .

dist:
	./scripts/build-plugin-zip.sh

clean:
	rm -rf dist .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
