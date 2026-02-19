.PHONY: setup test lint

setup:
	uv sync --extra test

test:
	uv run pytest tests/

lint:
	uv run ruff check .
